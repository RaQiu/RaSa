"""
unified_modality_grad_modulator.py
===================================
Unified Cross-Modal Adaptive Gradient Modulation Plugin.

A single plugin that reproduces BOTH the IRRA and RaSa AGM behaviors exactly,
configurable via hyperparameters. Use the preset class methods for zero-difference
equivalence with either implementation.

Core Algorithm (shared):
    1. Capture activations from 3 forward passes (normal, e_img, e_txt)
    2. Per-step: modal specificity detection via activation deltas
    3. Per-step: loss delta accumulation
    4. Half-epoch: DDP sync -> score computation -> pen vector -> gradient modulation
       -> encoder-level suppression -> reset accumulators

Lifecycle::

    config = UnifiedModulationConfig.irra_preset()   # or .rasa_preset()
    modulator = UnifiedModalityGradModulator(config)

    # IRRA-style: plugin manages hooks
    modulator.attach(model,
        shared_filter=..., img_enc_filter=..., txt_enc_filter=...)

    # RaSa-style: model manages hooks
    modulator.attach(model,
        model_activation_attr="ACTIVATIONS",
        model_belong_key="CM",
        img_enc_prefixes=("visual", "vision"),
        txt_enc_prefixes=("text",))

    for epoch:
        modulator.on_epoch_start(model)
        for step, batch in loader:
            modulator.pre_forward(model)          # no-op in model-managed mode
            ret = model(batch)
            modulator.capture('normal', model)
            ...  # e_img, e_txt forward passes + captures
            loss.backward()
            modulator.post_backward(model, step, data_len, batch_size,
                                    all_loss, e_txt_loss, e_img_loss)
            optimizer.step()
        stats = modulator.on_epoch_end(model, epoch)
"""

import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

logger = logging.getLogger("UnifiedModalityGradModulator")


# ======================================================================
# Helpers
# ======================================================================

def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP / FSDP model."""
    return model.module if hasattr(model, 'module') else model


# Type alias
ModuleFilter = Callable[[str, nn.Module], bool]


class HookMode(str, Enum):
    """How activations are captured."""
    PLUGIN = "plugin"   # Plugin registers forward hooks (IRRA-style)
    MODEL = "model"     # Model manages hooks, plugin reads model.ACTIVATIONS (RaSa-style)


class DDPScoreMode(str, Enum):
    """How DDP score computation is done."""
    ALL_COMPUTE = "all_compute"       # Every rank computes score (IRRA-style)
    RANK0_BROADCAST = "rank0_broadcast"  # Rank-0 computes, broadcasts (RaSa-style)


class LossReduction(str, Enum):
    """How multi-component losses are reduced before delta computation."""
    IDENTITY = "identity"  # Loss is already a scalar
    SUM = "sum"            # torch.sum() to collapse loss vector


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class UnifiedModulationConfig:
    """All hyperparameters for unified gradient modulation.

    Use irra_preset() or rasa_preset() for zero-difference equivalence.
    """
    # --- Core ---
    enabled: bool = True
    gammb: float = 1.65
    tau: int = 1                          # Modal specificity check interval (steps)
    softmax_temperature: float = 1.0      # Applied to score before softmax

    # --- Sigmoid stability modulation (Eqn.9, IRRA-specific) ---
    sigmoid_enabled: bool = False
    gaeta: float = 1.0                    # Activation threshold (half-epoch rounds)
    alpha_sigmoid: float = 0.9            # Sigmoid steepness
    beta_sigmoid: float = 0.999           # Training progress decay rate

    # --- Pen vector ---
    clamp_pen: bool = False               # Clamp pen to [0, 1]

    # --- DDP ---
    ddp_score_mode: DDPScoreMode = DDPScoreMode.ALL_COMPUTE

    # --- Loss ---
    loss_reduction: LossReduction = LossReduction.IDENTITY

    # --- Hook management ---
    hook_mode: HookMode = HookMode.PLUGIN

    # --- Tracking features ---
    grad_ratio_tracking: bool = False
    fig1c_enabled: bool = False
    fig1c_noise_std: float = 0.1
    output_dir: str = "./logs"

    @classmethod
    def irra_preset(cls, **overrides):
        """Configuration that reproduces IRRA AGM behavior exactly."""
        defaults = dict(
            gammb=1.65,
            tau=1,
            softmax_temperature=1.0,
            sigmoid_enabled=True,
            gaeta=1.0,
            alpha_sigmoid=0.9,
            beta_sigmoid=0.999,
            clamp_pen=True,
            ddp_score_mode=DDPScoreMode.ALL_COMPUTE,
            loss_reduction=LossReduction.IDENTITY,
            hook_mode=HookMode.PLUGIN,
            grad_ratio_tracking=True,
            fig1c_enabled=True,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def rasa_preset(cls, **overrides):
        """Configuration that reproduces RaSa AGM behavior exactly."""
        defaults = dict(
            gammb=2.75,
            tau=1,
            softmax_temperature=0.1,
            sigmoid_enabled=False,
            clamp_pen=False,
            ddp_score_mode=DDPScoreMode.RANK0_BROADCAST,
            loss_reduction=LossReduction.SUM,
            hook_mode=HookMode.MODEL,
            grad_ratio_tracking=False,
            fig1c_enabled=False,
        )
        defaults.update(overrides)
        return cls(**defaults)


# ======================================================================
# Main Plugin
# ======================================================================

class UnifiedModalityGradModulator:
    """
    Unified cross-modal adaptive gradient modulation plugin.

    Supports both IRRA and RaSa patterns via configuration.
    """

    def __init__(self, config: UnifiedModulationConfig):
        self.config = config

        # ---- Module registries (for plugin-managed hook mode) ----
        self._shared_modules: Dict[str, nn.Module] = {}      # name -> module
        self._img_enc_modules: Dict[str, nn.Module] = {}
        self._txt_enc_modules: Dict[str, nn.Module] = {}
        self._module_to_name: Dict[int, str] = {}            # id(module) -> name
        self._module_by_id: Dict[int, nn.Module] = {}        # id(module) -> module

        # ---- Model-managed hook mode state ----
        self._model_activation_attr: str = "ACTIVATIONS"
        self._model_belong_key: Optional[str] = None
        self._img_enc_prefixes: Tuple[str, ...] = ()
        self._txt_enc_prefixes: Tuple[str, ...] = ()

        # ---- Hook state (plugin-managed mode) ----
        self._activations: Dict[int, torch.Tensor] = {}
        self._handles: List = []

        # ---- Captured activation snapshots ----
        # In plugin mode: {key: {name: tensor}}
        # In model mode:  {key: {module_key: (module, tensor, name)}}
        self._captured: Dict[str, dict] = {}

        # ---- Modal specificity counts (key = canonical name or module_iid) ----
        self._modal_img_counts: Dict[str, torch.Tensor] = {}
        self._modal_txt_counts: Dict[str, torch.Tensor] = {}

        # ---- Sigmoid stability state (IRRA) ----
        self._modal_img_tot: Dict[str, torch.Tensor] = {}
        self._modal_txt_tot: Dict[str, torch.Tensor] = {}
        self._diff: Dict[str, torch.Tensor] = {}
        self._last: Dict[str, torch.Tensor] = {}
        self._times: int = 0

        # ---- Loss accumulators ----
        # Use 0 (int) as initial value to match IRRA's original behavior.
        # RaSa uses torch.zeros(1) on device; the tensor arithmetic works either way.
        self._delta_img_loss = 0
        self._delta_txt_loss = 0
        self._tot = 0

        # ---- LAYER_REC: module_iid -> name (RaSa compat) ----
        self._layer_rec: Dict[str, str] = {}

        # ---- Gradient ratio tracking (IRRA) ----
        self._grad_ratio: Dict[str, List] = {}
        self._grad_ratio_stats: Dict[str, List[float]] = {
            "mean": [], "max": [], "min": [],
        }

        # ---- Fig1(c) state (IRRA) ----
        self._fig1c_data: Dict[str, list] = {
            "epoch_list": [],
            "text_no_noise": [],
            "text_with_noise": [],
            "img_no_noise": [],
            "img_with_noise": [],
        }
        self._grad_cache: Dict[str, List[float]] = {
            "text_no_noise": [],
            "text_with_noise": [],
            "img_no_noise": [],
            "img_with_noise": [],
        }
        self._noise_experiment_done: bool = False

        # ---- Logger info (RaSa compat) ----
        self._logger_info: str = ""

    # ==================================================================
    # Public API
    # ==================================================================

    def attach(
        self,
        model: nn.Module,
        # --- Plugin-managed hook mode (IRRA-style) ---
        shared_filter: Optional[ModuleFilter] = None,
        img_enc_filter: Optional[ModuleFilter] = None,
        txt_enc_filter: Optional[ModuleFilter] = None,
        # --- Model-managed hook mode (RaSa-style) ---
        model_activation_attr: str = "ACTIVATIONS",
        model_belong_key: Optional[str] = None,
        img_enc_prefixes: Tuple[str, ...] = (),
        txt_enc_prefixes: Tuple[str, ...] = (),
    ):
        """Register module categories and initialize state.

        For IRRA-style (hook_mode=PLUGIN):
            Provide shared_filter, img_enc_filter, txt_enc_filter.
            Plugin manages forward hooks.

        For RaSa-style (hook_mode=MODEL):
            Provide model_activation_attr, model_belong_key,
            img_enc_prefixes, txt_enc_prefixes.
            Model manages hooks; plugin reads model attributes.
        """
        base = _unwrap(model)

        if self.config.hook_mode == HookMode.PLUGIN:
            # IRRA-style: scan model and register module categories
            self._shared_modules.clear()
            self._img_enc_modules.clear()
            self._txt_enc_modules.clear()
            self._module_to_name.clear()
            self._module_by_id.clear()

            for name, module in base.named_modules():
                is_leaf = not list(module.children())
                has_params = bool(list(module.parameters()))
                if not (is_leaf and has_params):
                    continue
                mid = id(module)
                if shared_filter and shared_filter(name, module):
                    self._shared_modules[name] = module
                    self._module_to_name[mid] = name
                    self._module_by_id[mid] = module
                if img_enc_filter and img_enc_filter(name, module):
                    self._img_enc_modules[name] = module
                    self._module_to_name[mid] = name
                if txt_enc_filter and txt_enc_filter(name, module):
                    self._txt_enc_modules[name] = module
                    self._module_to_name[mid] = name

            logger.info(
                "Attached (plugin mode): %d shared, %d img_enc, %d txt_enc",
                len(self._shared_modules),
                len(self._img_enc_modules),
                len(self._txt_enc_modules),
            )

        else:  # MODEL mode
            self._model_activation_attr = model_activation_attr
            self._model_belong_key = model_belong_key
            self._img_enc_prefixes = img_enc_prefixes
            self._txt_enc_prefixes = txt_enc_prefixes

            # Pre-scan encoder modules for suppression
            self._img_enc_modules.clear()
            self._txt_enc_modules.clear()
            for name, module in base.named_modules():
                is_leaf = not list(module.children())
                has_params = bool(list(module.parameters()))
                if not (is_leaf and has_params):
                    continue
                if img_enc_prefixes and name.startswith(img_enc_prefixes):
                    self._img_enc_modules[name] = module
                if txt_enc_prefixes and name.startswith(txt_enc_prefixes):
                    self._txt_enc_modules[name] = module

            # Initialize loss accumulators on device (RaSa uses tensor accumulators)
            device = next(model.parameters()).device
            self._delta_img_loss = torch.zeros(1, device=device)
            self._delta_txt_loss = torch.zeros(1, device=device)
            self._tot = torch.zeros(1, device=device)

            logger.info(
                "Attached (model mode): %d img_enc, %d txt_enc",
                len(self._img_enc_modules),
                len(self._txt_enc_modules),
            )

    def on_epoch_start(self, model: nn.Module):
        """Called at the start of each training epoch."""
        self._noise_experiment_done = False

        if self.config.hook_mode == HookMode.MODEL:
            # Build LAYER_REC mapping (RaSa compat)
            base = _unwrap(model)
            self._layer_rec = {}
            for name, module in base.named_modules():
                if not list(module.children()) and list(module.parameters()):
                    module_iid = f"{str(repr(module.__class__))}_{id(module)}"
                    self._layer_rec[module_iid] = name

    def pre_forward(self, model: nn.Module):
        """Register forward hooks (plugin-managed mode only).

        Call ONCE before the first of the three forward passes.
        No-op in model-managed mode.
        """
        if self.config.hook_mode != HookMode.PLUGIN:
            return

        self._handles = []
        for module in self._shared_modules.values():
            self._handles.append(module.register_forward_hook(self._hook_fn))

    def capture(self, key: str, model: nn.Module = None):
        """Snapshot current activations under *key*.

        In plugin mode: reads from internal hooks.
        In model mode: reads from model's activation attribute.

        Call after each forward pass::
            capture('normal', model)
            capture('e_img', model)
            capture('e_txt', model)
        """
        if self.config.hook_mode == HookMode.PLUGIN:
            # Plugin-managed: activations keyed by id(module) -> normalize to name
            snapshot = {}
            for mid, tensor in self._activations.items():
                name = self._module_to_name.get(mid)
                if name is not None:
                    snapshot[name] = tensor
            self._activations.clear()
            self._captured[key] = snapshot

        else:  # MODEL mode
            base = _unwrap(model)
            raw_acts = getattr(base, self._model_activation_attr, {})
            # Store as {module_ref: tensor} — needed for BELONG check later
            self._captured[key] = raw_acts.copy()
            setattr(base, self._model_activation_attr, {})

    def post_backward(
        self,
        model: nn.Module,
        step: int,
        data_len: int,
        batch_size: int,
        all_loss,
        e_txt_loss,
        e_img_loss,
    ):
        """Gradient modulation. Call after loss.backward(), before optimizer.step()."""

        # Fig1(c) experiment (IRRA feature)
        if self.config.fig1c_enabled and not self._noise_experiment_done:
            self._fig1c_noise_experiment(model)
            self._noise_experiment_done = True

        if not self.config.enabled:
            self._remove_hooks()
            self._captured.clear()
            return

        if self.config.hook_mode == HookMode.PLUGIN:
            self._modulate_plugin_mode(
                model, step, data_len, batch_size,
                all_loss, e_txt_loss, e_img_loss,
            )
        else:
            self._modulate_model_mode(
                model, step, data_len, batch_size,
                all_loss, e_txt_loss, e_img_loss,
            )

        self._captured.clear()

    def on_epoch_end(self, model: nn.Module, epoch: int):
        """Finalize epoch. Returns stats dict (IRRA) or logger info string (RaSa).

        The return type depends on hook_mode for backward compatibility:
        - PLUGIN mode: returns dict with cnt_txt, cnt_img, rho_num, grad_ratio_mean
        - MODEL mode: returns str (accumulated log info)
        """
        if self.config.hook_mode == HookMode.PLUGIN:
            return self._on_epoch_end_plugin(model, epoch)
        else:
            return self._on_epoch_end_model(model, epoch)

    # ==================================================================
    # Internal: Plugin-Managed Mode (IRRA-compatible)
    # ==================================================================

    def _hook_fn(self, module: nn.Module, input, output):
        """Forward hook: capture input[0]."""
        self._activations[id(module)] = input[0].detach()

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _modulate_plugin_mode(
        self, model, step, data_len, batch_size,
        all_loss, e_txt_loss, e_img_loss,
    ):
        """Core modulation for plugin-managed hook mode (IRRA path)."""
        base = _unwrap(model)
        base.eval()
        cfg = self.config

        # Reduce loss if needed
        all_l = self._reduce_loss(all_loss)
        e_txt_l = self._reduce_loss(e_txt_loss)
        e_img_l = self._reduce_loss(e_img_loss)

        # Accumulate loss deltas
        _ensure = lambda v: v if torch.is_tensor(v) else torch.tensor(v, device="cuda")
        self._delta_img_loss = self._delta_img_loss + (_ensure(e_img_l) - _ensure(all_l))
        self._delta_txt_loss = self._delta_txt_loss + (_ensure(e_txt_l) - _ensure(all_l))
        self._tot += 1

        all_act = self._captured.get('normal', {})
        e_txt_act = self._captured.get('e_txt', {})
        e_img_act = self._captured.get('e_img', {})

        with torch.no_grad():
            # Per-step: modal specificity check
            if step % cfg.tau == 0:
                for name in all_act:
                    if name not in e_img_act or name not in e_txt_act:
                        continue
                    act = all_act[name]
                    self._modal_specificity_check(
                        name, act, e_img_act[name], e_txt_act[name],
                        batch_size, act.device,
                    )

            # Half-epoch: gradient modulation
            if (step + 1) % (data_len // 2) == 0:
                self._times += 1
                self._sync_distributed_plugin()

                # Lazy-init per-module state (sigmoid)
                if cfg.sigmoid_enabled:
                    for name, act in all_act.items():
                        num_neurons = act.size(-1)
                        if name not in self._modal_img_tot:
                            self._modal_img_tot[name] = torch.zeros(num_neurons, device=act.device)
                            self._modal_txt_tot[name] = torch.zeros(num_neurons, device=act.device)
                        if name not in self._diff:
                            self._diff[name] = torch.zeros(num_neurons, device=act.device)

                # Score
                iscore_txt, iscore_img = self._compute_scores_plugin()

                # Pen vector + gradient modification
                for name in all_act:
                    module = self._shared_modules.get(name)
                    if module is None or name not in self._modal_img_counts:
                        continue

                    act = all_act[name]
                    pen = self._build_pen_vector(
                        name, act, iscore_txt, iscore_img,
                    )
                    self._apply_pen_to_module(name, module, pen)

                # Encoder-level suppression (exclude shared modules)
                shared_ids = {id(m) for m in self._shared_modules.values()}
                for _name, module in self._img_enc_modules.items():
                    if id(module) not in shared_ids:
                        for param in module.parameters():
                            if param.grad is not None:
                                param.grad *= (1 - iscore_img)

                for _name, module in self._txt_enc_modules.items():
                    if id(module) not in shared_ids:
                        for param in module.parameters():
                            if param.grad is not None:
                                param.grad *= (1 - iscore_txt)

                # Reset loss accumulators
                self._delta_img_loss = 0
                self._delta_txt_loss = 0
                self._tot = 0

        self._remove_hooks()
        base.train()

    def _compute_scores_plugin(self):
        """Compute iscore_txt, iscore_img (IRRA path: all ranks compute)."""
        cfg = self.config
        score = torch.tensor(
            [self._delta_txt_loss / self._tot,
             self._delta_img_loss / self._tot],
            device="cuda",
        )
        ratio = F.softmax(cfg.softmax_temperature * score, dim=0)
        r_min, _ = torch.min(ratio, dim=0)
        iscore = (ratio - r_min) ** cfg.gammb
        iscore_txt = iscore[0].item()
        iscore_img = iscore[1].item()

        logger.info(
            "Step delta_txt_loss = %s , delta_img_loss = %s",
            self._delta_txt_loss, self._delta_img_loss,
        )
        logger.info("iscore_txt = %s, iscore_img = %s", iscore_txt, iscore_img)
        return iscore_txt, iscore_img

    def _sync_distributed_plugin(self):
        """DDP sync for plugin mode (IRRA path: all-reduce everything)."""
        if not (_DIST_AVAILABLE and dist.is_initialized()):
            return
        if dist.get_world_size() <= 1:
            return

        for name in list(self._modal_img_counts.keys()):
            dist.all_reduce(self._modal_img_counts[name], op=dist.ReduceOp.SUM)
            dist.all_reduce(self._modal_txt_counts[name], op=dist.ReduceOp.SUM)

        if self.config.sigmoid_enabled:
            for name in list(self._diff.keys()):
                dist.all_reduce(self._diff[name], op=dist.ReduceOp.SUM)

        _val = lambda v: v.item() if torch.is_tensor(v) else float(v)
        sync_buf = torch.tensor(
            [_val(self._delta_txt_loss), _val(self._delta_img_loss), float(self._tot)],
            device="cuda",
        )
        dist.all_reduce(sync_buf, op=dist.ReduceOp.SUM)
        self._delta_txt_loss = sync_buf[0]
        self._delta_img_loss = sync_buf[1]
        self._tot = int(sync_buf[2].item())

    def _on_epoch_end_plugin(self, model, epoch):
        """Epoch end for plugin mode (IRRA path)."""
        stats: dict = {}

        if self.config.fig1c_enabled:
            self._save_fig1c_data(epoch)

        cnt_txt, cnt_img, equal = 0, 0, 0
        for name in list(self._modal_txt_counts.keys()):
            if name not in self._modal_img_counts:
                continue
            mtxt = (self._modal_txt_counts[name] > self._modal_img_counts[name]).sum().item()
            mimg = (self._modal_img_counts[name] > self._modal_txt_counts[name]).sum().item()
            meq = (self._modal_img_counts[name] == self._modal_txt_counts[name]).sum().item()
            logger.info("%s : cnt_txt: %d, cnt_img: %d, equal: %d", name, mtxt, mimg, meq)
            cnt_txt += mtxt
            cnt_img += mimg
            equal += meq

        epsilon = 1e-8
        rho_num = cnt_img / (cnt_txt + epsilon)
        logger.info("[ALL] : cnt_txt: %d, cnt_img: %d, equal: %d", cnt_txt, cnt_img, equal)
        logger.info("[ALL] : rho_num: %.4f", rho_num)
        stats.update(cnt_txt=cnt_txt, cnt_img=cnt_img, rho_num=rho_num)

        if self._grad_ratio_stats["mean"]:
            avg_mean = sum(self._grad_ratio_stats["mean"]) / len(self._grad_ratio_stats["mean"])
            avg_max = sum(self._grad_ratio_stats["max"]) / len(self._grad_ratio_stats["max"])
            avg_min = sum(self._grad_ratio_stats["min"]) / len(self._grad_ratio_stats["min"])
            logger.info(
                "[Epoch %d] Grad Ratio Stats - Mean: %.4f, Max: %.4f, Min: %.4f",
                epoch, avg_mean, avg_max, avg_min,
            )
            stats['grad_ratio_mean'] = avg_mean

        self._grad_ratio_stats = {"mean": [], "max": [], "min": []}
        self._modal_img_counts.clear()
        self._modal_txt_counts.clear()

        return stats

    # ==================================================================
    # Internal: Model-Managed Mode (RaSa-compatible)
    # ==================================================================

    def _modulate_model_mode(
        self, model, step, data_len, batch_size,
        all_loss, e_txt_loss, e_img_loss,
    ):
        """Core modulation for model-managed hook mode (RaSa path)."""
        model.eval()
        cfg = self.config
        base = _unwrap(model)
        device = next(model.parameters()).device

        all_act = self._captured.get('normal', {})
        e_txt_act = self._captured.get('e_txt', {})
        e_img_act = self._captured.get('e_img', {})

        # Reduce loss
        all_l = self._reduce_loss(all_loss)
        e_txt_l = self._reduce_loss(e_txt_loss)
        e_img_l = self._reduce_loss(e_img_loss)

        # Loss accumulation
        self._delta_img_loss = self._delta_img_loss + e_img_l - all_l
        self._delta_txt_loss = self._delta_txt_loss + e_txt_l - all_l
        self._tot = self._tot + 1

        # Modal specificity detection (every tau steps)
        with torch.no_grad():
            if step % cfg.tau == 0:
                for module in all_act.keys():
                    module_iid = f"{str(repr(module.__class__))}_{id(module)}"

                    delta_img = torch.abs(all_act[module] - e_img_act[module])
                    if delta_img.size(0) != batch_size:
                        delta_img = delta_img.permute(1, 0, 2)
                    delta_img = torch.mean(torch.mean(delta_img, dim=1), dim=0)

                    delta_txt = torch.abs(all_act[module] - e_txt_act[module])
                    if delta_txt.size(0) != batch_size:
                        delta_txt = delta_txt.permute(1, 0, 2)
                    delta_txt = torch.mean(torch.mean(delta_txt, dim=1), dim=0)

                    txt_specificity = delta_txt - delta_img
                    img_specificity = delta_img - delta_txt
                    num_neurons = delta_img.size(-1)
                    indicate = torch.ones(num_neurons, device=device) * -1

                    img_mask = (img_specificity > 0)
                    txt_mask = (txt_specificity > 0)
                    both_zero_mask = (img_specificity == 0) & (txt_specificity == 0)
                    indicate = indicate.masked_fill(img_mask, 1)
                    indicate = indicate.masked_fill(txt_mask, 0)
                    indicate = indicate.masked_fill(both_zero_mask, random.randint(0, 1))

                    if module_iid not in self._modal_img_counts and module_iid not in self._modal_txt_counts:
                        self._modal_img_counts[module_iid] = torch.ones(num_neurons, device=device) * -1
                        self._modal_txt_counts[module_iid] = torch.ones(num_neurons, device=device) * -1

                    self._modal_img_counts[module_iid] = self._modal_img_counts[module_iid] + (indicate == 1).float()
                    self._modal_txt_counts[module_iid] = self._modal_txt_counts[module_iid] + (indicate == 0).float()

        # Half-epoch gradient modification
        if (step + 1) % (data_len // 2) == 0 and dist.is_initialized():
            # DDP sync
            dist.all_reduce(self._delta_img_loss)
            dist.all_reduce(self._delta_txt_loss)
            dist.all_reduce(self._tot)
            for module_iid in self._modal_txt_counts:
                dist.all_reduce(self._modal_txt_counts[module_iid])
                dist.all_reduce(self._modal_img_counts[module_iid])

            # Rank-0 computes score, then broadcast
            if dist.get_rank() == 0:
                score_cm = torch.tensor([
                    self._delta_txt_loss / self._tot,
                    self._delta_img_loss / self._tot
                ], device=device)
                ratio_cm = F.softmax(cfg.softmax_temperature * score_cm, dim=0)
            else:
                score_cm = torch.zeros(2, device=device)
                ratio_cm = torch.zeros(2, device=device)
            dist.broadcast(score_cm, src=0)
            dist.broadcast(ratio_cm, src=0)

            r_min, _ = torch.min(ratio_cm, dim=0)
            iscore = (ratio_cm - r_min) ** cfg.gammb
            iscore_txt, iscore_img = iscore[0].item(), iscore[1].item()

            print("Step {}, delta_txt_loss_cm = {} , delta_img_loss_cm = {}".format(
                step, self._delta_txt_loss, self._delta_img_loss))
            print("iscore_txt_cm = {}, iscore_img_cm = {}".format(iscore_txt, iscore_img))

            # Pen vector construction & gradient modification
            already = set()
            belong_set = set()
            if self._model_belong_key:
                belong_dict = getattr(base, 'BELONG', {})
                belong_set = belong_dict.get(self._model_belong_key, set())

            for module in all_act.keys():
                if module in belong_set:
                    already.add(id(module))
                    pen = torch.zeros(all_act[module].size(-1)).cuda()
                    module_iid = f"{str(repr(module.__class__))}_{id(module)}"
                    pen_img = (self._modal_img_counts[module_iid] > self._modal_txt_counts[module_iid])
                    pen += pen_img * iscore_img
                    pen_txt = (self._modal_img_counts[module_iid] < self._modal_txt_counts[module_iid])
                    pen += pen_txt * iscore_txt

                    # Optional sigmoid stability modulation
                    if cfg.sigmoid_enabled and self._times >= cfg.gaeta:
                        name = self._layer_rec.get(module_iid, module_iid)
                        if module_iid not in self._diff:
                            num_neurons = all_act[module].size(-1)
                            self._diff[module_iid] = torch.zeros(num_neurons, device=pen.device)
                        k = max(iscore_img, iscore_txt) / 10
                        n_tensor = torch.tensor(self._times, dtype=torch.float32, device=pen.device)
                        R = cfg.gaeta * self._diff[module_iid] / self._times
                        f_R = 2.0 / (1.0 + torch.exp(-cfg.alpha_sigmoid * (R - 1.0))) - 1.0
                        g_n = 1.0 / (1.0 + cfg.beta_sigmoid * n_tensor)
                        delta_z = k * f_R * g_n
                        pen = torch.where(pen != 0, pen + delta_z, pen)

                    pen = 1 - pen

                    if cfg.clamp_pen:
                        pen = torch.clamp(pen, min=0, max=1)

                    for param in module.parameters():
                        if param.grad is not None:
                            if len(param.grad.size()) > 1 and param.grad.size()[1] == pen.size():
                                param.grad *= pen.unsqueeze(0)
                            elif len(param.grad.size()) == 1 and param.grad.size() == pen.size():
                                param.grad *= pen

            # Encoder-level suppression
            for name, module in base.named_modules():
                if ((not list(module.children())) and
                        name.startswith(self._txt_enc_prefixes) and
                        (id(module) not in already) and
                        list(module.parameters())):
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_txt)

            for name, module in base.named_modules():
                if ((not list(module.children())) and
                        name.startswith(self._img_enc_prefixes) and
                        (id(module) not in already) and
                        list(module.parameters())):
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_img)

            del pen
            torch.cuda.empty_cache()

            # Reset BELONG set
            if self._model_belong_key:
                belong_dict = getattr(base, 'BELONG', {})
                belong_dict[self._model_belong_key] = set()

            # Reset loss accumulators
            self._tot = torch.zeros(1, device=device)
            self._delta_txt_loss = torch.zeros(1, device=device)
            self._delta_img_loss = torch.zeros(1, device=device)

            # Log and reset modal counts
            cnt_txt, cnt_img, equal = 0, 0, 0
            for module_iid in self._modal_txt_counts.keys():
                module_txt = (self._modal_txt_counts[module_iid] > self._modal_img_counts[module_iid]).sum().item()
                module_img = (self._modal_img_counts[module_iid] > self._modal_txt_counts[module_iid]).sum().item()
                module_equal = (self._modal_img_counts[module_iid] == self._modal_txt_counts[module_iid]).sum().item()
                self._logger_info += "{} : cnt_txt: {}, cnt_img: {}, equal: {}\n".format(
                    self._layer_rec.get(module_iid, module_iid), module_txt, module_img, module_equal)
                cnt_txt += module_txt
                cnt_img += module_img
                equal += module_equal
            self._logger_info += "[Step{} ALL] : cnt_txt: {}, cnt_img: {},  equal: {}\n".format(
                step, cnt_txt, cnt_img, equal)
            self._modal_img_counts = {}
            self._modal_txt_counts = {}

        # Cleanup
        del delta_img, delta_txt, indicate
        torch.cuda.empty_cache()
        self._captured = {}

        model.train()

    def _on_epoch_end_model(self, model, epoch):
        """Epoch end for model-managed mode (RaSa path). Returns logger info string."""
        device = next(model.parameters()).device
        info = self._logger_info
        self._logger_info = ""

        self._modal_txt_counts = {}
        self._modal_img_counts = {}
        self._tot = torch.zeros(1, device=device)
        self._delta_txt_loss = torch.zeros(1, device=device)
        self._delta_img_loss = torch.zeros(1, device=device)

        return info

    # ==================================================================
    # Internal: Shared helpers
    # ==================================================================

    def _reduce_loss(self, loss):
        """Apply loss reduction according to config."""
        if self.config.loss_reduction == LossReduction.SUM:
            return torch.sum(loss) if torch.is_tensor(loss) else loss
        return loss

    def _modal_specificity_check(
        self, name, act, e_img_act, e_txt_act, batch_size, device,
    ):
        """Shared modal specificity detection logic (plugin mode, name-keyed)."""
        delta_img = torch.abs(act - e_img_act)
        if delta_img.size(0) != batch_size:
            delta_img = delta_img.permute(1, 0, 2)
        delta_img = torch.mean(torch.mean(delta_img, dim=1), dim=0)

        delta_txt = torch.abs(act - e_txt_act)
        if delta_txt.size(0) != batch_size:
            delta_txt = delta_txt.permute(1, 0, 2)
        delta_txt = torch.mean(torch.mean(delta_txt, dim=1), dim=0)

        txt_specificity = delta_txt - delta_img
        img_specificity = delta_img - delta_txt
        num_neurons = delta_img.size(-1)

        indicate = torch.ones(num_neurons, device=device) * -1
        indicate[img_specificity > 0] = 1
        indicate[txt_specificity > 0] = 0
        tied = (img_specificity == 0) & (txt_specificity == 0)
        indicate[tied] = random.randint(0, 1)

        if name not in self._modal_img_counts:
            self._modal_img_counts[name] = torch.ones(num_neurons, device=device) * -1
            self._modal_txt_counts[name] = torch.ones(num_neurons, device=device) * -1

        self._modal_img_counts[name] += (indicate == 1)
        self._modal_txt_counts[name] += (indicate == 0)

    def _build_pen_vector(self, name, act, iscore_txt, iscore_img):
        """Construct pen vector for a module (plugin mode)."""
        cfg = self.config
        num_neurons = act.size(-1)
        pen = torch.zeros(num_neurons, device=act.device)

        # Track oscillation (sigmoid mode)
        if cfg.sigmoid_enabled:
            now = torch.zeros(num_neurons, device=act.device)
            now += (self._modal_img_counts[name] > self._modal_txt_counts[name]).to(torch.int)
            if name in self._last:
                self._diff[name] = self._diff[name] + (now != self._last[name]).to(torch.int)

        pen_img = (self._modal_img_counts[name] > self._modal_txt_counts[name])
        pen += pen_img * iscore_img
        pen_txt = (self._modal_img_counts[name] < self._modal_txt_counts[name])
        pen += pen_txt * iscore_txt

        # Sigmoid stability modulation (Eqn.9)
        if cfg.sigmoid_enabled and self._times >= cfg.gaeta:
            k = max(iscore_img, iscore_txt) / 10
            n_tensor = torch.tensor(self._times, dtype=torch.float32, device="cuda")
            R = cfg.gaeta * self._diff[name] / self._times
            f_R = 2.0 / (1.0 + torch.exp(-cfg.alpha_sigmoid * (R - 1.0))) - 1.0
            g_n = 1.0 / (1.0 + cfg.beta_sigmoid * n_tensor)
            delta_z = k * f_R * g_n
            pen = torch.where(pen != 0, pen + delta_z, pen)

            print("DIFF", torch.max(self._diff[name]), torch.min(self._diff[name]))
            print("Delta_z", torch.max(delta_z), torch.min(delta_z))
            print("@Instruct Before", torch.max(pen), torch.min(pen))

        pen = 1 - pen

        if cfg.sigmoid_enabled:
            print("@Instruct After", torch.max(pen), torch.min(pen))

        if cfg.clamp_pen:
            pen = torch.clamp(pen, min=0, max=1)

        # Update last state (sigmoid)
        if cfg.sigmoid_enabled:
            self._last[name] = now

        return pen

    def _apply_pen_to_module(self, name, module, pen):
        """Apply pen vector to module gradients, with optional grad_ratio tracking."""
        for param in module.parameters():
            if param.grad is None:
                continue

            grad_before = param.grad.clone() if self.config.grad_ratio_tracking else None

            if len(param.grad.size()) > 1 and param.grad.size()[1] == pen.size():
                param.grad *= pen.unsqueeze(0)
            elif len(param.grad.size()) == 1 and param.grad.size() == pen.size():
                param.grad *= pen

            # Gradient ratio tracking
            if self.config.grad_ratio_tracking and grad_before is not None:
                grad_eps = 1e-8
                grad_ratio = torch.where(
                    grad_before.abs() > grad_eps,
                    param.grad / (grad_before + grad_eps),
                    torch.ones_like(grad_before),
                )
                if name not in self._grad_ratio:
                    self._grad_ratio[name] = []
                self._grad_ratio[name].append(grad_ratio)

                ratio_mean = grad_ratio.mean().item()
                ratio_max = grad_ratio.max().item()
                ratio_min = grad_ratio.min().item()
                self._grad_ratio_stats["mean"].append(ratio_mean)
                self._grad_ratio_stats["max"].append(ratio_max)
                self._grad_ratio_stats["min"].append(ratio_min)

                logger.info(
                    "Module %s - Param shape: %s | "
                    "Grad ratio: mean=%.4f, max=%.4f, min=%.4f",
                    name, param.grad.shape,
                    ratio_mean, ratio_max, ratio_min,
                )

    # ==================================================================
    # Internal: Fig1(c) Gradient Noise Analysis (IRRA feature)
    # ==================================================================

    def _fig1c_noise_experiment(self, model):
        """Run gradient noise sensitivity experiment (once per epoch)."""
        self._collect_fig1c_gradient(model, is_noise=False, modality="text")
        self._collect_fig1c_gradient(model, is_noise=False, modality="img")

        text_cache = self._add_gaussian_noise(model, "text")
        self._collect_fig1c_gradient(model, is_noise=True, modality="text")
        self._restore_gradient(model, "text", text_cache)

        img_cache = self._add_gaussian_noise(model, "img")
        self._collect_fig1c_gradient(model, is_noise=True, modality="img")
        self._restore_gradient(model, "img", img_cache)

    def _collect_fig1c_gradient(self, model, is_noise, modality):
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        grad_norms = []
        for module in modules.values():
            for param in module.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad, p='fro').item())
        avg = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        cache_key = f"{modality}_{'with_noise' if is_noise else 'no_noise'}"
        self._grad_cache[cache_key].append(avg)
        logger.info("fig1c: %s", self._grad_cache[cache_key])
        return avg

    def _add_gaussian_noise(self, model, modality):
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        cache = {}
        noise_std = self.config.fig1c_noise_std
        for name, module in modules.items():
            cache[name] = {}
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    cache[name][pname] = param.grad.clone()
                    param.grad += torch.randn_like(param.grad) * noise_std
        return cache

    def _restore_gradient(self, model, modality, cache):
        modules = self._img_enc_modules if modality == "img" else self._txt_enc_modules
        for name, module in modules.items():
            if name not in cache:
                continue
            for pname, param in module.named_parameters():
                if pname in cache[name] and param.grad is not None:
                    param.grad = cache[name][pname]

    def _save_fig1c_data(self, epoch):
        if _DIST_AVAILABLE and dist.is_initialized() and dist.get_rank() != 0:
            return
        for key in self._grad_cache:
            vals = self._grad_cache[key]
            self._fig1c_data[key].append(
                sum(vals) / len(vals) if vals else 0.0
            )
        self._fig1c_data["epoch_list"].append(epoch)
        save_dir = os.path.join(self.config.output_dir, "fig1c_data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "fig1c_gradient_data.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self._fig1c_data, f)
        for key in self._grad_cache:
            self._grad_cache[key] = []
        logger.info("[Fig1(c)] Epoch %d gradient data saved to %s", epoch, save_path)


# ======================================================================
# Integration Examples
# ======================================================================

IRRA_EXAMPLE = """
# ==================================================================
# IRRA Integration (zero-difference from original)
# ==================================================================

from unified_modality_grad_modulator import (
    UnifiedModalityGradModulator, UnifiedModulationConfig
)

config = UnifiedModulationConfig.irra_preset(
    tau=int(args.tau),
    gammb=args.gammb,
    gaeta=args.gaeta,
    alpha_sigmoid=args.alpha,
    beta_sigmoid=args.beta,
    enabled=args.modulation,
    output_dir=args.output_dir,
)

def shared_filter(name, mod):
    return (not name.startswith('base_model')
            and not name.endswith('classifier')
            and not name.startswith('ln_pre_'))

def img_enc_filter(name, mod):
    return name.startswith(('base_model.visual', 'ln_pre_i'))

def txt_enc_filter(name, mod):
    return name.startswith(('base_model.transformer', 'ln_pre_t'))

modulator = UnifiedModalityGradModulator(config)
modulator.attach(model, shared_filter, img_enc_filter, txt_enc_filter)

# Training loop:
for epoch in range(num_epochs):
    modulator.on_epoch_start(model)
    for step, batch in enumerate(loader):
        modulator.pre_forward(model)
        ret = model(batch)
        modulator.capture('normal')
        with torch.no_grad():
            e_txt_ret = model(batch, erase='e_txt')
            modulator.capture('e_txt')
            e_img_ret = model(batch, erase='e_img')
            modulator.capture('e_img')
        loss.backward()
        modulator.post_backward(model, step, len(loader), batch_size,
                                ret['mlm_loss'], e_txt_ret['mlm_loss'], e_img_ret['mlm_loss'])
        optimizer.step()
    stats = modulator.on_epoch_end(model, epoch)
"""

RASA_EXAMPLE = """
# ==================================================================
# RaSa Integration (zero-difference from original)
# ==================================================================

from unified_modality_grad_modulator import (
    UnifiedModalityGradModulator, UnifiedModulationConfig
)

config = UnifiedModulationConfig.rasa_preset()

modulator = UnifiedModalityGradModulator(config)
modulator.attach(model,
    model_activation_attr="ACTIVATIONS",
    model_belong_key="CM",
    img_enc_prefixes=("visual", "vision"),
    txt_enc_prefixes=("text",))

# Training loop:
for epoch in range(num_epochs):
    modulator.on_epoch_start(model)
    for step, batch in enumerate(loader):
        modulator.pre_forward(model)  # no-op for model mode
        ret = model(batch)
        modulator.capture('normal', model)
        with torch.no_grad():
            e_img_ret = model(batch, type='E_IMG')
            modulator.capture('e_img', model)
            e_txt_ret = model(batch, type='E_TXT')
            modulator.capture('e_txt', model)
        loss.backward()
        modulator.post_backward(model, step, len(loader), batch_size,
                                all_loss, e_txt_loss, e_img_loss)
        optimizer.step()
    info = modulator.on_epoch_end(model, epoch)
"""
