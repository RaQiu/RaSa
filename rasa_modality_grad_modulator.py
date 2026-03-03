"""
RaSa-specific Cross-Modal Adaptive Gradient Modulation Plugin.

This module extracts the AGM (Adaptive Gradient Modulation) logic from the
inline implementation in RaSa's Retrieval.py into a reusable plugin with
a lifecycle API. It is designed for ZERO-DIFFERENCE equivalence with the
original inline implementation.

Lifecycle:
    modulator = RaSaModalityGradModulator(config)
    modulator.attach(model)

    for epoch:
        modulator.on_epoch_start(model)
        for step, batch in loader:
            ret = model(batch)                         # Normal forward
            modulator.capture('normal', model)         # Snapshot activations
            e_img_ret = model(batch, type='E_IMG')     # Image-erased forward
            modulator.capture('e_img', model)          # Snapshot activations
            e_txt_ret = model(batch, type='E_TXT')     # Text-erased forward
            modulator.capture('e_txt', model)          # Snapshot activations
            loss.backward()                            # Compute gradients
            modulator.post_backward(model, step, data_len, batch_size,
                                    all_loss, e_txt_loss, e_img_loss)
            optimizer.step()
        stats = modulator.on_epoch_end(model, epoch)
"""

import random
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _unwrap(model):
    """Unwrap DDP model to get the underlying module."""
    return model.module if hasattr(model, 'module') else model


@dataclass
class RaSaModulationConfig:
    """Configuration for RaSa gradient modulation."""
    gammb: float = 2.75
    softmax_temperature: float = 0.1
    enabled: bool = True


class RaSaModalityGradModulator:
    """
    Cross-modal adaptive gradient modulation plugin for RaSa.

    Handles:
    - Modal specificity detection via activation deltas
    - Loss-based scoring with softmax temperature
    - Per-neuron pen vector construction
    - Encoder-level gradient suppression
    - DDP synchronization (all-reduce + rank-0 compute + broadcast)
    """

    def __init__(self, config: RaSaModulationConfig = None):
        if config is None:
            config = RaSaModulationConfig()
        self.config = config

        # Modal neuron counts: {module_iid: tensor}
        self.MODAL_IMG_COUNTS = {}
        self.MODAL_TXT_COUNTS = {}

        # Loss accumulators (will be moved to device during attach)
        self.DELTA_IMG_LOSS = None
        self.DELTA_TXT_LOSS = None
        self.TOT = None

        # LAYER_REC: module_iid -> name mapping
        self.LAYER_REC = {}

        # Activation snapshots: {key: {module: tensor}}
        self._act_snapshots = {}

        # Logger info accumulated during modulation
        self._logger_info = ""

    def attach(self, model):
        """
        Initialize the modulator with the model. Call once after model creation.
        Creates loss accumulator buffers on the correct device.
        """
        device = next(model.parameters()).device
        self.DELTA_IMG_LOSS = torch.zeros(1, device=device)
        self.DELTA_TXT_LOSS = torch.zeros(1, device=device)
        self.TOT = torch.zeros(1, device=device)

    def on_epoch_start(self, model):
        """
        Called at the start of each epoch. Builds LAYER_REC mapping.
        """
        base = _unwrap(model)
        self.LAYER_REC = {}
        for name, module in base.named_modules():
            if (not list(module.children()) and
                    list(module.parameters())):
                module_iid = f"{str(repr(module.__class__))}_{id(module)}"
                self.LAYER_REC[module_iid] = name

    def capture(self, key, model):
        """
        Capture activation snapshot from the model after a forward pass.

        Args:
            key: 'normal', 'e_img', or 'e_txt'
            model: the DDP-wrapped or plain model
        """
        base = _unwrap(model)
        self._act_snapshots[key] = base.ACTIVATIONS.copy()
        base.ACTIVATIONS = {}

    def post_backward(self, model, step, data_len, batch_size,
                      all_loss, e_txt_loss, e_img_loss):
        """
        Core modulation logic. Called after loss.backward(), before optimizer.step().

        Args:
            model: DDP-wrapped model
            step: current step index within epoch
            data_len: total number of steps in the epoch
            batch_size: training batch size
            all_loss: weighted loss vector from normal forward (tensor)
            e_txt_loss: weighted loss vector from text-erased forward (tensor)
            e_img_loss: weighted loss vector from image-erased forward (tensor)
        """
        if not self.config.enabled:
            self._act_snapshots = {}
            return

        model.eval()

        all_act = self._act_snapshots.get('normal', {})
        e_txt_act = self._act_snapshots.get('e_txt', {})
        e_img_act = self._act_snapshots.get('e_img', {})

        base = _unwrap(model)
        device = next(model.parameters()).device

        # --- Loss accumulation (every step) ---
        self.DELTA_IMG_LOSS = self.DELTA_IMG_LOSS + torch.sum(e_img_loss) - torch.sum(all_loss)
        self.DELTA_TXT_LOSS = self.DELTA_TXT_LOSS + torch.sum(e_txt_loss) - torch.sum(all_loss)
        self.TOT = self.TOT + 1

        # --- Modal specificity detection (every step, no tau) ---
        with torch.no_grad():
            for module in all_act.keys():
                module_iid = f"{str(repr(module.__class__))}_{id(module)}"
                # Compute activation deltas
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

                if module_iid not in self.MODAL_IMG_COUNTS and module_iid not in self.MODAL_TXT_COUNTS:
                    self.MODAL_IMG_COUNTS[module_iid] = torch.ones(num_neurons, device=device) * -1
                    self.MODAL_TXT_COUNTS[module_iid] = torch.ones(num_neurons, device=device) * -1

                self.MODAL_IMG_COUNTS[module_iid] = self.MODAL_IMG_COUNTS[module_iid] + (indicate == 1).float()
                self.MODAL_TXT_COUNTS[module_iid] = self.MODAL_TXT_COUNTS[module_iid] + (indicate == 0).float()

        # --- Half-epoch gradient modification (only when DDP is initialized) ---
        if (step + 1) % (data_len // 2) == 0 and dist.is_initialized():
            # DDP sync: all-reduce counts and loss accumulators
            dist.all_reduce(self.DELTA_IMG_LOSS)
            dist.all_reduce(self.DELTA_TXT_LOSS)
            dist.all_reduce(self.TOT)
            for module_iid in self.MODAL_TXT_COUNTS:
                dist.all_reduce(self.MODAL_TXT_COUNTS[module_iid])
                dist.all_reduce(self.MODAL_IMG_COUNTS[module_iid])

            # Rank-0 computes score, then broadcast
            if dist.get_rank() == 0:
                score_cm = torch.tensor([
                    self.DELTA_TXT_LOSS / self.TOT,
                    self.DELTA_IMG_LOSS / self.TOT
                ], device=device)
                ratio_cm = F.softmax(self.config.softmax_temperature * score_cm, dim=0)
            else:
                score_cm = torch.zeros(2, device=device)
                ratio_cm = torch.zeros(2, device=device)
            dist.broadcast(score_cm, src=0)
            dist.broadcast(ratio_cm, src=0)

            r_min, _ = torch.min(ratio_cm, dim=0)
            iscore = (ratio_cm - r_min) ** self.config.gammb
            iscore_txt, iscore_img = iscore[0].item(), iscore[1].item()

            print("Step {}, delta_txt_loss_cm = {} , delta_img_loss_cm = {}".format(
                step, self.DELTA_TXT_LOSS, self.DELTA_IMG_LOSS))
            print("iscore_txt_cm = {}, iscore_img_cm = {}".format(iscore_txt, iscore_img))

            # --- Pen vector construction & gradient modification ---
            already = set()
            for module in all_act.keys():
                if module in base.BELONG['CM']:
                    already.add(id(module))
                    pen = torch.zeros(all_act[module].size(-1)).cuda()
                    module_iid = f"{str(repr(module.__class__))}_{id(module)}"
                    pen_img = (self.MODAL_IMG_COUNTS[module_iid] > self.MODAL_TXT_COUNTS[module_iid])
                    pen += pen_img * iscore_img
                    pen_txt = (self.MODAL_IMG_COUNTS[module_iid] < self.MODAL_TXT_COUNTS[module_iid])
                    pen += pen_txt * iscore_txt
                    pen = 1 - pen

                    for param in module.parameters():
                        if param.grad is not None:
                            if len(param.grad.size()) > 1 and param.grad.size()[1] == pen.size():
                                param.grad *= pen.unsqueeze(0)
                            elif len(param.grad.size()) == 1 and param.grad.size() == pen.size():
                                param.grad *= pen

            # --- Encoder-level suppression ---
            for name, module in base.named_modules():
                if ((not list(module.children())) and
                        name.startswith("text") and
                        (id(module) not in already) and
                        list(module.parameters())):
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_txt)

            for name, module in base.named_modules():
                if ((not list(module.children())) and
                        name.startswith(("visual", "vision")) and
                        (id(module) not in already) and
                        list(module.parameters())):
                    for param in module.parameters():
                        if param.grad is not None:
                            param.grad *= (1 - iscore_img)

            del pen
            torch.cuda.empty_cache()

            # --- Reset BELONG set ---
            base.BELONG['CM'] = set()

            # --- Reset loss accumulators ---
            self.TOT = torch.zeros(1, device=device)
            self.DELTA_TXT_LOSS = torch.zeros(1, device=device)
            self.DELTA_IMG_LOSS = torch.zeros(1, device=device)

            # --- Log and reset modal counts ---
            cnt_txt = 0
            cnt_img = 0
            equal = 0
            for module_iid in self.MODAL_TXT_COUNTS.keys():
                module_txt = (self.MODAL_TXT_COUNTS[module_iid] > self.MODAL_IMG_COUNTS[module_iid]).sum().item()
                module_img = (self.MODAL_IMG_COUNTS[module_iid] > self.MODAL_TXT_COUNTS[module_iid]).sum().item()
                module_equal = (self.MODAL_IMG_COUNTS[module_iid] == self.MODAL_TXT_COUNTS[module_iid]).sum().item()
                self._logger_info += "{} : cnt_txt: {}, cnt_img: {}, equal: {}\n".format(
                    self.LAYER_REC[module_iid], module_txt, module_img, module_equal)
                cnt_txt += module_txt
                cnt_img += module_img
                equal += module_equal
            self._logger_info += "[Step{} ALL] : cnt_txt: {}, cnt_img: {},  equal: {}\n".format(
                step, cnt_txt, cnt_img, equal)
            self.MODAL_IMG_COUNTS = {}
            self.MODAL_TXT_COUNTS = {}

        # Cleanup activation snapshots
        del delta_img, delta_txt, indicate
        torch.cuda.empty_cache()
        self._act_snapshots = {}

        model.train()

    def on_epoch_end(self, model, epoch):
        """
        Called at the end of each epoch. Returns accumulated logger info and resets state.

        Returns:
            str: accumulated log info for this epoch
        """
        device = next(model.parameters()).device
        info = self._logger_info
        self._logger_info = ""

        # Reset all state for next epoch
        self.MODAL_TXT_COUNTS = {}
        self.MODAL_IMG_COUNTS = {}
        self.TOT = torch.zeros(1, device=device)
        self.DELTA_TXT_LOSS = torch.zeros(1, device=device)
        self.DELTA_IMG_LOSS = torch.zeros(1, device=device)

        return info
