"""Stage S arms (step S-L4b): S2 (separate encoders), B3 (shared, geometry detached), B4 (shared, coupled).

CPU, untrained, synthetic-probe scope. The band width is fixed at 8 rows, matching Reader.
"""

from __future__ import annotations

import torch
from torch import nn

from babaorum.experimental.stage_s_model import Encoder, HeadD, HeadG, refine_lines
from babaorum.experimental.stage_s_reader import Reader, expected_ctc_loss

ARMS = ("S2", "B3", "B4")
ROWS = 8


class GradScale(torch.autograd.Function):
    """Identity in the forward pass; the backward gradient is multiplied by the scale s (E2E-5, lambda)."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class StageSModel(nn.Module):
    def __init__(self, arm: str, alphabet_size: int, channels: int = 64, K: int = 8, hidden: int = 128,
                 ctc_det_grad_scale: float = 1.0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        if not (isinstance(ctc_det_grad_scale, (int, float)) and 0.0 < ctc_det_grad_scale <= 1.0 + 1e-12):
            raise ValueError(f"ctc_det_grad_scale must be in (0, 1], got {ctc_det_grad_scale!r}")
        self.arm = arm
        self.K = K
        # Scale of the L_ctc gradient entering enc_det through features_read (B3 and B4 only; S2 has enc_read).
        self.ctc_det_grad_scale = float(ctc_det_grad_scale)
        self.enc_det = Encoder(channels)
        self.enc_read = Encoder(channels) if arm == "S2" else None
        self.head_d = HeadD(channels)
        self.head_g = HeadG(channels, K, hidden=hidden)
        self.reader = Reader(channels * ROWS, alphabet_size, hidden=hidden)

    def forward_lines(self, image, polylines_page_edges, labels_list, *, sigma_px: float) -> dict:
        features_det = self.enc_det(image)
        logit, neg_map, pos_map = self.head_d(features_det)
        det = {"logit": logit, "neg": neg_map, "pos": pos_map}
        if len(polylines_page_edges) == 0:
            return {"refined": None, "ctc_losses": [], "statuses": [], "det": det}
        refined = refine_lines(features_det, polylines_page_edges, self.head_g,
                               head_d=self.head_d, K=self.K)
        if self.arm == "S2":
            features_read = self.enc_read(image)
        else:
            features_read = GradScale.apply(features_det, self.ctc_det_grad_scale)

        if self.arm == "B4":
            geometry = refined
        else:
            geometry = {key: refined[key].detach() for key in ("baseline", "neg", "pos")}

        losses, statuses = [], []
        for i, labels in enumerate(labels_list):
            line = {"baseline": geometry["baseline"][i], "neg": geometry["neg"][i],
                    "pos": geometry["pos"][i]}
            loss, status = expected_ctc_loss(features_read, line, self.reader, labels,
                                             sigma_px=sigma_px)
            losses.append(loss)
            statuses.append(status)
        return {"refined": refined, "ctc_losses": losses, "statuses": statuses, "det": det}
