from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F

from pioformer.model.outputs import PioformerOutput, TrajectoryPrediction
from pioformer.model.stages import TrainingStage


@dataclass
class PredictionLoss:
    regression: torch.Tensor
    classification: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.regression + self.classification


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def prediction_loss(prediction: TrajectoryPrediction, target: torch.Tensor, valid_mask: torch.Tensor,
                    regression: str = "laplace") -> PredictionLoss:
    valid_steps = valid_mask.sum(dim=-1)
    actor_mask = valid_steps > 0
    if not torch.any(actor_mask):
        zero = _zero(prediction.positions)
        return PredictionLoss(zero, zero)

    distance = torch.norm(prediction.positions - target.unsqueeze(0), p=2, dim=-1)
    mode_cost = (distance * valid_mask.unsqueeze(0)).sum(dim=-1)
    winner = mode_cost.argmin(dim=0)
    actor_index = torch.arange(target.size(0), device=target.device)
    best_positions = prediction.positions[winner, actor_index]

    selected = valid_mask.unsqueeze(-1).expand_as(target)
    if regression == "laplace":
        if prediction.scale is None:
            raise ValueError("Laplace regression requires positive scale predictions")
        best_scale = prediction.scale[winner, actor_index]
        nll = torch.log(2.0 * best_scale) + torch.abs(best_positions - target) / best_scale
        regression_loss = nll[selected].mean()
    elif regression == "smooth_l1":
        regression_loss = F.smooth_l1_loss(best_positions[selected], target[selected])
    else:
        raise ValueError("Unknown regression loss: {}".format(regression))

    soft_target = torch.softmax(
        -mode_cost[:, actor_mask] / valid_steps[actor_mask].to(mode_cost.dtype).unsqueeze(0), dim=0,
    ).transpose(0, 1).detach()
    log_probability = F.log_softmax(prediction.logits[actor_mask], dim=-1)
    classification_loss = -(soft_target * log_probability).sum(dim=-1).mean()
    return PredictionLoss(regression_loss, classification_loss)


def compute_stage_loss(output: PioformerOutput, target: torch.Tensor, valid_mask: torch.Tensor,
                       stage: int, lambda1: float = 1.0, lambda2: float = 5.0) -> Dict[str, torch.Tensor]:
    stage = TrainingStage.parse(stage)
    aux = prediction_loss(output.auxiliary, target, valid_mask, "laplace")
    values = {
        "reg_aux": aux.regression,
        "cls_aux": aux.classification,
        "stage1": aux.total,
    }
    total = aux.total
    if stage >= TrainingStage.TPN:
        if output.proposal is None:
            raise ValueError("Stage 2/3 output must include proposal")
        proposal = prediction_loss(output.proposal, target, valid_mask, "laplace")
        values.update(reg_proposal=proposal.regression, cls_proposal=proposal.classification)
        total = total + lambda1 * proposal.total
        values["stage2"] = total
    if stage >= TrainingStage.PRN:
        if output.refined is None:
            raise ValueError("Stage 3 output must include refined prediction")
        refined = prediction_loss(output.refined, target, valid_mask, "smooth_l1")
        values.update(reg_refined=refined.regression, cls_refined=refined.classification)
        total = total + lambda2 * refined.total
        values["stage3"] = total
    values["total"] = total
    return values
