from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TrajectoryPrediction:
    positions: torch.Tensor
    logits: torch.Tensor
    scale: Optional[torch.Tensor] = None

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.softmax(dim=-1)


@dataclass(frozen=True)
class PioformerOutput:
    auxiliary: TrajectoryPrediction
    proposal: Optional[TrajectoryPrediction] = None
    refined: Optional[TrajectoryPrediction] = None
