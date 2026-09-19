from pathlib import Path
from typing import Dict, Optional

import torch
from pytorch_lightning import Callback

from pioformer.training.reporting import scalar_validation_metrics


SAFE_CHECKPOINT_SCHEMA = 1


def save_tensor_weights(model, path: str) -> None:
    """Save a model-only checkpoint that PyTorch 1.13 can load with weights_only=True."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(
        {
            "schema_version": torch.tensor(SAFE_CHECKPOINT_SCHEMA),
            "state_dict": state_dict,
        },
        str(temporary),
    )
    temporary.replace(destination)


class SafeWeightsCheckpoint(Callback):
    """Track one metric while writing only tensors to ``best.ckpt``."""

    def __init__(self, dirpath: str, monitor: str, mode: str = "min") -> None:
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_metrics: Dict[str, float] = {}
        self.best_model_path = str(Path(dirpath) / "best.ckpt")

    @property
    def state_key(self) -> str:
        return "{}.{}".format(type(self).__qualname__, self.monitor)

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        score_tensor = torch.as_tensor(value).detach().float().cpu()
        if score_tensor.numel() != 1 or not torch.isfinite(score_tensor).item():
            return
        score = float(score_tensor.item())
        if self._improved(score):
            save_tensor_weights(pl_module.model, self.best_model_path)
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch) + 1
            self.best_metrics = scalar_validation_metrics(trainer.callback_metrics)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> Dict[str, object]:
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
        }

    def on_load_checkpoint(self, trainer, pl_module, callback_state: Dict[str, object]) -> None:
        value = callback_state.get("best_score")
        self.best_score = None if value is None else float(value)
        epoch = callback_state.get("best_epoch")
        self.best_epoch = None if epoch is None else int(epoch)
        metrics = callback_state.get("best_metrics", {})
        self.best_metrics = {str(key): float(metric) for key, metric in metrics.items()}
