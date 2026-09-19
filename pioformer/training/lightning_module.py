from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule

from pioformer.model import PioformerModel, TrainingStage
from pioformer.training.losses import compute_stage_loss
from pioformer.training.metrics import focal_metrics


class PioformerTask(LightningModule):
    def __init__(self, model_config: Dict[str, Any], stage: int,
                 learning_rate: float = 5e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 64, lambda1: float = 1.0, lambda2: float = 5.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.stage_id = TrainingStage.parse(stage)
        self.model = PioformerModel(stage=stage, **model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, batch):
        return self.model(batch, self.stage_id)

    def _losses(self, batch):
        output = self(batch)
        target = self.model.targets(batch)
        valid = ~batch.padding_mask[:, self.model.historical_steps:]
        losses = compute_stage_loss(
            output, target, valid, self.stage_id, lambda1=self.lambda1, lambda2=self.lambda2,
        )
        return output, target, valid, losses

    def training_step(self, batch, batch_idx):
        _, _, _, losses = self._losses(batch)
        for name, value in losses.items():
            metric_name = "train_loss" if name == "total" else "train_{}".format(name)
            self.log(metric_name, value, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch.num_graphs)
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        output, target, valid, losses = self._losses(batch)
        selected = output.refined or output.proposal or output.auxiliary
        metrics = focal_metrics(selected, target, valid, batch.agent_index)
        self.log("val_loss", losses["total"], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)
        for name, value in metrics.items():
            self.log("val_{}".format(name), value, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=batch.num_graphs, sync_dist=True)

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
