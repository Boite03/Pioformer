from pytorch_lightning.callbacks.progress import TQDMProgressBar


class PioformerProgressBar(TQDMProgressBar):
    """Keep Lightning's familiar HiVT-style bar while removing noisy fields."""

    def get_metrics(self, trainer, pl_module):
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        if "loss" in metrics:
            loss = metrics.pop("loss")
            metrics = {"train_loss": loss, **metrics}
        return metrics
