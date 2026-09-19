import gc
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from pioformer.data.datamodule import Argoverse1DataModule
from pioformer.training.checkpoints import (
    initialize_main_decoder, load_checkpoint_strict, write_report,
)
from pioformer.training.lightning_module import PioformerTask
from pioformer.training.progress import PioformerProgressBar
from pioformer.training.reporting import (
    make_stage_summary, print_run_summary, print_stage_result, print_stage_start, write_summary,
)
from pioformer.training.safe_checkpoint import SafeWeightsCheckpoint


def make_model_config(embed_dim: int = 64, local_radius: float = 50.0) -> Dict:
    if embed_dim not in (64, 128):
        raise ValueError("embed_dim must be 64 (Pioformer-S) or 128 (Pioformer-L)")
    return {
        "historical_steps": 20,
        "future_steps": 30,
        "num_modes": 6,
        "rotate": True,
        "node_dim": 2,
        "edge_dim": 2,
        "embed_dim": embed_dim,
        "num_heads": 8,
        "dropout": 0.1,
        "num_temporal_layers": 4,
        "num_global_layers": 3,
        "local_radius": local_radius,
        "parallel": False,
        "hyperedge_size": 4,
    }


def _task(config: Dict, stage: int) -> PioformerTask:
    training = config["training"]
    return PioformerTask(
        config["model"], stage,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        max_epochs=training["max_epochs"],
        lambda1=1.0,
        lambda2=5.0,
    )


def _trainer(config: Dict, stage_dir: Path) -> Trainer:
    training = config["training"]
    checkpoint = ModelCheckpoint(
        dirpath=str(stage_dir), filename="best-resume", monitor="val_minFDE",
        mode="min", save_last=True, save_top_k=1,
    )
    safe_checkpoint = SafeWeightsCheckpoint(
        dirpath=str(stage_dir), monitor="val_minFDE", mode="min",
    )
    logger = CSVLogger(save_dir=str(stage_dir), name="", version="")
    return Trainer(
        accelerator=training["accelerator"],
        devices=training["devices"],
        precision=training["precision"],
        max_epochs=training["max_epochs"],
        callbacks=[
            safe_checkpoint,
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            PioformerProgressBar(),
        ],
        logger=logger,
        deterministic=training["deterministic"],
        log_every_n_steps=training["log_every_n_steps"],
        enable_model_summary=False,
    )


def _validate_stages(stages: List[int], init_from: Optional[str], resume: Optional[str]) -> None:
    if not stages or any(stage not in (1, 2, 3) for stage in stages):
        raise ValueError("stages must contain only 1, 2, and 3")
    if stages != list(range(stages[0], stages[-1] + 1)):
        raise ValueError("stages must be contiguous and ordered")
    if stages[0] > 1 and not (init_from or resume):
        raise ValueError("Stage 2/3 requires --init_from or --resume")
    if init_from and resume:
        raise ValueError("--init_from and --resume are mutually exclusive")


def run_stages(config: Dict, stages: List[int], init_from: Optional[str] = None,
               resume: Optional[str] = None) -> None:
    _validate_stages(stages, init_from, resume)
    run_dir = Path(config["output_root"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_summary(config, run_dir / "config.json")

    previous_best = init_from
    stage_summaries = []
    for index, stage in enumerate(stages):
        stage_dir = run_dir / "stage-{}".format(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        task = _task(config, stage)
        report = {"stage": stage, "source": previous_best}
        resume_path = resume if index == 0 else None
        if resume_path is None and previous_best:
            report["load"] = load_checkpoint_strict(task.model, previous_best)
        if stage == 2 and resume_path is None:
            report["decoder_initialization"] = initialize_main_decoder(task.model)
        write_report(report, str(stage_dir / "weight_migration.json"))

        training = config["training"]
        print_stage_start(stage, training["max_epochs"], training["batch_size"])
        datamodule = Argoverse1DataModule(
            root=config["root"],
            batch_size=training["batch_size"],
            num_workers=training["num_workers"],
            local_radius=config["model"]["local_radius"],
        )
        trainer = _trainer(config, stage_dir)
        trainer.fit(task, datamodule=datamodule, ckpt_path=resume_path)

        safe_checkpoint = next(
            callback for callback in trainer.callbacks
            if isinstance(callback, SafeWeightsCheckpoint)
        )
        previous_best = safe_checkpoint.best_model_path
        if not Path(previous_best).is_file():
            raise RuntimeError("Stage {} finished without a safe best checkpoint".format(stage))
        summary = make_stage_summary(
            stage, safe_checkpoint.best_epoch, safe_checkpoint.best_metrics, previous_best,
        )
        write_summary(summary, stage_dir / "summary.json")
        print_stage_result(summary)
        stage_summaries.append(summary)
        resume = None

        del trainer, task, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary({"stages": stage_summaries}, run_dir / "training_summary.json")
    print_run_summary(stage_summaries)
