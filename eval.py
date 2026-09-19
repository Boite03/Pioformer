from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from pioformer.data.datamodule import Argoverse1DataModule
from pioformer.training.checkpoints import load_checkpoint_strict
from pioformer.training.lightning_module import PioformerTask
from pioformer.training.progress import PioformerProgressBar
from pioformer.training.reporting import scalar_validation_metrics
from pioformer.training.runner import make_model_config


def main() -> None:
    parser = ArgumentParser(description="Evaluate Pioformer on Argoverse 1 validation set")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=64, choices=(64, 128))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--local_radius", type=float, default=50.0)
    parser.add_argument("--gpus", type=int, default=1, help="Use 0 for CPU")
    parser.add_argument("--precision", type=int, default=32, choices=(16, 32))
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    seed_everything(args.seed)
    task = PioformerTask(make_model_config(args.embed_dim, args.local_radius), stage=3)
    load_checkpoint_strict(task.model, args.checkpoint)
    datamodule = Argoverse1DataModule(
        args.root, args.batch_size, args.num_workers, args.local_radius,
    )
    trainer = Trainer(
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        logger=False,
        callbacks=[PioformerProgressBar()],
        enable_model_summary=False,
    )
    results = trainer.validate(task, datamodule=datamodule)
    metrics = scalar_validation_metrics(results[0] if results else {})
    print("\n=== Pioformer evaluation result ===")
    print("minADE  : {:.4f}".format(metrics["val_minADE"]))
    print("minFDE  : {:.4f}".format(metrics["val_minFDE"]))
    print("MR      : {:.4f}".format(metrics["val_MR"]))
    print("val_loss: {:.4f}".format(metrics["val_loss"]))


if __name__ == "__main__":
    main()
