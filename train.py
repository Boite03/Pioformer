from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from pioformer.training.runner import make_model_config, run_stages


def main() -> None:
    parser = ArgumentParser(description="Train Pioformer on Argoverse 1")
    parser.add_argument("--root", type=str, required=True, help="Argoverse 1 dataset root")
    parser.add_argument("--embed_dim", type=int, default=64, choices=(64, 128))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--local_radius", type=float, default=50.0)
    parser.add_argument("--gpus", type=int, default=1, help="Use 0 for CPU")
    parser.add_argument("--precision", type=int, default=32, choices=(16, 32))
    parser.add_argument("--stages", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--init_from", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=20)
    args = parser.parse_args()

    seed_everything(args.seed)
    size = "s" if args.embed_dim == 64 else "l"
    config = {
        "root": args.root,
        "run_name": args.run_name or "pioformer-{}".format(size),
        "output_root": args.output_dir,
        "model": make_model_config(args.embed_dim, args.local_radius),
        "training": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "accelerator": "gpu" if args.gpus > 0 else "cpu",
            "devices": args.gpus if args.gpus > 0 else 1,
            "precision": args.precision,
            "deterministic": args.deterministic,
            "log_every_n_steps": args.log_every_n_steps,
        },
    }
    run_stages(config, list(args.stages), args.init_from, args.resume)


if __name__ == "__main__":
    main()
