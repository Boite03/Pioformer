import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch


VALIDATION_METRICS = ("val_loss", "val_minADE", "val_minFDE", "val_MR")
STAGE_NAMES = {
    1: "CTN",
    2: "TPN",
    3: "PRN",
}


def scalar_validation_metrics(metrics: Mapping[str, object]) -> Dict[str, float]:
    """Extract finite scalar validation metrics from Lightning's metric mapping."""
    result: Dict[str, float] = {}
    for name in VALIDATION_METRICS:
        if name not in metrics:
            continue
        value = torch.as_tensor(metrics[name]).detach().float().cpu()
        if value.numel() == 1 and torch.isfinite(value).item():
            result[name] = float(value.item())
    return result


def make_stage_summary(stage: int, best_epoch: Optional[int], metrics: Dict[str, float],
                       checkpoint: str) -> Dict[str, object]:
    return {
        "stage": int(stage),
        "name": STAGE_NAMES[int(stage)],
        "best_epoch": best_epoch,
        "selection_metric": "val_minFDE",
        "metrics": dict(metrics),
        "checkpoint": str(Path(checkpoint).resolve()),
    }


def write_summary(summary: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _format_metric(metrics: Mapping[str, float], name: str) -> str:
    value = metrics.get(name)
    return "n/a" if value is None else "{:.4f}".format(value)


def print_stage_start(stage: int, max_epochs: int, batch_size: int) -> None:
    print("\n=== Stage {}: {} (epochs={}, batch_size={}) ===".format(
        stage, STAGE_NAMES[stage], max_epochs, batch_size,
    ), flush=True)


def print_stage_result(summary: Mapping[str, object]) -> None:
    metrics = summary["metrics"]
    print("\nStage {} ({}) best validation result".format(summary["stage"], summary["name"]))
    print("  epoch   : {}".format(summary["best_epoch"] or "n/a"))
    print("  minADE  : {}".format(_format_metric(metrics, "val_minADE")))
    print("  minFDE  : {}".format(_format_metric(metrics, "val_minFDE")))
    print("  MR      : {}".format(_format_metric(metrics, "val_MR")))
    print("  val_loss: {}".format(_format_metric(metrics, "val_loss")))
    print("  weights : {}".format(summary["checkpoint"]), flush=True)


def print_run_summary(summaries: Iterable[Mapping[str, object]]) -> None:
    rows = list(summaries)
    if not rows:
        return
    print("\n=== Pioformer training summary ===")
    print("Stage  Model  Best epoch  minADE    minFDE    MR")
    for summary in rows:
        metrics = summary["metrics"]
        epoch = summary["best_epoch"] or "n/a"
        print("{:<6} {:<6} {:<11} {:<9} {:<9} {}".format(
            summary["stage"], summary["name"], epoch,
            _format_metric(metrics, "val_minADE"),
            _format_metric(metrics, "val_minFDE"),
            _format_metric(metrics, "val_MR"),
        ))
    final = rows[-1]
    print("\nFinal result: Stage {} ({})".format(final["stage"], final["name"]))
    print("Best checkpoint: {}".format(final["checkpoint"]), flush=True)
