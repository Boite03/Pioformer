from typing import Dict

import torch

from pioformer.model.outputs import TrajectoryPrediction


def focal_metrics(prediction: TrajectoryPrediction, target: torch.Tensor,
                  valid_mask: torch.Tensor, focal_index: torch.Tensor,
                  miss_threshold: float = 2.0) -> Dict[str, torch.Tensor]:
    index = focal_index.long().reshape(-1)
    positions = prediction.positions[:, index]
    truth = target[index]
    mask = valid_mask[index]
    metrics = {"minADE": [], "minFDE": [], "MR": []}
    for scene in range(index.numel()):
        valid = torch.nonzero(mask[scene], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        distance = torch.norm(positions[:, scene, valid] - truth[scene, valid], p=2, dim=-1)
        ade = distance.mean(dim=-1)
        fde = distance[:, -1]
        metrics["minADE"].append(ade.min())
        metrics["minFDE"].append(fde.min())
        metrics["MR"].append((fde.min() > miss_threshold).to(distance.dtype))
    result = {}
    reference = prediction.positions
    for name, values in metrics.items():
        result[name] = torch.stack(values).mean() if values else reference.sum() * 0.0
    return result
