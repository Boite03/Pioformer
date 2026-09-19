import json
from pathlib import Path
from typing import Dict, Mapping

import torch


def _unwrap_checkpoint(payload) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise TypeError("Checkpoint state_dict must be a string-keyed mapping")
    return state


def read_weights_only(path: str) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("Safe checkpoint loading requires a PyTorch version with weights_only support") from error
    return _unwrap_checkpoint(payload)


def _strip_known_prefix(key: str) -> str:
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_checkpoint_strict(model, path: str) -> Dict[str, object]:
    source = {_strip_known_prefix(key): value for key, value in read_weights_only(path).items()}
    target = model.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    mismatched = sorted(
        key for key in set(source) & set(target)
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    report = {"checkpoint": str(path), "missing": missing, "unexpected": unexpected, "mismatched": mismatched}
    if missing or unexpected or mismatched:
        raise RuntimeError("Strict checkpoint compatibility failed:\n" + json.dumps(report, indent=2))
    model.load_state_dict(source, strict=True)
    return report


def initialize_main_decoder(model) -> Dict[str, object]:
    source = model.aux_decoder.state_dict()
    target = model.decoder.state_dict()
    candidates = {
        "loc": "loc",
        "scale": "scale",
        "pi": "hybrid_pi",
        "aggr_embed": "hybrid_aggr_embed",
    }
    copied, reinitialized = [], []
    with torch.no_grad():
        for source_prefix, target_prefix in candidates.items():
            pairs = {}
            for source_key, value in source.items():
                if not source_key.startswith(source_prefix + "."):
                    continue
                target_key = target_prefix + source_key[len(source_prefix):]
                pairs[target_key] = (source_key, value)
            incompatible_modules = {
                target_key.rsplit(".", 1)[0]
                for target_key, (_, value) in pairs.items()
                if target_key not in target or target[target_key].shape != value.shape
            }
            for target_key, (source_key, value) in pairs.items():
                module_name = target_key.rsplit(".", 1)[0]
                if module_name not in incompatible_modules:
                    target[target_key].copy_(value)
                    copied.append("{} -> {}".format(source_key, target_key))
                else:
                    reinitialized.append(target_key)
    model.decoder.load_state_dict(target, strict=True)
    return {"copied": sorted(copied), "reinitialized": sorted(set(reinitialized))}


def write_report(report: Mapping[str, object], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
