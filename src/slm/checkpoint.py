from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from .model import ModelArgs, TransformerLM


def strip_wrapper_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("_orig_mod.", "module.", "model.")
    clean = {}
    for name, tensor in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        clean[name] = tensor
    return clean


def load_training_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> TransformerLM:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    raw_args = dict(checkpoint["model_args"])
    if "max_block_size" in raw_args:
        raw_args["block_size"] = raw_args.pop("max_block_size")
    allowed = {field.name for field in fields(ModelArgs)}
    model = TransformerLM(ModelArgs(**{key: value for key, value in raw_args.items() if key in allowed}))
    state = strip_wrapper_prefixes(checkpoint["model"])
    model.load_state_dict(state)
    return model.to(device).eval()


def convert_checkpoint(source: str | Path, output_dir: str | Path) -> None:
    model = load_training_checkpoint(source)
    model.save_pretrained(output_dir)
