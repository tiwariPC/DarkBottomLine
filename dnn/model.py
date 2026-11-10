"""PyTorch model definition for the tabular DNN classifier.

Purpose:
- Define a small MLP (multi-layer perceptron) for tabular features.
- Provide save/load helpers that bundle architecture hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    n_inputs: int
    hidden_layers: tuple[int, ...]
    dropout: float
    parametric: bool = False
    mass_grid: Optional[list] = field(default=None)


def parse_hidden_layers(s: str) -> tuple[int, ...]:
    s = str(s).strip()
    if not s:
        return ()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError("Hidden layer sizes must be positive")
        out.append(v)
    return tuple(out)


def build_mlp(spec: ModelSpec):
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    in_f = int(spec.n_inputs)

    for h in spec.hidden_layers:
        layers.append(nn.Linear(in_f, int(h)))
        layers.append(nn.ReLU())
        if float(spec.dropout) > 0:
            layers.append(nn.Dropout(float(spec.dropout)))
        in_f = int(h)

    layers.append(nn.Linear(in_f, 1))  # logits
    return nn.Sequential(*layers)


def save_checkpoint(path: str, *, model, spec: ModelSpec):
    import torch

    ckpt = {
        "state_dict": model.state_dict(),
        "spec": {
            "n_inputs": int(spec.n_inputs),
            "hidden_layers": list(spec.hidden_layers),
            "dropout": float(spec.dropout),
            "parametric": bool(spec.parametric),
            "mass_grid": spec.mass_grid,
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, *, map_location: str = "cpu"):
    import torch

    ckpt = torch.load(path, map_location=map_location)
    spec_d = ckpt.get("spec") or {}
    state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint {path!r} has neither 'state_dict' nor 'model_state_dict'."
        )

    # Older checkpoints stored the training YAML under model_config and used
    # ParametricDNN's "_net." state-dict prefix.
    legacy = ckpt.get("model_config") or {}
    if spec_d:
        spec = ModelSpec(
            n_inputs=int(spec_d["n_inputs"]),
            hidden_layers=tuple(int(x) for x in spec_d.get("hidden_layers", [])),
            dropout=float(spec_d.get("dropout", 0.0)),
            parametric=bool(spec_d.get("parametric", False)),
            mass_grid=spec_d.get("mass_grid"),
        )
    else:
        hidden_raw = legacy.get(
            "hidden_layers",
            ckpt.get("hidden_layers", ()),
        )
        hidden = (
            parse_hidden_layers(hidden_raw)
            if isinstance(hidden_raw, str)
            else tuple(int(x) for x in hidden_raw)
        )
        parametric = bool(
            legacy.get("parametric_input", ckpt.get("parametric_input", False))
        )
        first_weight = next(
            (
                value for key, value in state_dict.items()
                if key.replace("_net.", "", 1).endswith("weight")
                and getattr(value, "ndim", 0) == 2
            ),
            None,
        )
        if first_weight is None:
            raise KeyError(f"Could not infer model input width from checkpoint {path!r}.")
        model_inputs = int(first_weight.shape[1])
        event_inputs = int(
            legacy.get(
                "input_features",
                model_inputs - (2 if parametric else 0),
            )
        )
        spec = ModelSpec(
            n_inputs=event_inputs + (2 if parametric else 0),
            hidden_layers=hidden,
            dropout=float(
                legacy.get("dropout", ckpt.get("dropout", 0.0))
            ),
            parametric=parametric,
            mass_grid=legacy.get("mass_grid", ckpt.get("mass_grid")),
        )

    if any(str(key).startswith("_net.") for key in state_dict):
        state_dict = {
            str(key).replace("_net.", "", 1): value
            for key, value in state_dict.items()
        }
    model = build_mlp(spec)
    model.load_state_dict(state_dict)
    model.eval()
    return model, spec
