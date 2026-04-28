"""PyTorch model definition for the tabular DNN classifier.

Purpose:
- Define a small MLP (multi-layer perceptron) for tabular features.
- Provide save/load helpers that bundle architecture hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    n_inputs: int
    hidden_layers: tuple[int, ...]
    dropout: float


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
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, *, map_location: str = "cpu"):
    import torch

    ckpt = torch.load(path, map_location=map_location)
    spec_d = ckpt.get("spec") or {}
    spec = ModelSpec(
        n_inputs=int(spec_d["n_inputs"]),
        hidden_layers=tuple(int(x) for x in spec_d.get("hidden_layers", [])),
        dropout=float(spec_d.get("dropout", 0.0)),
    )
    model = build_mlp(spec)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, spec
