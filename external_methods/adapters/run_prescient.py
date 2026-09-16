"""Run pinned PRESCIENT on common state-composition coordinates.

The official model and Sinkhorn loss are retained.  The benchmark supplies its
own already-processed arrays, so the optional R-data reader from the upstream
CLI is intentionally outside this adapter's environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from prescient.train.model import AutoGenerator, OTLoss


def _simplex(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    zero = values.sum(axis=1) <= 0
    values[zero] = 1.0
    return values / values.sum(axis=1, keepdims=True)


def _endpoint_coupling(generated: np.ndarray, target: np.ndarray, source_mass: np.ndarray) -> np.ndarray:
    distance = np.sum((generated[:, None, :] - target[None, :, :]) ** 2, axis=2)
    positive = distance[distance > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    conditional = np.exp(-distance / max(scale, 1e-8))
    conditional /= conditional.sum(axis=1, keepdims=True)
    coupling = source_mass[:, None] * conditional
    return coupling / coupling.sum()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-input", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = torch.tensor(panel["source_composition"], dtype=torch.float32)
    target = torch.tensor(panel["target_composition"], dtype=torch.float32)
    config = SimpleNamespace(
        x_dim=source.shape[1],
        k_dim=args.hidden_dim,
        layers=2,
        activation="softplus",
        sinkhorn_blur=0.05,
        sinkhorn_scaling=0.7,
        train_sd=0.05,
        train_dt=1.0 / args.steps,
    )
    device = torch.device("cpu")
    model = AutoGenerator(config).to(device)
    loss_function = OTLoss(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    uniform_source = torch.full((len(source),), 1.0 / len(source), dtype=torch.float32)
    uniform_target = torch.full((len(target),), 1.0 / len(target), dtype=torch.float32)
    losses = []
    for _ in range(args.epochs):
        prediction = source
        for _ in range(args.steps):
            noise = torch.randn_like(prediction) * config.train_sd
            prediction = model._step(prediction, dt=config.train_dt, z=noise)
        loss = loss_function(uniform_source, prediction, uniform_target, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    torch.manual_seed(args.seed + 1)
    if args.initial_input is None:
        initial = source
        initial_counts = np.asarray(panel["source_counts"], dtype=float)
    else:
        initial_panel = np.load(args.initial_input, allow_pickle=False)
        initial = torch.tensor(initial_panel["source_composition"], dtype=torch.float32)
        initial_counts = np.asarray(initial_panel["source_counts"], dtype=float)
    generated = initial
    for _ in range(args.steps):
        generated = model._step(generated, dt=config.train_dt, z=torch.randn_like(generated) * config.train_sd)
    generated = _simplex(generated.detach().numpy())
    predicted_population = generated.mean(axis=0)
    predicted_population /= predicted_population.sum()
    source_mass = np.sqrt(initial_counts.sum(axis=1))
    source_mass /= source_mass.sum()
    coupling = _endpoint_coupling(generated, np.asarray(panel["target_composition"]), source_mass)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        generated_target=generated,
        coupling=coupling,
        predicted_population=predicted_population,
        state_names=np.asarray(panel["state_names"]).astype(str),
        training_loss=np.asarray(losses),
    )
    metadata = {
        "method": "PRESCIENT",
        "method_version": "0.1.0@50971c7d495e8763eaa60f83af91f51555ed7ece",
        "official_components": ["prescient.train.model.AutoGenerator", "prescient.train.model.OTLoss"],
        "input": str(args.input.resolve()),
        "initial_input": str(args.initial_input.resolve()) if args.initial_input else None,
        "output": str(args.output.resolve()),
        "epochs": args.epochs,
        "steps": args.steps,
        "seed": args.seed,
        "final_loss": losses[-1],
        "runtime_seconds": time.perf_counter() - started,
        "environment_deviation": "pyreadr omitted because benchmark input bypasses upstream R preprocessing CLI",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
