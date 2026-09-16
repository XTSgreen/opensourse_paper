"""Run pinned TIGON on common state-composition coordinates."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint


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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--initial-input", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    from utility import UOT, initialize_weights, train_model  # noqa: PLC0415

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = torch.tensor(panel["source_composition"], dtype=torch.float32)
    target = torch.tensor(panel["target_composition"], dtype=torch.float32)
    data = [source, target]
    device = torch.device("cpu")
    model = UOT(source.shape[1], hidden_dim=16, n_hiddens=2, activation="Tanh").to(device)
    model.apply(initialize_weights)
    settings = SimpleNamespace(niters=args.iterations, num_samples=min(args.samples, len(source), len(target)))
    options = {
        "method": "Dopri5", "h": None, "rtol": 1e-3, "atol": 1e-5,
        "print_neval": False, "neval_max": 1000000, "safety": None,
    }
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.01)
    mse = nn.MSELoss()
    sigma = 1.0
    losses = []
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad()
        loss, _, sigma, _, _ = train_model(
            mse, model, settings, data, [0, 1], [0.0, 1.0], sigma, options, device, iteration
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    if args.initial_input is None:
        initial = source
        initial_counts = np.asarray(panel["source_counts"], dtype=float)
    else:
        initial_panel = np.load(args.initial_input, allow_pickle=False)
        initial = torch.tensor(initial_panel["source_composition"], dtype=torch.float32)
        initial_counts = np.asarray(initial_panel["source_counts"], dtype=float)
    zeros = torch.zeros((len(initial), 1), dtype=torch.float32)
    generated, _, _ = odeint(
        model, (initial, zeros, zeros), torch.tensor([0.0, 1.0]),
        atol=1e-5, rtol=1e-5, method="midpoint", options={"step_size": 0.1},
    )
    generated = _simplex(generated[-1].detach().numpy())
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
        "method": "TIGON",
        "method_version": "1ed92cfcc250415fc01b4d344a308b0680cc9635",
        "official_components": ["utility.UOT", "utility.train_model"],
        "input": str(args.input.resolve()),
        "initial_input": str(args.initial_input.resolve()) if args.initial_input else None,
        "output": str(args.output.resolve()),
        "iterations": args.iterations,
        "samples": settings.num_samples,
        "seed": args.seed,
        "final_loss": losses[-1],
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
