"""Run pinned MIOFlow on state-composition coordinates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
from mioflow.mioflow import MIOFlow


def _simplex(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    zero = totals[:, 0] <= 0
    if np.any(zero):
        values[zero] = 1.0
        totals = values.sum(axis=1, keepdims=True)
    return values / totals


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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = np.asarray(panel["source_composition"], dtype=np.float32)
    target = np.asarray(panel["target_composition"], dtype=np.float32)
    features = np.vstack([source, target])
    obs = pd.DataFrame({"time": np.concatenate([np.zeros(len(source)), np.ones(len(target))])})
    adata = ad.AnnData(X=features, obs=obs)
    adata.obsm["X_pca"] = features.copy()
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    model = MIOFlow(
        adata,
        gaga_model=None,
        gaga_input_key="X_pca",
        obs_time_key="time",
        hidden_dim=args.hidden_dim,
        use_cuda=use_cuda,
        n_epochs=args.epochs,
        lambda_ot=1.0,
        lambda_density=0.0,
        lambda_energy=0.01,
        sample_size=min(256, len(source), len(target)),
        n_trajectories=len(source),
        n_bins=2,
        exp_dir=str(args.output.parent / "runtime"),
    ).fit()
    if args.initial_input is None:
        generated = _simplex(np.asarray(model.trajectories[-1], dtype=float))
        initial_counts = np.asarray(panel["source_counts"], dtype=float)
    else:
        initial_panel = np.load(args.initial_input, allow_pickle=False)
        initial = np.asarray(initial_panel["source_composition"], dtype=np.float32)
        initial_counts = np.asarray(initial_panel["source_counts"], dtype=float)
        normalized = (initial - model.mean_vals) / model.std_vals
        initial_tensor = torch.tensor(normalized, dtype=torch.float32, device=model.device)
        model.ode_model.eval()
        with torch.no_grad():
            trajectory = odeint(model.ode_model, initial_tensor, torch.tensor([0.0, 1.0], device=model.device))
        generated = _simplex(trajectory[-1].cpu().numpy() * model.std_vals + model.mean_vals)
    predicted_population = generated.mean(axis=0)
    target_population = target.mean(axis=0)
    predicted_population /= predicted_population.sum()
    target_population /= target_population.sum()
    source_mass = np.sqrt(initial_counts.sum(axis=1))
    source_mass /= source_mass.sum()
    coupling = _endpoint_coupling(generated, target, source_mass)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        generated_target=generated,
        coupling=coupling,
        predicted_population=predicted_population,
        target_population=target_population,
        state_names=np.asarray(panel["state_names"]).astype(str),
    )
    metadata = {
        "method": "MIOFlow",
        "method_version": "0.1.14@36365403d0f23cc3ad1065781c7331bf81debf4e",
        "input": str(args.input.resolve()),
        "initial_input": str(args.initial_input.resolve()) if args.initial_input else None,
        "output": str(args.output.resolve()),
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "device": str(model.device),
        "generated_shape": list(generated.shape),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
