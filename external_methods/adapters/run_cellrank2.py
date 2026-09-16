"""Run pinned CellRank 2 on a precomputed temporal coupling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import cellrank as cr
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--coupling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    started = time.perf_counter()
    panel = np.load(args.panel, allow_pickle=False)
    coupling_artifact = np.load(args.coupling, allow_pickle=False)
    source = np.asarray(panel["source_composition"], dtype=float)
    target = np.asarray(panel["target_composition"], dtype=float)
    state_names = np.asarray(panel["state_names"]).astype(str)
    coupling = np.asarray(coupling_artifact["coupling"], dtype=float)
    coupling = coupling / np.maximum(coupling.sum(axis=1, keepdims=True), 1e-300)

    obs_names = [f"source_{i}" for i in range(len(source))] + [f"target_{i}" for i in range(len(target))]
    labels = np.concatenate([state_names[source.argmax(axis=1)], state_names[target.argmax(axis=1)]])
    obs = pd.DataFrame(
        {
            "time": pd.Categorical(np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])),
            "state": pd.Categorical(labels),
        },
        index=obs_names,
    )
    adata = ad.AnnData(X=np.vstack([source, target]), obs=obs)
    kernel = cr.kernels.RealTimeKernel(
        adata,
        time_key="time",
        couplings={(0, 1): coupling},
        policy="sequential",
    ).compute_transition_matrix(self_transitions="all", threshold=None, conn_weight=0.2)
    estimator = cr.estimators.GPCCA(kernel)
    terminal: dict[str, list[str]] = {}
    target_labels = labels[len(source) :]
    for state_name in state_names:
        indices = np.flatnonzero(target_labels == state_name)
        if len(indices):
            terminal[state_name] = [f"target_{int(index)}" for index in indices]
    estimator.set_terminal_states(terminal)
    estimator.compute_fate_probabilities(solver="direct", use_petsc=False, show_progress_bar=False)
    fate = np.asarray(estimator.fate_probabilities, dtype=float)[: len(source)]
    fate_names = np.asarray(estimator.fate_probabilities.names).astype(str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        fate_probabilities=fate,
        fate_names=fate_names,
        source_ids=np.asarray(panel["lineage_id"]).astype(str),
        state_names=state_names,
    )
    metadata = {
        "method": "CellRank 2",
        "method_version": "d7191d75684c86b58adbb317c8aae7d06f2682f3",
        "panel": str(args.panel.resolve()),
        "coupling": str(args.coupling.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "kernel": "RealTimeKernel with pinned moscot coupling",
        "terminal_states": {key: len(value) for key, value in terminal.items()},
        "fate_shape": list(fate.shape),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
