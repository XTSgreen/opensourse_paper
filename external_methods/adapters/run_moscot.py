"""Run the pinned official moscot TemporalProblem on one common panel."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import anndata as ad
import numpy as np
import pandas as pd
from moscot.problems.time import TemporalProblem


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--epsilon", type=float, default=0.05)
    result.add_argument("--tau-a", type=float, default=1.0)
    result.add_argument("--tau-b", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=20260916)
    return result


def main() -> int:
    args = parser().parse_args()
    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = np.asarray(panel["source_composition"], dtype=float)
    target = np.asarray(panel["target_composition"], dtype=float)
    features = np.vstack([source, target])
    obs = pd.DataFrame(
        {
            "time": np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]),
            "row_id": [f"source_{i}" for i in range(len(source))] + [f"target_{i}" for i in range(len(target))],
        }
    )
    adata = ad.AnnData(X=features, obs=obs)
    problem = TemporalProblem(adata).prepare(
        "time",
        policy="sequential",
        joint_attr={"attr": "X"},
    )
    problem = problem.solve(
        epsilon=float(args.epsilon),
        tau_a=float(args.tau_a),
        tau_b=float(args.tau_b),
        rank=-1,
        scale_cost="mean",
        max_iterations=5000,
        threshold=1e-6,
        jit=True,
    )
    subproblem = problem[0, 1]
    coupling = np.asarray(subproblem.solution.transport_matrix, dtype=float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coupling=coupling,
        source_composition=source,
        target_composition=target,
        source_ids=np.asarray(panel["lineage_id"]).astype(str),
        target_ids=np.asarray(panel["lineage_id"]).astype(str),
        state_names=np.asarray(panel["state_names"]).astype(str),
    )
    metadata = {
        "method": "moscot",
        "method_version": "0.5.2@440093ccbb8e70de209157d91da839c55b897821",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "epsilon": args.epsilon,
        "tau_a": args.tau_a,
        "tau_b": args.tau_b,
        "seed": args.seed,
        "converged": bool(subproblem.solution.converged),
        "coupling_shape": list(coupling.shape),
        "coupling_total": float(coupling.sum()),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
