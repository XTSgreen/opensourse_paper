"""Run the pinned official LineageOT implementation on clone-aggregated panels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import lineageot
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = np.asarray(panel["source_composition"], dtype=float)
    target = np.asarray(panel["target_composition"], dtype=float)
    if len(source) != len(target):
        raise ValueError("clone-aggregated LineageOT adapter requires paired source and target rows")
    features = np.vstack([source, target])
    obs = pd.DataFrame(
        {"time": np.concatenate([np.zeros(len(source)), np.ones(len(target))])},
        index=[f"source_{i}" for i in range(len(source))] + [f"target_{i}" for i in range(len(target))],
    )
    adata = ad.AnnData(X=features, obs=obs)
    target_view = adata[adata.obs["time"] == 1.0].copy()
    target_view.obsm["X_clone"] = np.eye(len(target), dtype=bool)
    tree = lineageot.fit_tree(target_view, 1.0, method="non-nested clones")
    coupling_adata = lineageot.fit_lineage_coupling(
        adata,
        0.0,
        1.0,
        tree,
        time_key="time",
        epsilon=float(args.epsilon),
        normalize_cost=True,
        marginal_1=np.full(len(source), 1.0 / len(source)),
        marginal_2=np.full(len(target), 1.0 / len(target)),
    )
    coupling = np.asarray(coupling_adata.X, dtype=float)
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
        "method": "LineageOT",
        "method_version": "0.2.0@6081b402074f7e5934e729e81669aef430219da8",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "epsilon": args.epsilon,
        "seed": args.seed,
        "adapter_input_mode": "clone_aggregate_non_nested_tree",
        "coupling_shape": list(coupling.shape),
        "coupling_total": float(coupling.sum()),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
