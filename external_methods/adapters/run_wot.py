"""Run the pinned official Waddington-OT implementation on one common panel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import wot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    started = time.perf_counter()
    panel = np.load(args.input, allow_pickle=False)
    source = np.asarray(panel["source_composition"], dtype=float)
    target = np.asarray(panel["target_composition"], dtype=float)
    features = np.vstack([source, target])
    obs = pd.DataFrame(
        {
            "time": np.concatenate([np.zeros(len(source)), np.ones(len(target))]),
            "id": [f"source_{i}" for i in range(len(source))] + [f"target_{i}" for i in range(len(target))],
        },
        index=[f"cell_{i}" for i in range(len(features))],
    )
    adata = ad.AnnData(X=features, obs=obs)
    squared = np.sum(source**2, axis=1, keepdims=True) + np.sum(target**2, axis=1)[None, :] - 2.0 * source @ target.T
    squared = np.maximum(squared, 0.0)
    positive = squared[squared > 0]
    if len(positive):
        squared /= np.median(positive)
    model = wot.ot.OTModel(
        adata,
        day_field="time",
        epsilon=float(args.epsilon),
        lambda1=float(args.lambda1),
        lambda2=float(args.lambda2),
        local_pca=0,
    )
    transport = model.compute_transport_map(0.0, 1.0, cost_matrix=squared)
    coupling = np.asarray(transport.X, dtype=float)
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
        "method": "Waddington-OT",
        "method_version": "ca5e94f05699997b01cf5ae13383f9810f0613f6",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "epsilon": args.epsilon,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "seed": args.seed,
        "coupling_shape": list(coupling.shape),
        "coupling_total": float(coupling.sum()),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
