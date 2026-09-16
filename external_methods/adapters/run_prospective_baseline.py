"""Generate transparent prospective population baselines for B3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--initial-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("source-carry-forward", "development-target-transfer"), required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    train = np.load(args.input, allow_pickle=False)
    test_input = np.load(args.initial_input, allow_pickle=False)
    if args.method == "source-carry-forward":
        counts = np.asarray(test_input["source_counts"], dtype=float).sum(axis=0)
    else:
        counts = np.asarray(train["target_counts"], dtype=float).sum(axis=0)
    prediction = counts / counts.sum()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predicted_population=prediction, state_names=train["state_names"])
    metadata = {
        "method": args.method,
        "input": str(args.input.resolve()),
        "initial_input": str(args.initial_input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "target_information_used": "training-panel target composition" if args.method == "development-target-transfer" else "none",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
