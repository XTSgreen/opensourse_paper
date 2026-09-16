"""Run UOT-IOT on every frozen B2 panel across the five protocol seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.external_evaluation import write_external_evaluation
from iot_benchmark.models import UOTIOTBenchmarkModel


SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.package_root.resolve()
    panels = sorted((root / "data" / "derived").glob("*_panel.npz")) + sorted(
        (root / "data" / "derived").glob("gse140802_*.npz")
    )
    output_root = root / "results" / "B2_lineage_transition" / "uot_iot"
    rows = []
    for panel_path in panels:
        panel = np.load(panel_path, allow_pickle=False)
        for seed in SEEDS:
            started = time.perf_counter()
            model = UOTIOTBenchmarkModel().fit(
                panel["source_composition"], panel["target_composition"], panel["group"],
                panel["source_counts"], panel["target_counts"], seed=seed,
            )
            source_mass = np.sqrt(panel["source_counts"].sum(axis=1))
            target_mass = np.sqrt(panel["target_counts"].sum(axis=1))
            coupling = model.clone_coupling(
                panel["source_composition"], panel["target_composition"], source_mass, target_mass
            )
            direction, magnitude = model.direction()
            output = output_root / f"{panel_path.stem}_seed{seed}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                coupling=coupling,
                direction=direction,
                magnitude=magnitude,
                feature_names=np.asarray(model.feature_names_),
                state_prediction=model.predict_state(panel["source_composition"]),
            )
            metric_path = output.with_name(output.stem + "_metrics.json")
            write_external_evaluation(panel_path, output, metric_path)
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": panel_path.stem,
                    "seed": seed,
                    "method": "UOT-IOT",
                    "objective": model.objective_,
                    "runtime_seconds": time.perf_counter() - started,
                    "artifact": str(output),
                    **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "uot_iot_all_runs.csv", index=False)
    summary = frame.groupby("dataset", as_index=False).mean(numeric_only=True)
    summary.to_csv(output_root / "uot_iot_summary.csv", index=False)
    print(json.dumps({"panels": len(panels), "runs": len(frame), "seeds": SEEDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
