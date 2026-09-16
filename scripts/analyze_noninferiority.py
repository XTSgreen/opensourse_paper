"""Apply the frozen paired non-inferiority margins to B1, B2 and B3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from iot_benchmark.statistics import paired_bootstrap_interval


LOWER_RELATIVE = {
    "cross_entropy": 0.05,
    "sinkhorn_divergence": 0.05,
    "coupling_relative_frobenius": 0.05,
    "transition_weighted_l1": 0.05,
}
HIGHER_ABSOLUTE = {"top1_accuracy": 0.03, "fate_spearman": 0.05}


def _record(benchmark: str, metric: str, comparator: str, iot: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if metric in LOWER_RELATIVE:
        margin = LOWER_RELATIVE[metric]
        scores = reference * (1.0 + margin) - iot
        definition = "reference*(1+relative_margin)-IOT"
    else:
        margin = HIGHER_ABSOLUTE[metric]
        scores = iot - reference + margin
        definition = "IOT-reference+absolute_margin"
    mean, lower, upper = paired_bootstrap_interval(scores)
    return {
        "benchmark": benchmark,
        "metric": metric,
        "comparator": comparator,
        "paired_n": len(scores),
        "margin": margin,
        "score_definition": definition,
        "mean_margin_score": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "noninferior": bool(lower >= 0),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "noninferiority"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    b1 = pd.read_csv(root / "results" / "B1_known_truth_final" / "b1_all_runs.csv")
    keys = ["scenario", "sample_size", "sample_count", "seed"]
    soft = b1.loc[b1.method.eq("soft_iot")].set_index(keys)
    hard = b1.loc[b1.method.eq("hard_ot")].set_index(keys)
    common = soft.index.intersection(hard.index)
    rows.append(_record(
        "B1_known_truth", "coupling_relative_frobenius", "hard_ot",
        soft.loc[common, "heldout_coupling_relative_frobenius"].to_numpy(),
        hard.loc[common, "heldout_coupling_relative_frobenius"].to_numpy(),
    ))

    b2 = pd.read_csv(root / "results" / "B2_lineage_transition" / "b2_all_runs.csv")
    metrics = ["coupling_relative_frobenius", "transition_weighted_l1", "fate_spearman", "fate_top1_accuracy"]
    means = b2.groupby(["dataset", "method"], as_index=False)[metrics].mean(numeric_only=True)
    iot = means.loc[means.method.eq("UOT-IOT")].set_index("dataset")
    for method in sorted(set(means.method) - {"UOT-IOT"}):
        reference = means.loc[means.method.eq(method)].set_index("dataset")
        for metric, frozen_name in [
            ("coupling_relative_frobenius", "coupling_relative_frobenius"),
            ("transition_weighted_l1", "transition_weighted_l1"),
            ("fate_spearman", "fate_spearman"),
            ("fate_top1_accuracy", "top1_accuracy"),
        ]:
            common = iot.index.intersection(reference.index)
            valid = np.isfinite(iot.loc[common, metric]) & np.isfinite(reference.loc[common, metric])
            common = common[valid]
            if len(common) < 2:
                continue
            rows.append(_record(
                "B2_lineage_transition", frozen_name, method,
                iot.loc[common, metric].to_numpy(), reference.loc[common, metric].to_numpy(),
            ))

    b3 = pd.read_csv(root / "results" / "B3_prospective_composition" / "b3_state_prediction_metrics.csv")
    persist = b3.loc[b3.method.eq("PERSIST-IOT")].set_index("dataset")
    ablated = b3.loc[b3.method.eq("PERSIST-no-IOT")].set_index("dataset")
    common = persist.index.intersection(ablated.index)
    for column, frozen_name in [
        ("cross_entropy", "cross_entropy"),
        ("sinkhorn_divergence", "sinkhorn_divergence"),
        ("top1_accuracy", "top1_accuracy"),
    ]:
        rows.append(_record(
            "B3_prospective_composition", frozen_name, "PERSIST-no-IOT",
            persist.loc[common, column].to_numpy(), ablated.loc[common, column].to_numpy(),
        ))

    statistics = pd.DataFrame(rows)
    statistics.to_csv(output / "paired_noninferiority.csv", index=False)
    b1_pass = bool(statistics.loc[statistics.benchmark.eq("B1_known_truth"), "noninferior"].all())
    b2_primary = statistics.loc[
        statistics.benchmark.eq("B2_lineage_transition")
        & statistics.metric.isin(["coupling_relative_frobenius", "transition_weighted_l1"])
    ]
    b2_pass = bool(len(b2_primary) > 0 and b2_primary["noninferior"].all())
    b3_rows = statistics.loc[statistics.benchmark.eq("B3_prospective_composition")]
    b3_pass = bool(len(b3_rows) > 0 and b3_rows["noninferior"].all())
    decisions = {"B1_known_truth": b1_pass, "B2_lineage_transition": b2_pass,
                 "B3_prospective_composition": b3_pass}
    manifest = {
        "gate_pass": sum(decisions.values()) >= 2,
        "minimum_benchmarks_required": 2,
        "benchmark_decisions": decisions,
        "benchmarks_passed": int(sum(decisions.values())),
        "bootstrap_replicates": 10000,
        "scope_note": (
            "B3 is a paired PERSIST-IOT ablation comparison. External dynamics methods are reported separately "
            "because they predict population composition rather than clone-level state probabilities."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
