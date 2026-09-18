"""Run pseudotime evaluation experiments E1-E6 on frozen synthetic and real data."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from iot_benchmark.pseudotime import (
    chain_case,
    diffusion_pseudotime,
    hitting_time_pseudotime,
    jackknife_pseudotime_stability,
    pairwise_stability,
    rank_correlation,
    row_normalize_transition,
    state_coupling_from_clone,
    state_expected_time,
    time_separation_auc,
)
from iot_benchmark.synthetic import Observation, hard_ot_plan, soft_uot_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pseudotime"
LEGACY = ROOT / "figures" / "legacy_v5" / "technical_route_v6" / "source_data"
PANELS = {
    "gse140802_t2_t16": ROOT / "data" / "derived" / "gse140802_t2_t16.npz",
    "gse140802_t2_t9": ROOT / "data" / "derived" / "gse140802_t2_t9.npz",
    "gse239651_transition_panel": ROOT / "data" / "derived" / "gse239651_transition_panel.npz",
    "macsgestalt_transition_panel": ROOT / "data" / "derived" / "macsgestalt_transition_panel.npz",
}
METHODS = ["uot_iot", "wot", "moscot", "lineageot", "cellrank2", "tigon", "mioflow", "prescient"]
SEEDS = [20260916, 20260917, 20260918, 20260919, 20260920]
MU_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TIME_ORDER = ["D0", "D3", "D6", "D9"]
INPUTS: set[str] = set()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    totals = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(totals, 1e-300)


def plan_from_theta(phi, theta, observation, mode, *, mu=0.5, epsilon=1.0, iterations=500, tolerance=1e-10):
    if mode == "soft_iot":
        return soft_uot_plan(
            phi, theta, observation.source, observation.target_reference,
            epsilon=epsilon, mu=mu, iterations=iterations, tolerance=tolerance,
        )
    return hard_ot_plan(phi, theta, observation.source, observation.observed_plan.sum(axis=0), epsilon=epsilon)


def fit_theta(phi, observations, mode, seed, *, mu=0.5, restarts=3, maxiter=200, epsilon=1.0):
    def loss(theta):
        total = 0.0
        for observation in observations:
            plan = plan_from_theta(phi, theta, observation, mode, mu=mu, epsilon=epsilon)
            estimate = normalize_rows(plan)
            truth = normalize_rows(observation.observed_plan)
            row_weights = observation.observed_plan.sum(axis=1)
            row_weights = row_weights / max(float(row_weights.sum()), 1e-300)
            total += float(-np.sum(row_weights[:, None] * truth * np.log(np.maximum(estimate, 1e-300))))
        return total / len(observations)

    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(restarts):
        start = rng.normal(0.0, 0.2, size=phi.shape[-1])
        result = minimize(
            loss, start, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
        )
        fits.append((float(result.fun), np.asarray(result.x, dtype=float)))
    fits.sort(key=lambda item: item[0])
    return [vector for _, vector in fits]


def direction_curvature(phi, observations, mode, theta, index, *, mu=0.5, step=1e-3):
    def loss(value):
        total = 0.0
        for observation in observations:
            plan = plan_from_theta(phi, value, observation, mode, mu=mu)
            estimate = normalize_rows(plan)
            truth = normalize_rows(observation.observed_plan)
            total += float(-np.sum(truth * np.log(np.maximum(estimate, 1e-300))))
        return total / len(observations)

    direction = np.zeros_like(theta)
    direction[index] = 1.0
    return float(
        (loss(theta + step * direction) - 2.0 * loss(theta) + loss(theta - step * direction)) / (step**2)
    )


def transition_operator(phi, theta, observations, mode, *, mu=0.5):
    plans = [plan_from_theta(phi, theta, observation, mode, mu=mu) for observation in observations]
    return row_normalize_transition(np.sum(plans, axis=0))


def experiment1() -> pd.DataFrame:
    rows = []
    settings = [(0.0, 400), (0.0, 2000), (0.0, 10000), (0.15, 2000)]
    for dropout, sample_count in settings:
        for seed in SEEDS:
            phi, theta_true, train, held, truth = chain_case(
                states=6, sample_count=sample_count, seed=seed, dropout=dropout
            )
            for mode in ("soft_iot", "hard_ot"):
                started = time.perf_counter()
                restarts = fit_theta(phi, train, mode, seed + 1000, restarts=3)
                runtime = time.perf_counter() - started
                pseudotimes = []
                for vector in restarts:
                    transition = transition_operator(phi, vector, train, mode)
                    pseudotimes.append(hitting_time_pseudotime(transition, root=0))
                normalized = np.asarray(restarts) / np.maximum(
                    np.linalg.norm(restarts, axis=1, keepdims=True), 1e-12
                )
                rows.append({
                    "experiment": "E1_synthetic_chain",
                    "dropout": dropout,
                    "sample_count": sample_count,
                    "seed": seed,
                    "method": mode,
                    "runtime_seconds": runtime,
                    "ordering": rank_correlation(pseudotimes[0], truth),
                    "restart_stability": pairwise_stability(pseudotimes),
                    "direction_dispersion": float(np.mean(np.var(normalized, axis=0, ddof=1))),
                    "pure_column_curvature": direction_curvature(phi, train, mode, restarts[0], 4),
                })
    return pd.DataFrame(rows)


def load_panel(path: Path) -> dict:
    INPUTS.add(relative(path))
    data = np.load(path, allow_pickle=True)
    return {
        "source_counts": np.asarray(data["source_counts"], dtype=float),
        "target_counts": np.asarray(data["target_counts"], dtype=float),
        "source_composition": np.asarray(data["source_composition"], dtype=float),
        "target_composition": np.asarray(data["target_composition"], dtype=float),
        "source_time": np.asarray(data["source_time"]),
        "target_time": np.asarray(data["target_time"]),
        "state_names": list(map(str, data["state_names"])),
    }


def method_files(method: str, dataset: str) -> list[Path]:
    base = ROOT / "results" / "B2_lineage_transition" / method
    files = sorted(base.glob(f"{dataset}_seed*.npz"))
    if not files:
        single = base / f"{dataset}.npz"
        files = [single] if single.exists() else []
    return files


def unit_state_couplings(coupling: np.ndarray, source_composition: np.ndarray, target_composition: np.ndarray):
    contributions = []
    for index in range(len(coupling)):
        contributions.append(np.outer(source_composition[index], coupling[index] @ target_composition))
    return contributions


def experiment3() -> pd.DataFrame:
    rows = []
    for dataset, path in PANELS.items():
        panel = load_panel(path)
        source_counts = panel["source_counts"]
        target_counts = panel["target_counts"]
        source = panel["source_composition"]
        target = panel["target_composition"]
        root = int(np.argmax(source_counts.sum(axis=0)))
        expected_time = state_expected_time(
            source_counts, target_counts, panel["source_time"], panel["target_time"]
        )
        for method in METHODS:
            files = method_files(method, dataset)
            base_row = {"experiment": "E3_panels", "dataset": dataset, "method": method,
                        "n_states": len(panel["state_names"]), "root": root}
            if not files:
                rows.append({**base_row, "seed": -1, "status": "missing_artifact", "ordering": np.nan,
                             "time_auc": np.nan, "jackknife_stability": np.nan})
                continue
            for path in files:
                INPUTS.add(relative(path))
                data = np.load(path, allow_pickle=True)
                if "coupling" not in data.files:
                    rows.append({**base_row, "seed": -1, "status": "no_coupling", "ordering": np.nan,
                                 "time_auc": np.nan, "jackknife_stability": np.nan})
                    continue
                coupling = np.asarray(data["coupling"], dtype=float)
                state_coupling = state_coupling_from_clone(coupling, source, target)
                transition = row_normalize_transition(state_coupling)
                pseudotime = hitting_time_pseudotime(transition, root)
                contributions = unit_state_couplings(coupling, source, target)
                jackknife = jackknife_pseudotime_stability(contributions, [1.0] * len(contributions), root)
                seed_value = -1
                if "_seed" in path.stem:
                    seed_value = int(path.stem.split("_seed")[-1])
                rows.append({**base_row, "seed": seed_value, "status": "success",
                             "ordering": rank_correlation(pseudotime, expected_time),
                             "time_auc": time_separation_auc(pseudotime, source_counts, target_counts),
                             "jackknife_stability": jackknife})
        source_distribution = source_counts.sum(axis=0)
        source_distribution = source_distribution / max(source_distribution.sum(), 1e-300)
        target_distribution = target_counts.sum(axis=0)
        target_distribution = target_distribution / max(target_distribution.sum(), 1e-300)
        rows.append({**base_row, "seed": -1, "status": "baseline",
                     "ordering": rank_correlation(target_distribution, expected_time),
                     "time_auc": time_separation_auc(target_distribution, source_counts, target_counts),
                     "jackknife_stability": np.nan})
    return pd.DataFrame(rows)


def gse228154_frames() -> pd.DataFrame:
    cells_path = LEGACY / "GSE228154_cells.csv"
    pca_path = LEGACY / "GSE228154_R_PCA_coordinates.csv"
    for path in (cells_path, pca_path):
        INPUTS.add(relative(path))
    cells = pd.read_csv(cells_path)
    pca = pd.read_csv(pca_path)
    return cells.merge(pca, on=["cell", "timepoint", "state"], how="inner")


def gse228154_inverse_ot() -> dict:
    path = ROOT / "data" / "derived" / "pseudotime" / "gse228154_inverse_ot.npz"
    INPUTS.add(relative(path))
    data = np.load(path, allow_pickle=True)
    return {
        "source": np.asarray(data["source"], dtype=float),
        "phi": np.asarray(data["phi"], dtype=float),
        "feature_names": [str(name) for name in data["feature_names"]],
        "sites": [str(site) for site in data["sites"]],
        "target_sites": np.asarray(data["target_sites"], dtype=float),
        "transition_ops": np.asarray(data["transition_ops"], dtype=float),
        "theta_frozen": np.asarray(data["theta_frozen"], dtype=float),
    }


def zscore_phi(phi: np.ndarray) -> np.ndarray:
    flat = phi.reshape(-1, phi.shape[-1])
    scaled = (flat - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-12)
    return scaled.reshape(phi.shape)


def gse228154_observations(payload: dict, *, shuffle_targets: bool = False, seed: int = 0) -> list[Observation]:
    observations = []
    rng = np.random.default_rng(seed)
    for index, _ in enumerate(payload["sites"]):
        operator = payload["transition_ops"][index].copy()
        target = payload["target_sites"][index]
        if shuffle_targets:
            permutation = rng.permutation(operator.shape[1])
            operator = operator[:, permutation]
            target = target[permutation]
        operator = operator / np.maximum(operator.sum(axis=1, keepdims=True), 1e-300)
        observations.append(Observation(payload["source"], target, operator, operator, 0))
    return observations


def gse228154_state_day(frame: pd.DataFrame) -> np.ndarray:
    day_index = frame.timepoint.map({name: index for index, name in enumerate(TIME_ORDER)}).to_numpy(float)
    return (
        frame.groupby("state")
        .apply(lambda part: day_index[part.index].mean(), include_groups=False)
        .sort_index()
        .to_numpy(float)
    )


def experiment2() -> pd.DataFrame:
    payload = gse228154_inverse_ot()
    phi = zscore_phi(payload["phi"])
    observations = gse228154_observations(payload)
    frame = gse228154_frames()
    day_index = frame.timepoint.map({name: index for index, name in enumerate(TIME_ORDER)}).to_numpy(float)
    state_day = gse228154_state_day(frame)
    state = frame.state.to_numpy(int)
    root = int(np.argmax(payload["source"]))
    pooled_target = payload["target_sites"].sum(axis=0)
    rows = []
    for mode in ("soft_iot", "hard_ot"):
        all_pseudotimes = []
        for seed in SEEDS[:3]:
            restarts = fit_theta(phi, observations, mode, seed, restarts=3)
            for vector in restarts:
                transition = transition_operator(phi, vector, observations, mode)
                all_pseudotimes.append(hitting_time_pseudotime(transition, root))
        best = all_pseudotimes[0]
        site_scores = []
        for drop in range(len(observations)):
            subset = [item for index, item in enumerate(observations) if index != drop]
            transition = transition_operator(phi, fit_theta(phi, subset, mode, SEEDS[0], restarts=2)[0], subset, mode)
            site_scores.append(rank_correlation(best, hitting_time_pseudotime(transition, root)))
        rows.append({
            "experiment": "E2_gse228154", "method": mode, "seed": -1, "status": "summary",
            "ordering_state": rank_correlation(best, state_day),
            "ordering_cell": rank_correlation(best[state], day_index),
            "time_auc": time_separation_auc(best, payload["source"][None, :], pooled_target[None, :]),
            "restart_stability": pairwise_stability(all_pseudotimes),
            "site_stability": float(np.nanmean(site_scores)),
            "root": root,
        })
    frozen = hitting_time_pseudotime(transition_operator(phi, payload["theta_frozen"], observations, "soft_iot"), root)
    rows.append({
        "experiment": "E2_gse228154", "method": "frozen_uot_direction", "seed": -1, "status": "summary",
        "ordering_state": rank_correlation(frozen, state_day),
        "ordering_cell": rank_correlation(frozen[state], day_index),
        "time_auc": time_separation_auc(frozen, payload["source"][None, :], pooled_target[None, :]),
        "restart_stability": np.nan, "site_stability": np.nan, "root": root,
    })
    pca = frame[["PC1", "PC2"]].to_numpy(float)
    d0 = frame[frame.timepoint.eq("D0")][["PC1", "PC2"]].to_numpy(float)
    root_cell = int(np.argmin(np.linalg.norm(pca - d0.mean(axis=0)[None, :], axis=1)))
    dpt = diffusion_pseudotime(pca, root_cell, neighbors=15)
    soft_reference = hitting_time_pseudotime(
        transition_operator(phi, fit_theta(phi, observations, "soft_iot", SEEDS[0], restarts=2)[0], observations, "soft_iot"),
        root,
    )
    rows.append({
        "experiment": "E2_gse228154", "method": "dpt_style", "seed": -1, "status": "baseline",
        "ordering_state": np.nan,
        "ordering_cell": rank_correlation(dpt, day_index),
        "time_auc": np.nan, "restart_stability": np.nan, "site_stability": np.nan,
        "root": root_cell, "agreement_with_uot": rank_correlation(dpt, soft_reference[state]),
    })
    marginal_pseudotime = pooled_target / max(pooled_target.sum(), 1e-300)
    rows.append({
        "experiment": "E2_gse228154", "method": "marginal_prevalence", "seed": -1, "status": "baseline",
        "ordering_state": rank_correlation(marginal_pseudotime, state_day),
        "ordering_cell": rank_correlation(marginal_pseudotime[state], day_index),
        "time_auc": time_separation_auc(marginal_pseudotime, payload["source"][None, :], pooled_target[None, :]),
        "restart_stability": np.nan, "site_stability": np.nan, "root": root,
    })
    return pd.DataFrame(rows)


def experiment5() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        phi, theta_true, train, held, truth = chain_case(states=6, sample_count=2000, seed=seed)
        for mu in MU_GRID:
            restarts = fit_theta(phi, train, "soft_iot", seed + 2000, mu=mu, restarts=3)
            pseudotimes = [
                hitting_time_pseudotime(transition_operator(phi, vector, train, "soft_iot", mu=mu), 0)
                for vector in restarts
            ]
            normalized = np.asarray(restarts) / np.maximum(
                np.linalg.norm(restarts, axis=1, keepdims=True), 1e-12
            )
            rows.append({
                "experiment": "E5_mu_scan", "dataset": "synthetic_chain", "mu": mu, "seed": seed,
                "ordering": rank_correlation(pseudotimes[0], truth),
                "restart_stability": pairwise_stability(pseudotimes),
                "min_pure_column_curvature": direction_curvature(phi, train, "soft_iot", restarts[0], 4, mu=mu),
                "direction_dispersion": float(np.mean(np.var(normalized, axis=0, ddof=1))),
            })
    payload = gse228154_inverse_ot()
    phi = zscore_phi(payload["phi"])
    observations = gse228154_observations(payload)
    frame = gse228154_frames()
    state_day = gse228154_state_day(frame)
    root = int(np.argmax(payload["source"]))
    pooled_target = payload["target_sites"].sum(axis=0)
    pure_columns = [index for index, name in enumerate(payload["feature_names"]) if name.endswith("_j")]
    for mu in MU_GRID:
        restarts = fit_theta(phi, observations, "soft_iot", SEEDS[0] + 3000, mu=mu, restarts=3)
        pseudotimes = [
            hitting_time_pseudotime(transition_operator(phi, vector, observations, "soft_iot", mu=mu), root)
            for vector in restarts
        ]
        curvatures = [
            direction_curvature(phi, observations, "soft_iot", restarts[0], index, mu=mu)
            for index in pure_columns
        ]
        normalized = np.asarray(restarts) / np.maximum(
            np.linalg.norm(restarts, axis=1, keepdims=True), 1e-12
        )
        rows.append({
            "experiment": "E5_mu_scan", "dataset": "gse228154", "mu": mu, "seed": SEEDS[0],
            "ordering": rank_correlation(pseudotimes[0], state_day),
            "time_auc": time_separation_auc(pseudotimes[0], payload["source"][None, :], pooled_target[None, :]),
            "restart_stability": pairwise_stability(pseudotimes),
            "min_pure_column_curvature": float(np.min(curvatures)),
            "direction_dispersion": float(np.mean(np.var(normalized, axis=0, ddof=1))),
        })
    return pd.DataFrame(rows)


def experiment6() -> pd.DataFrame:
    rows = []
    phi, theta_true, train, held, truth = chain_case(states=6, sample_count=2000, seed=SEEDS[0])
    transition = transition_operator(phi, fit_theta(phi, train, "soft_iot", 42, restarts=2)[0], train, "soft_iot")
    rows.append({"experiment": "E6_controls", "dataset": "synthetic_chain", "method": "soft_iot",
                 "control": "random_root", "ordering": rank_correlation(hitting_time_pseudotime(transition, 3), truth)})
    permuted = np.random.default_rng(5).permutation(truth)
    rows.append({"experiment": "E6_controls", "dataset": "synthetic_chain", "method": "soft_iot",
                 "control": "permuted_truth",
                 "ordering": rank_correlation(hitting_time_pseudotime(transition, 0), permuted)})
    payload = gse228154_inverse_ot()
    phi_real = zscore_phi(payload["phi"])
    observations = gse228154_observations(payload)
    frame = gse228154_frames()
    state_day = gse228154_state_day(frame)
    state = frame.state.to_numpy(int)
    day_index = frame.timepoint.map({name: index for index, name in enumerate(TIME_ORDER)}).to_numpy(float)
    root = int(np.argmax(payload["source"]))
    transition = transition_operator(phi_real, fit_theta(phi_real, observations, "soft_iot", 77, restarts=2)[0],
                                     observations, "soft_iot")
    rows.append({"experiment": "E6_controls", "dataset": "gse228154", "method": "soft_iot",
                 "control": "random_root",
                 "ordering": rank_correlation(hitting_time_pseudotime(transition, (root + 5) % 10), state_day)})
    shuffled = gse228154_observations(payload, shuffle_targets=True, seed=78)
    shuffled_transition = transition_operator(
        phi_real, fit_theta(phi_real, shuffled, "soft_iot", 79, restarts=2)[0], shuffled, "soft_iot"
    )
    rows.append({"experiment": "E6_controls", "dataset": "gse228154", "method": "soft_iot",
                 "control": "shuffled_target_labels",
                 "ordering": rank_correlation(hitting_time_pseudotime(shuffled_transition, root)[state], day_index)})
    panel = load_panel(PANELS["gse140802_t2_t16"])
    files = method_files("uot_iot", "gse140802_t2_t16")
    data = np.load(files[0], allow_pickle=True)
    coupling = np.asarray(data["coupling"], dtype=float)
    root = int(np.argmax(panel["source_counts"].sum(axis=0)))
    transition = row_normalize_transition(state_coupling_from_clone(
        coupling, panel["source_composition"], panel["target_composition"]))
    expected = state_expected_time(panel["source_counts"], panel["target_counts"],
                                   panel["source_time"], panel["target_time"])
    rows.append({"experiment": "E6_controls", "dataset": "gse140802_t2_t16", "method": "uot_iot",
                 "control": "random_root",
                 "ordering": rank_correlation(hitting_time_pseudotime(transition, (root + 3) % 7), expected)})
    rng = np.random.default_rng(7)
    permutation = rng.permutation(coupling.shape[1])
    shuffled_coupling = coupling[:, permutation]
    shuffled_transition = row_normalize_transition(state_coupling_from_clone(
        shuffled_coupling, panel["source_composition"], panel["target_composition"]))
    rows.append({"experiment": "E6_controls", "dataset": "gse140802_t2_t16", "method": "uot_iot",
                 "control": "shuffled_coupling",
                 "ordering": rank_correlation(hitting_time_pseudotime(shuffled_transition, root), expected)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tables = {
        "e1_synthetic.csv": experiment1(),
        "e2_gse228154.csv": experiment2(),
        "e3_panels.csv": experiment3(),
        "e5_mu_scan.csv": experiment5(),
        "e6_controls.csv": experiment6(),
    }
    for name, table in tables.items():
        table.to_csv(OUT / name, index=False)
        print(name, len(table), "rows", flush=True)
    manifest = {
        "script": relative(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "protocol_reference": "configs/protocol_pseudotime_v1.yaml",
        "seeds": SEEDS,
        "mu_grid": MU_GRID,
        "panels": {name: relative(path) for name, path in PANELS.items()},
        "inputs": {path: sha256(ROOT / path) for path in sorted(INPUTS)},
        "outputs": sorted(tables),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "pseudotime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
