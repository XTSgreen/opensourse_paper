"""Standardize the frozen GSE228154 inverse-OT setup for pseudotime evaluation.

Reads the parent project's preparation outputs (source distribution, feature
tensor, observed transition operators, frozen UOT direction) and writes a
self-contained derived file for the pseudotime experiments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    prep = project_root / "output" / "03_data_preparation" / "output"
    drivers = project_root / "output" / "04_driver_inference" / "drivers_uot.json"
    sources = [prep / "distributions.npz", prep / "phi.npz", prep / "transition_ops.npz", drivers]
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(path)

    distributions = np.load(sources[0], allow_pickle=True)
    phi_payload = np.load(sources[1], allow_pickle=True)
    operators = np.load(sources[2], allow_pickle=True)
    frozen = json.loads(drivers.read_text(encoding="utf-8"))

    sites = sorted(operators.files)
    target_sites = distributions["b_sites"].item()
    phi = np.asarray(phi_payload["phi"], dtype=float)
    feature_names = [str(name) for name in phi_payload["feats"]]
    if phi.shape[-1] != len(feature_names):
        raise ValueError("feature names do not match phi")

    output = package_root / "data" / "derived" / "pseudotime"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "gse228154_inverse_ot.npz",
        source=np.asarray(distributions["a"], dtype=float),
        phi=phi,
        feature_names=np.array(feature_names, dtype=object),
        sites=np.array(sites, dtype=object),
        target_sites=np.stack([np.asarray(target_sites[site], dtype=float) for site in sites]),
        transition_ops=np.stack([np.asarray(operators[site], dtype=float) for site in sites]),
        theta_frozen=np.asarray(frozen["theta_best"], dtype=float),
    )
    manifest = {
        "dataset": "GSE228154",
        "purpose": "pseudotime evaluation on the frozen inverse-OT observation",
        "sites": sites,
        "feature_names": feature_names,
        "source_hashes": {str(path.relative_to(project_root)): sha256(path) for path in sources},
        "derived_file": "data/derived/pseudotime/gse228154_inverse_ot.npz",
    }
    (output / "gse228154_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sites": sites, "features": feature_names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
