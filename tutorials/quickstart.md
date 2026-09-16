# IOT reproducibility quickstart

This package separates three information modes: retrospective transition reconstruction, strictly prospective composition prediction, and oracle-only diagnostics. Do not compare metrics across these modes as if they were one ranking.

## 1. Main environment

Create the main environment from the pinned file, activate it, and install the package:

```bash
conda env create -f environment.lock.yml
conda activate iot-reproducibility
python scripts/reproduce.py setup
```

External methods use the four method-level specifications in `external_methods/environments/`. Their official source commits are fixed in `external_methods/SOURCE_PINS.yaml`; `scripts/setup_external_methods.ps1` or `.sh` clones the repositories, checks out the pinned commits, and verifies each HEAD. Pass `-CreateEnvironments` to the PowerShell script or `--create-envs` to the shell script to create all four Conda environments. Runtime directories themselves are intentionally excluded from version control.

## 2. Data preparation

Frozen, processed source panels are included under `data/processed/` with SHA-256 provenance; the preparation commands rebuild the method-neutral benchmark panels from these packaged files. Full GEO archives are optional and can be fetched with `python data/download/download_public_data.py` if raw-accession checks are required.

Run:

```bash
python scripts/reproduce.py prepare
```

The command writes SHA-256 manifests and physically separates B3 training input, prediction-time input and locked future truth. The packaged workflow includes the GSE239651 cross-experiment panels and deterministic held-out-animal GSE140802 panels. A method process must receive only the training and prediction-time files.

## 3. Minimal verification

```bash
python scripts/reproduce.py smoke
```

This checks the frozen protocol, data contracts, probability normalization, gauge rules, B3 leakage boundary and a small known-truth experiment.

## 4. Full benchmark

```bash
python scripts/reproduce.py benchmarks
```

The full neural runs are compute-intensive. Every method writes a status row and log; failed, timed-out and not-applicable runs remain visible.

If a long B3 run is interrupted after writing a checkpoint, resume it with:

```bash
python scripts/run_b3_external.py --resume
```

## 5. Reviewer tables and figures

```bash
python scripts/reproduce.py outputs
```

Reviewer-facing tables are written under `results/reviewer_tables/`. Figure outputs and their source-data manifest are written under `figures/` after the final benchmark has completed.
