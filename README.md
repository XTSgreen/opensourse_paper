# IOT benchmark and reproducibility package

This directory is the clean-room revision package for UOT-IOT, GH-IOT and PERSIST-IOT. It is separated from the historical analysis archive and uses a frozen protocol hash, method-level source pins, common data contracts and reviewer-facing result schemas. The executable GH-IOT and PERSIST-IOT implementations and the frozen processed source panels are included in the package and do not import model code from the historical project tree. Full raw GEO archives are optional for rebuilding the frozen panels. Release, DOI and licensing metadata will be frozen after all benchmark gates pass.

## Current executable scope

1. B1 includes six known-truth scenarios, three sample sizes, five seeds, paired bootstrap inference and explicit parameter-recovery gates.
2. B2 includes Waddington-OT, moscot, LineageOT, CellRank 2, TIGON, MIOFlow, PRESCIENT and UOT-IOT adapters against common lineage panels.
3. B3 physically separates expt1 training data, expt2 prediction-time input and locked expt2 future truth, and adds leave-one-animal-out GSE140802 panels. Composition is evaluated with cross-entropy, state-cost Sinkhorn divergence, top-1 accuracy, Brier score and pooled state-wise expected calibration error.
4. Reviewer tables retain successes, failures, timeouts and `NA-by-design` outcomes instead of silently dropping methods.

## Quick start

```powershell
python -m pip install -e ".[test]"
python scripts/reproduce.py prepare
python scripts/reproduce.py smoke
```

Run `python scripts/reproduce.py all` for the complete benchmark and Figures 1–5 pipeline after the pinned external environments have been created. See `tutorials/quickstart.md` for the staged workflow. Historical output files remain read-only provenance inputs for the locked PERSIST experiment and frozen-direction external validation. No result should be cited as final before `results/FINAL_GATE_STATUS.json` reports the relevant gate as passed.
