# External method environments

The benchmark uses the official repositories and commit pins listed in `SOURCE_PINS.yaml`. Runtime environments are intentionally excluded from the public repository because they contain platform-specific binaries and large package caches. Run `scripts/setup_external_methods.ps1` or `.sh` to clone and verify the exact source commits; add `-CreateEnvironments` on PowerShell or `--create-envs` in Bash to create all method environments from the YAML specifications here.

The four environment groups are:

1. `modern_scverse.yaml` for moscot and CellRank 2.
2. `legacy_ot.yaml` for Waddington-OT and LineageOT.
3. `neural_modern.yaml` for MIOFlow.
4. `neural_legacy.yaml` for TIGON and PRESCIENT.

The public status table records the exact commit, environment group and any adapter-level deviation. `NA-by-design`, dependency failure, timeout and numerical failure remain distinct statuses.
