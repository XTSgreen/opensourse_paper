"""Command-line entry points for protocol and release auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_project, write_audit
from .data_contracts import build_all_panels
from .external_evaluation import write_external_evaluation
from .protocol import default_protocol_path, load_protocol, protocol_sha256, write_lock
from .synthetic import run_suite
from .statistics import analyze_b1


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iot-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-protocol", help="Validate and hash the frozen protocol")
    verify.add_argument("--protocol", type=Path, default=default_protocol_path())

    lock = subparsers.add_parser("lock-protocol", help="Write the protocol hash lock")
    lock.add_argument("--protocol", type=Path, default=default_protocol_path())
    lock.add_argument("--output", type=Path, default=_package_root() / "configs" / "protocol_v1.lock.json")

    audit = subparsers.add_parser("audit", help="Audit local assets without evaluating outcomes")
    audit.add_argument("--project-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=_package_root() / "results" / "audits" / "asset_audit.json")

    b1 = subparsers.add_parser("run-b1", help="Run the known-truth identifiability benchmark")
    b1.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    b1.add_argument("--output", type=Path, default=_package_root() / "results" / "B1_known_truth")

    prepare = subparsers.add_parser("prepare-data", help="Build frozen method-neutral benchmark panels")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=_package_root() / "data" / "derived")

    evaluate = subparsers.add_parser("evaluate-external", help="Evaluate one normalized external coupling")
    evaluate.add_argument("--panel", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    analyze = subparsers.add_parser("analyze-b1", help="Run paired B1 statistics and gate checks")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-protocol":
        protocol = load_protocol(args.protocol)
        print(json.dumps({"valid": True, "version": protocol["protocol_version"], "sha256": protocol_sha256(args.protocol)}, indent=2))
        return 0
    if args.command == "lock-protocol":
        output = write_lock(args.output, args.protocol)
        print(output)
        return 0
    if args.command == "audit":
        report = audit_project(args.project_root, _package_root())
        output = write_audit(report, args.output)
        print(json.dumps({"output": str(output), "all_declared_local_assets_exist": report["all_declared_local_assets_exist"], "adapter_pending_count": report["adapter_pending_count"]}, indent=2))
        return 0
    if args.command == "run-b1":
        print(json.dumps(run_suite(args.output, smoke=args.mode == "smoke"), indent=2))
        return 0
    if args.command == "prepare-data":
        manifest = build_all_panels(args.project_root, args.output)
        print(json.dumps({"panels": len(manifest["panels"]), "all_boundary_checks_pass": manifest["all_boundary_checks_pass"]}, indent=2))
        return 0
    if args.command == "evaluate-external":
        output = write_external_evaluation(args.panel, args.artifact, args.output)
        print(output)
        return 0
    if args.command == "analyze-b1":
        print(json.dumps(analyze_b1(args.input, args.output), indent=2))
        return 0
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
