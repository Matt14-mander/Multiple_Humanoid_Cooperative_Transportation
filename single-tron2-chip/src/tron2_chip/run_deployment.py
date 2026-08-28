"""Run the fixed-base deployment-mode compliance test in MuJoCo."""

import argparse
from pathlib import Path

from .paths import DEFAULT_CONFIG, GENERATED_MODEL, PROJECT_ROOT
from .run_sanity import run as run_rollout


def run(config_path=DEFAULT_CONFIG, model_path=GENERATED_MODEL, headless=False,
        duration_s=None, rebuild=False, compliance=None, force=None,
        record_path=None, quiet=False):
    output = record_path or PROJECT_ROOT / "runs" / "deployment_latest.csv"
    return run_rollout(
        config_path=config_path,
        model_path=model_path,
        headless=headless,
        duration_s=duration_s,
        rebuild=rebuild,
        compliance=compliance,
        force=force,
        record_path=output,
        goal_mode="deployment",
        quiet=quiet,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=GENERATED_MODEL)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--compliance", type=float, nargs=3)
    parser.add_argument("--force", type=float, nargs=3)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    run(args.config, args.model, args.headless, args.duration, args.rebuild,
        args.compliance, args.force, args.record)


if __name__ == "__main__":
    main()

