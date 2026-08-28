"""Run a repeatable deployment-mode force/compliance sweep."""

import argparse
import csv
from pathlib import Path

from ..paths import PROJECT_ROOT
from ..run_deployment import run
from .metrics import compliance_metrics


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def run_sweep(compliances, forces, axes=("x",), output_dir=None,
              duration_s=1.5):
    if not compliances or not forces or not axes:
        raise ValueError("compliances, forces and axes cannot be empty")
    output_dir = Path(output_dir or PROJECT_ROOT / "runs" / "deployment_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    first = True
    for axis in axes:
        if axis not in AXIS_INDEX:
            raise ValueError("axes must contain only x, y or z")
        for compliance in compliances:
            if compliance <= 0.0:
                raise ValueError("sweep compliance must be positive")
            for force_magnitude in forces:
                force = [0.0, 0.0, 0.0]
                compliance_vector = [0.0, 0.0, 0.0]
                force[AXIS_INDEX[axis]] = float(force_magnitude)
                compliance_vector[AXIS_INDEX[axis]] = float(compliance)
                name = "axis-{}_c-{:g}_f-{:g}.csv".format(axis, compliance, force_magnitude)
                csv_path = output_dir / name
                run(
                    headless=True,
                    duration_s=duration_s,
                    rebuild=first,
                    compliance=compliance_vector,
                    force=force,
                    record_path=csv_path,
                    quiet=True,
                )
                first = False
                metrics = compliance_metrics(csv_path, compliance, axis)
                metrics["commanded_force_n"] = float(force_magnitude)
                metrics["csv"] = str(csv_path.resolve())
                summary.append(metrics)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compliances", nargs="+", type=float, default=[0.001, 0.002, 0.004])
    parser.add_argument("--forces", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--axes", nargs="+", choices=("x", "y", "z"), default=["x"])
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path, rows = run_sweep(
        args.compliances, args.forces, args.axes, args.output_dir, args.duration
    )
    maximum_error = max(row["relative_error"] for row in rows)
    print("sweep: cases={} max_relative_error={:.2f}% summary={}".format(
        len(rows), 100.0 * maximum_error, summary_path.resolve()
    ))


if __name__ == "__main__":
    main()
