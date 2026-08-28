"""Plot the CSV produced by the single-TRON2 CHIP sanity rollout."""

import argparse
import csv
from pathlib import Path

import numpy as np

from .paths import PROJECT_ROOT


AXES = ("x", "y", "z")
DEFAULT_CSV = PROJECT_ROOT / "runs" / "arm_sanity_latest.csv"


def load_rollout_csv(path: Path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames or []
    required = {
        "time_s", "ee_x", "ee_y", "ee_z", "ref_x", "ref_y", "ref_z",
        "actor_goal_x", "actor_goal_y", "actor_goal_z", "force_x_n",
        "force_y_n", "force_z_n", "max_abs_ctrl",
    }
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))
    if not rows:
        raise ValueError("CSV contains no samples: " + str(path))
    result = {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in required
    }
    if "max_control_fraction" in columns:
        result["max_control_fraction"] = np.asarray(
            [float(row["max_control_fraction"]) for row in rows], dtype=float
        )
    return result


def dominant_force_axis(data) -> str:
    peaks = [np.max(np.abs(data["force_{}_n".format(axis)])) for axis in AXES]
    return AXES[int(np.argmax(peaks))]


def _force_intervals(time_s, active):
    intervals = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = time_s[index]
        if start is not None and (not value or index == len(active) - 1):
            end_index = index if not value else index
            intervals.append((start, time_s[end_index]))
            start = None
    return intervals


def plot_rollout(csv_path=DEFAULT_CSV, output_path=None, axis="auto",
                 control_limit=80.0, show=False):
    import matplotlib.pyplot as plt

    data = load_rollout_csv(csv_path)
    selected_axis = dominant_force_axis(data) if axis == "auto" else axis.lower()
    if selected_axis not in AXES:
        raise ValueError("axis must be auto, x, y or z")
    time_s = data["time_s"]
    force_norm = np.sqrt(sum(data["force_{}_n".format(item)] ** 2 for item in AXES))
    active = force_norm > 1e-9
    intervals = _force_intervals(time_s, active)

    ee = data["ee_{}".format(selected_axis)]
    reference = data["ref_{}".format(selected_axis)]
    actor_goal = data["actor_goal_{}".format(selected_axis)]
    actual_offset = ee - reference
    hindsight_offset = actor_goal - reference

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    figure.suptitle(
        "TRON2 CHIP rollout diagnostics — {} axis, peak force {:.2f} N".format(
            selected_axis.upper(), float(np.max(force_norm))
        )
    )

    axes[0].plot(time_s, ee, label="actual end-effector", linewidth=1.8)
    axes[0].plot(time_s, reference, label="original goal", linewidth=1.4, linestyle="--")
    axes[0].plot(time_s, actor_goal, label="actor goal", linewidth=1.4, linestyle=":")
    axes[0].set_ylabel("Position {} [m]".format(selected_axis))
    axes[0].set_title("End-effector and tracking goals")
    axes[0].legend(loc="best", ncols=3)

    for item in AXES:
        axes[1].plot(time_s, data["force_{}_n".format(item)], label="F{}".format(item), linewidth=1.5)
    axes[1].set_ylabel("Force [N]")
    axes[1].set_title("Applied world-frame perturbation")
    axes[1].legend(loc="best", ncols=3)

    axes[2].plot(time_s, actual_offset * 1000.0, label="actual minus original", linewidth=1.8)
    axes[2].plot(time_s, hindsight_offset * 1000.0, label="actor goal minus original", linewidth=1.5, linestyle="--")
    axes[2].axhline(0.0, color="0.45", linewidth=0.8)
    axes[2].set_ylabel("Offset [mm]")
    axes[2].set_title("Compliance response on selected axis")
    axes[2].legend(loc="best")

    if "max_control_fraction" in data:
        axes[3].plot(time_s, 100.0 * data["max_control_fraction"], label="maximum limit usage", linewidth=1.5)
        axes[3].axhline(100.0, label="saturation threshold", linewidth=1.2, linestyle="--")
        axes[3].set_ylabel("Limit usage [%]")
    else:
        axes[3].plot(time_s, data["max_abs_ctrl"], label="max |control|", linewidth=1.5)
        if control_limit is not None:
            axes[3].axhline(float(control_limit), label="configured arm limit", linewidth=1.2, linestyle="--")
        axes[3].set_ylabel("Control [Nm/N]")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title("Actuator effort and saturation")
    axes[3].legend(loc="best")

    for subplot in axes:
        subplot.grid(True, alpha=0.25)
        for start, end in intervals:
            subplot.axvspan(start, end, alpha=0.08, color="tab:orange")

    csv_path = Path(csv_path)
    output = Path(output_path) if output_path else csv_path.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return {
        "output": output,
        "axis": selected_axis,
        "peak_force_n": float(np.max(force_norm)),
        "peak_actual_offset_m": float(np.max(np.abs(actual_offset))),
        "peak_control": float(np.max(data["max_abs_ctrl"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--control-limit", type=float, default=80.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    result = plot_rollout(
        args.csv, args.output, args.axis, args.control_limit, args.show
    )
    print(
        "plotted: axis={} peak_force={:.3f}N peak_offset={:.5f}m "
        "peak_control={:.3f} output={}".format(
            result["axis"], result["peak_force_n"],
            result["peak_actual_offset_m"], result["peak_control"],
            result["output"].resolve(),
        )
    )


if __name__ == "__main__":
    main()
