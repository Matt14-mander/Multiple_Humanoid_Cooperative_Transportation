import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tron2_chip.plot_csv import dominant_force_axis, load_rollout_csv, plot_rollout


FIELDS = (
    "time_s", "ee_x", "ee_y", "ee_z", "ref_x", "ref_y", "ref_z",
    "actor_goal_x", "actor_goal_y", "actor_goal_z", "force_x_n",
    "force_y_n", "force_z_n", "max_abs_ctrl",
)


def _write_example(path: Path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(5):
            active = 10.0 if 1 <= index <= 3 else 0.0
            writer.writerow({
                "time_s": index * 0.1,
                "ee_x": 0.5 + index * 0.001,
                "ee_y": 0.0,
                "ee_z": 1.0,
                "ref_x": 0.5,
                "ref_y": 0.0,
                "ref_z": 1.0,
                "actor_goal_x": 0.5 - 0.002 * active,
                "actor_goal_y": 0.0,
                "actor_goal_z": 1.0,
                "force_x_n": active,
                "force_y_n": 0.0,
                "force_z_n": 0.0,
                "max_abs_ctrl": 20.0 + index,
            })


def test_plot_csv_selects_force_axis_and_creates_png(tmp_path: Path):
    source = tmp_path / "rollout.csv"
    output = tmp_path / "rollout.png"
    _write_example(source)
    data = load_rollout_csv(source)
    assert dominant_force_axis(data) == "x"
    result = plot_rollout(source, output, show=False)
    assert result["axis"] == "x"
    assert result["peak_force_n"] == 10.0
    assert output.exists()
    assert output.stat().st_size > 1000

