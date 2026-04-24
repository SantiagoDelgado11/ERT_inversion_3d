"""Run a tiny synthetic train->invert experiment for quick result inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import load_project_config
from src.main import run_minimal_inverse
from src.utils.io import save_json


def main() -> None:
    config = load_project_config(PROJECT_ROOT / "configs")

    exp_name = "exp_synthetic_tiny"
    output_root = PROJECT_ROOT / "outputs" / exp_name

    model_cfg = config["model"].get("model", config["model"])
    model_cfg["hidden_dim"] = 32
    model_cfg["num_hidden_layers"] = 2

    sampling = config["data"].setdefault("sampling", {})
    sampling["interior_points_per_epoch"] = 256
    sampling["boundary_points_per_face_per_epoch"] = 64
    sampling["measurement_points"] = 128

    training = config["training"].setdefault("training", {})
    training["epochs"] = 5
    training["log_every"] = 1

    inversion = config["inverse"].setdefault("inversion", {})
    inversion["epochs"] = 5
    inversion["log_every"] = 1
    inversion_obs = inversion.setdefault("observations", {})
    inversion_obs["path"] = None

    training_summary = run_minimal_inverse(config=config, output_root=output_root, mode="train")

    train_pred = np.load(output_root / "training_predictions.npz")
    points = np.asarray(train_pred["points"], dtype=np.float64)
    potential = np.asarray(train_pred["potential"], dtype=np.float64).reshape(-1, 1)

    rng = np.random.default_rng(42)
    n_obs = min(96, points.shape[0])
    idx = rng.choice(points.shape[0], size=n_obs, replace=False)
    obs_table = np.hstack([points[idx], potential[idx]])

    obs_path = output_root / "synthetic_observations.csv"
    np.savetxt(
        obs_path,
        obs_table,
        delimiter=",",
        header="x,y,z,potential",
        comments="# ",
    )

    inversion_obs["path"] = str(obs_path.relative_to(PROJECT_ROOT))
    inversion_obs["delimiter"] = ","
    inversion_obs["skiprows"] = 1
    inversion_obs["point_columns"] = [0, 1, 2]
    inversion_obs["value_column"] = 3

    inversion_summary = run_minimal_inverse(config=config, output_root=output_root, mode="invert")

    result = {
        "mode": "tiny_synthetic_train_then_invert",
        "output_root": str(output_root),
        "training": training_summary,
        "inversion": inversion_summary,
        "observations": {
            "path": str(obs_path),
            "count": int(n_obs),
        },
    }
    save_json(result, output_root / "tiny_run_summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
