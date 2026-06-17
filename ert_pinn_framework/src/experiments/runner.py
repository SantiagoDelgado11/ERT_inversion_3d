"""Load project configuration and dispatch experiment pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..main import run_minimal_inverse
from ..utils.io import load_yaml, save_json


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _section(root: dict[str, Any], key: str) -> dict[str, Any]:
    return root.get(key, root) if isinstance(root.get(key, root), dict) else root


def _missing(value: object) -> bool:
    return value in (None, "", "null")


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_synthetic_observations(
    predictions_path: Path,
    observations_path: Path,
    *,
    count: int,
    seed: int,
) -> dict[str, object]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Training predictions not found: {predictions_path}")

    data = np.load(predictions_path)
    points = np.asarray(data["points"], dtype=np.float64)
    potential = np.asarray(data["potential"], dtype=np.float64).reshape(-1, 1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Training prediction points must have shape (N, 3); got {points.shape}")
    if points.shape[0] != potential.shape[0]:
        raise ValueError("Training prediction points and potential arrays must have the same row count")

    rng = np.random.default_rng(seed)
    if 0 < count < points.shape[0]:
        idx = rng.choice(points.shape[0], size=count, replace=False)
        points = points[idx]
        potential = potential[idx]

    observations_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        observations_path,
        np.hstack([points, potential]),
        delimiter=",",
        header="x,y,z,potential",
        comments="# ",
    )
    return {
        "path": str(observations_path),
        "count": int(points.shape[0]),
        "source": str(predictions_path),
    }


def load_project_config(config_dir: str | Path) -> dict:
    """Load base config and all referenced default sections."""
    cfg_dir = Path(config_dir).resolve()
    base_cfg_path = cfg_dir / "base.yaml"
    base_cfg = load_yaml(base_cfg_path)

    defaults = base_cfg.get("defaults", {})
    required_sections = ["data", "model", "training", "inverse", "electrodes", "experiment"]

    config = {"base": base_cfg}
    for section in required_sections:
        rel_path = defaults.get(section)
        if rel_path is None:
            raise KeyError(f"Missing defaults entry for section '{section}' in {base_cfg_path}")
        config[section] = load_yaml(cfg_dir / rel_path)

    config["_meta"] = {
        "config_dir": str(cfg_dir),
        "project_root": str(cfg_dir.parent),
    }

    return config


def run_experiment(
    config_dir: str | Path,
    override_mode: str | None = None,
    override_name: str | None = None,
    observation_overrides: dict[str, Any] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> dict:
    """Run selected experiment mode and return summary dictionary."""
    config = load_project_config(config_dir)

    if runtime_overrides:
        _deep_update(config, runtime_overrides)

    if observation_overrides:
        inverse_root = config["inverse"]
        if isinstance(inverse_root.get("inversion"), dict):
            observation_cfg = inverse_root["inversion"].setdefault("observations", {})
        else:
            observation_cfg = inverse_root.setdefault("observations", {})
        for key, value in observation_overrides.items():
            if value is not None:
                observation_cfg[key] = value

    base_cfg = config["base"]
    exp_cfg = config["experiment"].get("experiment", config["experiment"])

    mode = override_mode or str(exp_cfg.get("mode", "train_then_invert"))
    exp_name = override_name or str(exp_cfg.get("name", "exp_default"))

    output_root_rel = str(base_cfg.get("paths", {}).get("output_root", "outputs"))
    project_root = Path(config["_meta"]["project_root"]).resolve()
    output_root = project_root / output_root_rel / exp_name

    config["_meta"] = {
        "config_dir": str(Path(config_dir).resolve()),
        "project_root": str(project_root.resolve()),
    }

    normalized_mode = mode.strip().lower()
    if normalized_mode in {"train", "training"}:
        return run_minimal_inverse(config=config, output_root=output_root, mode="train")

    if normalized_mode in {"invert", "inversion"}:
        return run_minimal_inverse(config=config, output_root=output_root, mode="invert")

    if normalized_mode == "train_then_invert":
        training_summary = run_minimal_inverse(config=config, output_root=output_root, mode="train")
        inverse_cfg = _section(config["inverse"], "inversion")
        obs_cfg = inverse_cfg.setdefault("observations", {})
        synthetic_observations = None
        if _missing(obs_cfg.get("path")):
            sampling_cfg = config["data"].get("sampling", {})
            count = int(sampling_cfg.get("measurement_points", 512))
            seed = int(config["base"].get("project", {}).get("seed", 42))
            observations_path = output_root / "synthetic_observations.csv"
            synthetic_observations = _write_synthetic_observations(
                output_root / "training_predictions.npz",
                observations_path,
                count=count,
                seed=seed,
            )
            obs_cfg["path"] = _safe_rel_path(observations_path, project_root)
            obs_cfg["delimiter"] = ","
            obs_cfg["skiprows"] = 1
            obs_cfg["point_columns"] = [0, 1, 2]
            obs_cfg["value_column"] = 3

        inversion_summary = run_minimal_inverse(config=config, output_root=output_root, mode="invert")
        summary = {
            "mode": "train_then_invert",
            "training": training_summary,
            "synthetic_observations": synthetic_observations,
            "inversion": inversion_summary,
        }
        save_json(summary, output_root / "experiment_summary.json")
        return summary

    options = "invert, inversion, train, training, train_then_invert"
    raise KeyError(f"Unknown experiment mode '{mode}'. Available: {options}")
