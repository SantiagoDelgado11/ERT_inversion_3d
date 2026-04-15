"""Load project configuration and dispatch experiment pipelines."""

from __future__ import annotations

from pathlib import Path

from ..utils.io import load_yaml
from .registry import get_experiment


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
) -> dict:
    """Run selected experiment mode and return summary dictionary."""
    config = load_project_config(config_dir)

    base_cfg = config["base"]
    exp_cfg = config["experiment"].get("experiment", config["experiment"])

    mode = override_mode or str(exp_cfg.get("mode", "train_then_invert"))
    exp_name = override_name or str(exp_cfg.get("name", "exp_default"))

    output_root_rel = str(base_cfg.get("paths", {}).get("output_root", "outputs"))
    project_root = Path(config_dir).parent
    output_root = project_root / output_root_rel / exp_name

    config["_meta"] = {
        "config_dir": str(Path(config_dir).resolve()),
        "project_root": str(project_root.resolve()),
    }

    pipeline = get_experiment(mode)
    return pipeline(config=config, output_root=output_root)
