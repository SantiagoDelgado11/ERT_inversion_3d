"""Composable pipelines for ERT PINN training and inversion."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..data.domain import FACE_NAMES, BoxDomain3D
from ..data.sampler import DomainSampler, SamplerConfig
from ..forward.solver import ForwardSolver
from ..inverse.parameterizations import build_conductivity_parameterization
from ..inverse.regularization import l2_parameter_regularization, total_variation_regularization
from ..models.pinn import PINNModel
from ..physics.bcs import neumann_loss
from ..physics.pde import conductivity_pde_residual
from ..training.losses import LossWeights, WeightedLossComposer
from ..training.optimizers import build_optimizer, build_scheduler
from ..training.trainer import Trainer, TrainerConfig
from ..utils.io import ensure_dir, save_json
from ..utils.logger import build_logger


def _resolve_device(base_config: dict) -> torch.device:
    requested = str(base_config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve_dtype(base_config: dict) -> torch.dtype:
    dtype_name = str(base_config.get("runtime", {}).get("dtype", "float32")).lower()
    if dtype_name == "float64":
        return torch.float64
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def _set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def _to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype, requires_grad: bool) -> Tensor:
    return torch.tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def _prepare_output_dirs(base_config: dict, output_root: Path) -> tuple[Path, Path, Path]:
    paths = base_config.get("paths", {})
    root = ensure_dir(output_root)
    logs_dir = ensure_dir(root / str(paths.get("logs_dir", "logs")))
    checkpoints_dir = ensure_dir(root / str(paths.get("checkpoints_dir", "checkpoints")))
    return root, logs_dir, checkpoints_dir


def _resolve_project_root(config: dict, output_root: Path) -> Path:
    meta = config.get("_meta", {})
    project_root = meta.get("project_root")
    if project_root:
        return Path(project_root)

    # Fallback when metadata is not provided: <project>/outputs/<exp_name>
    return Path(output_root).resolve().parent.parent


def _subsample_points(points: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if batch_size <= 0 or batch_size >= points.shape[0]:
        return points
    indices = rng.choice(points.shape[0], size=batch_size, replace=False)
    return points[indices]


def _face_normals(face: str, n_points: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    normals = torch.zeros((n_points, 3), device=device, dtype=dtype)
    if face == "x_min":
        normals[:, 0] = -1.0
    elif face == "x_max":
        normals[:, 0] = 1.0
    elif face == "y_min":
        normals[:, 1] = -1.0
    elif face == "y_max":
        normals[:, 1] = 1.0
    elif face == "z_min":
        normals[:, 2] = -1.0
    elif face == "z_max":
        normals[:, 2] = 1.0
    else:
        raise ValueError(f"Unknown boundary face: {face}")
    return normals


def _build_electrode_positions(electrodes_cfg: dict, device: torch.device, dtype: torch.dtype) -> Tensor:
    source = electrodes_cfg.get("electrodes", electrodes_cfg)

    count = int(source.get("count", 16))
    radius = float(source.get("radius", 1.0))
    z_coord = float(source.get("z", 0.0))
    arrangement = str(source.get("arrangement", "circular")).lower()

    if count < 2:
        raise ValueError("electrodes.count must be >= 2")
    if arrangement != "circular":
        raise ValueError(f"Unsupported electrode arrangement: {arrangement}")

    angles = torch.linspace(0.0, 2.0 * torch.pi, steps=count + 1, device=device, dtype=dtype)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = torch.full((count,), z_coord, device=device, dtype=dtype)
    return torch.stack([x, y, z], dim=1)


def _adjacent_pair_indices(count: int, epoch: int, skip: int) -> tuple[int, int]:
    source_idx = (epoch - 1) % count
    sink_idx = (source_idx + skip + 1) % count
    if sink_idx == source_idx:
        sink_idx = (source_idx + 1) % count
    return source_idx, sink_idx


def _load_observations(
    inversion_cfg: dict,
    project_root: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Path] | None:
    obs_cfg = inversion_cfg.get("observations", {})
    raw_path = obs_cfg.get("path")

    if raw_path in (None, "", "null"):
        return None

    obs_path = Path(str(raw_path))
    if not obs_path.is_absolute():
        obs_path = project_root / obs_path
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    if obs_path.suffix.lower() == ".npy":
        raw = np.load(obs_path)
    else:
        delimiter = str(obs_cfg.get("delimiter", ","))
        skiprows = int(obs_cfg.get("skiprows", 0))
        raw = np.loadtxt(obs_path, delimiter=delimiter, skiprows=skiprows)

    table = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    if table.shape[1] < 4:
        raise ValueError("Observation table must have at least 4 columns: x, y, z, value")

    point_columns = obs_cfg.get("point_columns", [0, 1, 2])
    if len(point_columns) != 3:
        raise ValueError("point_columns must contain exactly 3 indices")

    value_column = int(obs_cfg.get("value_column", 3))
    points = table[:, point_columns]
    values = table[:, value_column : value_column + 1]

    point_tensor = torch.tensor(points, device=device, dtype=dtype)
    value_tensor = torch.tensor(values, device=device, dtype=dtype)
    return point_tensor, value_tensor, obs_path


def _split_observations(
    points: Tensor,
    values: Tensor,
    train_split: float,
    val_split: float,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n_samples = points.shape[0]
    if n_samples < 2:
        return points, values, points, values

    train_ratio = float(train_split)
    val_ratio = float(val_split)

    if train_ratio <= 0.0 or train_ratio >= 1.0:
        train_ratio = 0.8
    if val_ratio < 0.0:
        val_ratio = 0.2
    if train_ratio + val_ratio > 1.0:
        val_ratio = max(0.0, 1.0 - train_ratio)

    perm = torch.as_tensor(rng.permutation(n_samples), device=points.device, dtype=torch.long)
    n_train = int(round(train_ratio * n_samples))
    n_train = max(1, min(n_samples - 1, n_train))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if val_idx.numel() == 0:
        val_idx = train_idx

    return points[train_idx], values[train_idx], points[val_idx], values[val_idx]


def run_training_pipeline(config: dict, output_root: Path, force_final_checkpoint: bool = False) -> dict:
    """Run forward PINN training and save final checkpoint."""
    base_cfg = config["base"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    electrodes_cfg = config["electrodes"]
    experiment_cfg = config["experiment"]

    device = _resolve_device(base_cfg)
    dtype = _resolve_dtype(base_cfg)

    project_cfg = base_cfg.get("project", {})
    runtime_cfg = base_cfg.get("runtime", {})
    _set_seed(
        seed=int(project_cfg.get("seed", 42)),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(runtime_cfg.get("cudnn_benchmark", False)),
    )

    out_root, logs_dir, checkpoints_dir = _prepare_output_dirs(base_cfg, output_root)
    logger = build_logger("ert.training", logs_dir, file_name="training.log")
    logger.info("Training started on device: %s", device)

    exp_section = experiment_cfg.get("experiment", experiment_cfg)
    save_predictions = bool(exp_section.get("save_predictions", True))
    save_checkpoints = bool(exp_section.get("save_checkpoints", True))

    data_section = data_cfg.get("dataset", {})
    if bool(data_section.get("normalize_inputs", False)):
        logger.warning("dataset.normalize_inputs is currently not applied in physics losses")
    if bool(data_section.get("normalize_targets", False)):
        logger.warning("dataset.normalize_targets is only used in inversion observation loss")

    domain = BoxDomain3D.from_config(data_cfg)
    sampler = DomainSampler(SamplerConfig.from_config(data_cfg))

    training_section = training_cfg.get("training", training_cfg)
    batch_size = int(training_section.get("batch_size", sampler.config.interior_points))
    checkpoint_every = int(training_section.get("checkpoint_every", 0))

    electrodes_section = electrodes_cfg.get("electrodes", electrodes_cfg)
    electrode_points = _build_electrode_positions(electrodes_section, device=device, dtype=dtype)
    electrode_count = electrode_points.shape[0]
    injection_cfg = electrodes_section.get("injection_patterns", {})
    injection_type = str(injection_cfg.get("type", "adjacent")).lower()
    if injection_type != "adjacent":
        raise ValueError(f"Unsupported injection pattern type: {injection_type}")
    injection_skip = int(injection_cfg.get("skip", 0))

    measurement_cfg = electrodes_section.get("measurement_patterns", {})
    measurement_type = str(measurement_cfg.get("type", "adjacent")).lower()
    if measurement_type != "adjacent":
        raise ValueError(f"Unsupported measurement pattern type: {measurement_type}")
    measurement_skip = int(measurement_cfg.get("skip", 1))

    contact_impedance = max(float(electrodes_section.get("contact_impedance", 0.0)), 0.0)
    drive_scale = 1.0 / (1.0 + contact_impedance)

    model = PINNModel.from_config(model_cfg).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model.parameters(), training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg)

    weights = LossWeights.from_config(training_cfg)
    composer = WeightedLossComposer(weights)
    trainer_cfg = TrainerConfig.from_config(training_cfg)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, logger=logger, config=trainer_cfg)

    solver = ForwardSolver(model=model, conductivity=1.0, source=0.0)

    def step_fn(epoch: int):
        batch = sampler.sample(domain)

        interior_np = _subsample_points(batch.interior, batch_size=batch_size, rng=sampler.rng)
        interior = _to_tensor(interior_np, device=device, dtype=dtype, requires_grad=True)

        pde = solver.pde_loss(interior)

        source_idx, sink_idx = _adjacent_pair_indices(electrode_count, epoch=epoch, skip=injection_skip)
        electrode_target = torch.zeros((electrode_count, 1), device=device, dtype=dtype)
        electrode_target[source_idx, 0] = drive_scale
        electrode_target[sink_idx, 0] = -drive_scale

        electrode_pred = model.potential(electrode_points)
        electrode_data = torch.mean((electrode_pred - electrode_target) ** 2)
        rolled_pred = torch.roll(electrode_pred, shifts=-(measurement_skip + 1), dims=0)
        measurement_balance = torch.mean(electrode_pred - rolled_pred) ** 2
        gauge = torch.mean(electrode_pred) ** 2 + measurement_balance

        neumann_terms: list[Tensor] = []
        for face_name in FACE_NAMES:
            face_points = _to_tensor(batch.boundaries[face_name], device=device, dtype=dtype, requires_grad=False)
            normals = _face_normals(face_name, face_points.shape[0], device=device, dtype=dtype)
            target_flux = torch.zeros((face_points.shape[0], 1), device=device, dtype=dtype)
            neumann_terms.append(
                neumann_loss(
                    model=model,
                    points=face_points,
                    normals=normals,
                    target_flux=target_flux,
                    conductivity=1.0,
                )
            )
        neumann = torch.stack(neumann_terms).mean()

        total, logs = composer.total(
            {
                "pde": pde,
                "neumann_bc": neumann,
                "data": electrode_data,
                "dirichlet_bc": gauge,
            }
        )
        logs["lr"] = float(optimizer.param_groups[0]["lr"])
        logs["source_electrode"] = float(source_idx)
        logs["sink_electrode"] = float(sink_idx)
        logs["measurement_balance"] = float(measurement_balance.detach().cpu().item())
        return total, logs

    def on_epoch_end(epoch: int, _metrics: dict[str, float]) -> None:
        if not save_checkpoints or checkpoint_every <= 0:
            return
        if epoch % checkpoint_every != 0:
            return

        epoch_checkpoint = checkpoints_dir / f"forward_epoch_{epoch:06d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_cfg,
                "device": str(device),
            },
            epoch_checkpoint,
        )

    history = trainer.train(step_fn, on_epoch_end=on_epoch_end)

    checkpoint_path: Path | None = None
    if save_checkpoints or force_final_checkpoint:
        checkpoint_path = checkpoints_dir / "forward_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_cfg,
                "device": str(device),
            },
            checkpoint_path,
        )

    if save_predictions:
        measurement_count = int(data_cfg.get("sampling", {}).get("measurement_points", 512))
        prediction_points_np = domain.sample_uniform(measurement_count, sampler.rng)
        prediction_points = _to_tensor(
            prediction_points_np,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        with torch.no_grad():
            predicted_potential = model.potential(prediction_points).detach().cpu().numpy()

        np.savez(
            out_root / "training_predictions.npz",
            points=prediction_points_np,
            potential=predicted_potential,
        )

    summary = {
        "mode": "train",
        "epochs": len(history),
        "final_metrics": history[-1] if history else {},
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    }
    save_json(summary, out_root / "training_summary.json")

    if checkpoint_path is None:
        logger.info("Training finished without saving final checkpoint")
    else:
        logger.info("Training finished. Checkpoint: %s", checkpoint_path)
    return summary


def run_inversion_pipeline(config: dict, output_root: Path) -> dict:
    """Run conductivity inversion using PDE loss, regularization and optional observations."""
    base_cfg = config["base"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    inverse_cfg = config["inverse"]
    experiment_cfg = config["experiment"]

    inv_section = inverse_cfg.get("inversion", inverse_cfg)
    if not bool(inv_section.get("enabled", True)):
        raise RuntimeError("Inversion is disabled by config.inverse.inversion.enabled=false")

    device = _resolve_device(base_cfg)
    dtype = _resolve_dtype(base_cfg)

    project_cfg = base_cfg.get("project", {})
    runtime_cfg = base_cfg.get("runtime", {})
    _set_seed(
        seed=int(project_cfg.get("seed", 42)),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(runtime_cfg.get("cudnn_benchmark", False)),
    )

    out_root, logs_dir, checkpoints_dir = _prepare_output_dirs(base_cfg, output_root)
    logger = build_logger("ert.inversion", logs_dir, file_name="inversion.log")
    logger.info("Inversion started on device: %s", device)

    exp_section = experiment_cfg.get("experiment", experiment_cfg)
    save_predictions = bool(exp_section.get("save_predictions", True))
    save_checkpoints = bool(exp_section.get("save_checkpoints", True))

    project_root = _resolve_project_root(config, output_root)

    domain = BoxDomain3D.from_config(data_cfg)
    sampler = DomainSampler(SamplerConfig.from_config(data_cfg))

    training_section = training_cfg.get("training", training_cfg)
    batch_size = int(training_section.get("batch_size", sampler.config.interior_points))
    checkpoint_every = int(inv_section.get("checkpoint_every", training_section.get("checkpoint_every", 0)))

    model = PINNModel.from_config(model_cfg).to(device=device, dtype=dtype)
    require_forward_checkpoint = bool(inv_section.get("require_forward_checkpoint", True))
    forward_checkpoint = checkpoints_dir / "forward_model.pt"
    if forward_checkpoint.exists():
        state = torch.load(forward_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded forward checkpoint: %s", forward_checkpoint)
    elif require_forward_checkpoint:
        raise FileNotFoundError(
            "Required forward checkpoint not found. Run training first or set "
            "inversion.require_forward_checkpoint=false"
        )
    else:
        logger.warning("Forward checkpoint not found. Inversion starts from current model weights")

    jointly_optimize_potential = bool(inv_section.get("jointly_optimize_potential", True))
    if jointly_optimize_potential:
        model.train()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    conductivity_model = build_conductivity_parameterization(inv_section).to(device=device, dtype=dtype)

    parameters = list(conductivity_model.parameters())
    if jointly_optimize_potential:
        parameters.extend(model.parameters())

    optimizer = build_optimizer(parameters, inv_section)
    scheduler = build_scheduler(optimizer, inv_section)

    reg_cfg = inv_section.get("regularization", {})
    reg_l2 = float(reg_cfg.get("l2_weight", 1e-4))
    reg_tv = float(reg_cfg.get("tv_weight", 0.0))

    loss_cfg = inv_section.get("loss_weights", {})
    weights = LossWeights(
        pde=float(loss_cfg.get("pde", 1.0)),
        data=float(loss_cfg.get("data", 0.0)),
        regularization=float(loss_cfg.get("regularization", 1.0)),
        dirichlet_bc=0.0,
        neumann_bc=0.0,
    )
    composer = WeightedLossComposer(weights)

    observation_pack = _load_observations(inv_section, project_root=project_root, device=device, dtype=dtype)
    obs_train_points: Tensor | None = None
    obs_train_values: Tensor | None = None
    obs_val_points: Tensor | None = None
    obs_val_values: Tensor | None = None
    obs_path: Path | None = None
    target_mean = torch.zeros((), device=device, dtype=dtype)
    target_std = torch.ones((), device=device, dtype=dtype)

    if observation_pack is not None:
        all_points, all_values, obs_path = observation_pack
        dataset_section = data_cfg.get("dataset", {})
        train_split = float(dataset_section.get("train_split", 0.8))
        val_split = float(dataset_section.get("val_split", 0.2))

        (
            obs_train_points,
            obs_train_values,
            obs_val_points,
            obs_val_values,
        ) = _split_observations(
            all_points,
            all_values,
            train_split=train_split,
            val_split=val_split,
            rng=sampler.rng,
        )

        if bool(dataset_section.get("normalize_targets", False)):
            target_mean = obs_train_values.mean()
            target_std = obs_train_values.std(unbiased=False).clamp_min(1e-8)

        logger.info(
            "Loaded observations from %s (train=%d, val=%d)",
            obs_path,
            obs_train_points.shape[0],
            obs_val_points.shape[0],
        )

    if weights.data > 0.0 and obs_train_points is None:
        raise ValueError(
            "Data-loss weight is positive, but no observation file was provided at "
            "inversion.observations.path"
        )
    if weights.data > 0.0 and not jointly_optimize_potential:
        raise ValueError(
            "Data-loss optimization requires inversion.jointly_optimize_potential=true"
        )

    epochs = int(inv_section.get("epochs", max(200, int(training_section.get("epochs", 1000)) // 2)))
    log_every = int(inv_section.get("log_every", training_section.get("log_every", 20)))

    training_target = torch.nn.ModuleList([model, conductivity_model]) if jointly_optimize_potential else conductivity_model

    trainer = Trainer(
        model=training_target,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        config=TrainerConfig(
            epochs=epochs,
            mixed_precision=False,
            grad_clip_norm=None,
            log_every=log_every,
        ),
    )

    def step_fn(epoch: int):
        batch = sampler.sample(domain)

        interior_np = _subsample_points(batch.interior, batch_size=batch_size, rng=sampler.rng)
        interior = _to_tensor(interior_np, device=device, dtype=dtype, requires_grad=True)

        pde_res = conductivity_pde_residual(
            model=model,
            points=interior,
            conductivity=conductivity_model,
            source=0.0,
        )
        pde_loss = torch.mean(pde_res**2)

        if obs_train_points is not None and obs_train_values is not None and weights.data > 0.0:
            pred_train = model.potential(obs_train_points)
            pred_train_scaled = (pred_train - target_mean) / target_std
            train_values_scaled = (obs_train_values - target_mean) / target_std
            data_loss = torch.mean((pred_train_scaled - train_values_scaled) ** 2)
        else:
            data_loss = torch.zeros((), device=device, dtype=dtype)

        sigma_interior = conductivity_model(interior)
        tv_loss = total_variation_regularization(sigma_interior, interior)
        l2_loss = l2_parameter_regularization(conductivity_model)
        regularization = reg_l2 * l2_loss + reg_tv * tv_loss

        total, logs = composer.total(
            {
                "pde": pde_loss,
                "data": data_loss,
                "regularization": regularization,
            }
        )

        if obs_val_points is not None and obs_val_values is not None and weights.data > 0.0:
            with torch.no_grad():
                pred_val = model.potential(obs_val_points)
                pred_val_scaled = (pred_val - target_mean) / target_std
                val_values_scaled = (obs_val_values - target_mean) / target_std
                val_data = torch.mean((pred_val_scaled - val_values_scaled) ** 2)
            logs["val_data"] = float(val_data.detach().cpu().item())

        logs["sigma_mean"] = float(sigma_interior.mean().detach().cpu().item())
        return total, logs

    def on_epoch_end(epoch: int, _metrics: dict[str, float]) -> None:
        if not save_checkpoints or checkpoint_every <= 0:
            return
        if epoch % checkpoint_every != 0:
            return

        epoch_checkpoint = checkpoints_dir / f"conductivity_epoch_{epoch:06d}.pt"
        payload = {
            "epoch": epoch,
            "conductivity_state_dict": conductivity_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "inverse_config": inverse_cfg,
            "device": str(device),
        }
        if jointly_optimize_potential:
            payload["model_state_dict"] = model.state_dict()
        torch.save(payload, epoch_checkpoint)

    history = trainer.train(step_fn, on_epoch_end=on_epoch_end)

    checkpoint_path: Path | None = None
    if save_checkpoints:
        checkpoint_path = checkpoints_dir / "conductivity_model.pt"
        payload = {
            "conductivity_state_dict": conductivity_model.state_dict(),
            "inverse_config": inverse_cfg,
            "device": str(device),
        }
        if jointly_optimize_potential:
            payload["model_state_dict"] = model.state_dict()
        torch.save(payload, checkpoint_path)

    if save_predictions:
        if obs_train_points is not None and obs_train_values is not None:
            with torch.no_grad():
                pred_train = model.potential(obs_train_points).detach().cpu().numpy()
                sigma_train = conductivity_model(obs_train_points).detach().cpu().numpy()

                pred_val = None
                if obs_val_points is not None:
                    pred_val = model.potential(obs_val_points).detach().cpu().numpy()

            np.savez(
                out_root / "inversion_predictions.npz",
                train_points=obs_train_points.detach().cpu().numpy(),
                train_potential_true=obs_train_values.detach().cpu().numpy(),
                train_potential_pred=pred_train,
                train_conductivity_pred=sigma_train,
                val_points=None if obs_val_points is None else obs_val_points.detach().cpu().numpy(),
                val_potential_true=None if obs_val_values is None else obs_val_values.detach().cpu().numpy(),
                val_potential_pred=pred_val,
            )
        else:
            measurement_count = int(data_cfg.get("sampling", {}).get("measurement_points", 512))
            pred_points_np = domain.sample_uniform(measurement_count, sampler.rng)
            pred_points = _to_tensor(pred_points_np, device=device, dtype=dtype, requires_grad=False)
            with torch.no_grad():
                sigma_pred = conductivity_model(pred_points).detach().cpu().numpy()

            np.savez(
                out_root / "inversion_predictions.npz",
                points=pred_points_np,
                conductivity=sigma_pred,
            )

    summary = {
        "mode": "invert",
        "epochs": len(history),
        "final_metrics": history[-1] if history else {},
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "observations": None if obs_path is None else str(obs_path),
        "jointly_optimized_potential": jointly_optimize_potential,
    }
    save_json(summary, out_root / "inversion_summary.json")

    if checkpoint_path is None:
        logger.info("Inversion finished without saving final checkpoint")
    else:
        logger.info("Inversion finished. Checkpoint: %s", checkpoint_path)
    return summary


def run_train_then_invert_pipeline(config: dict, output_root: Path) -> dict:
    """Run forward training followed by inversion."""
    inverse_section = config["inverse"].get("inversion", config["inverse"])
    inversion_enabled = bool(inverse_section.get("enabled", True))

    training_summary = run_training_pipeline(
        config=config,
        output_root=output_root,
        force_final_checkpoint=inversion_enabled,
    )

    if not inversion_enabled:
        summary = {
            "mode": "train_only",
            "training": training_summary,
            "inversion": {
                "skipped": True,
                "reason": "inversion.enabled=false",
            },
        }
        save_json(summary, Path(output_root) / "experiment_summary.json")
        return summary

    inversion_summary = run_inversion_pipeline(config=config, output_root=output_root)

    summary = {
        "mode": "train_then_invert",
        "training": training_summary,
        "inversion": inversion_summary,
    }
    save_json(summary, Path(output_root) / "experiment_summary.json")
    return summary
