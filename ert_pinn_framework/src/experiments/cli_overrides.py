"""Command-line override helpers for ERT PINN experiments."""

from __future__ import annotations

import argparse
from typing import Any


ACTIVATION_CHOICES = ["tanh", "relu", "silu", "gelu"]


def _set_nested(root: dict[str, Any], keys: list[str], value: Any) -> None:
    if value is None:
        return
    current = root
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def add_pinn_runtime_args(parser: argparse.ArgumentParser, *, include_warm_start: bool = False) -> None:
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs")
    parser.add_argument("--log-every", type=int, default=None, help="Override console logging interval")
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="Override the Adam learning rate",
    )
    parser.add_argument("--weight-decay", type=float, default=None, help="Override Adam weight decay")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Alias for interior collocation points sampled per epoch",
    )
    parser.add_argument(
        "--boundary-points-per-face",
        type=int,
        default=None,
        help="Override boundary points sampled per face and epoch",
    )
    parser.add_argument(
        "--neumann-points-per-face",
        type=int,
        default=None,
        help="Override Neumann points sampled per face and epoch",
    )
    parser.add_argument("--flux-source-points", type=int, default=None)
    parser.add_argument("--flux-sink-points", type=int, default=None)
    parser.add_argument("--measurement-points", type=int, default=None)
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=None,
        help="Override hidden layers for both potential and conductivity networks",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override hidden width for both potential and conductivity networks",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=ACTIVATION_CHOICES,
        default=None,
        help="Override activation for both potential and conductivity networks",
    )
    parser.add_argument("--potential-hidden-layers", type=int, default=None)
    parser.add_argument("--potential-hidden-dim", type=int, default=None)
    parser.add_argument("--potential-activation", type=str, choices=ACTIVATION_CHOICES, default=None)
    parser.add_argument("--conductivity-hidden-layers", type=int, default=None)
    parser.add_argument("--conductivity-hidden-dim", type=int, default=None)
    parser.add_argument("--conductivity-activation", type=str, choices=ACTIVATION_CHOICES, default=None)
    parser.add_argument("--sigma-floor", type=float, default=None, help="Override conductivity positivity floor")
    parser.add_argument("--loss-data-weight", type=float, default=None)
    parser.add_argument("--loss-pde-weight", type=float, default=None)
    parser.add_argument("--loss-bc-weight", type=float, default=None)
    parser.add_argument("--loss-reg-weight", type=float, default=None)
    parser.add_argument("--loss-flux-weight", type=float, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override the output root directory configured in base.yaml",
    )
    if include_warm_start:
        parser.add_argument(
            "--warm-start-checkpoint",
            type=str,
            default=None,
            help="Checkpoint path used to initialize inversion networks",
        )


def build_pinn_runtime_overrides(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    _set_nested(overrides, ["base", "paths", "output_root"], args.output_root)
    _set_nested(overrides, ["data", "sampling", "interior_points_per_epoch"], args.batch_size)
    _set_nested(overrides, ["data", "sampling", "boundary_points_per_face_per_epoch"], args.boundary_points_per_face)
    _set_nested(overrides, ["data", "sampling", "neumann_points_per_face_per_epoch"], args.neumann_points_per_face)
    _set_nested(overrides, ["data", "sampling", "flux_source_points"], args.flux_source_points)
    _set_nested(overrides, ["data", "sampling", "flux_sink_points"], args.flux_sink_points)
    _set_nested(overrides, ["data", "sampling", "measurement_points"], args.measurement_points)

    if mode == "train":
        run_cfg_path = ["training", "training"]
        optimizer_path = ["training", "optimizer"]
        loss_weights_path = ["training", "training", "loss_weights"]
    elif mode == "invert":
        run_cfg_path = ["inverse", "inversion"]
        optimizer_path = ["inverse", "inversion", "optimizer"]
        loss_weights_path = ["inverse", "inversion", "loss_weights"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    _set_nested(overrides, run_cfg_path + ["epochs"], args.epochs)
    _set_nested(overrides, run_cfg_path + ["log_every"], args.log_every)
    _set_nested(overrides, optimizer_path + ["lr"], args.learning_rate)
    _set_nested(overrides, optimizer_path + ["weight_decay"], args.weight_decay)

    potential_path = ["model", "model", "potential"]
    conductivity_path = ["inverse", "inversion", "conductivity"]

    _set_nested(overrides, potential_path + ["num_hidden_layers"], args.hidden_layers)
    _set_nested(overrides, conductivity_path + ["num_hidden_layers"], args.hidden_layers)
    _set_nested(overrides, potential_path + ["hidden_dim"], args.hidden_dim)
    _set_nested(overrides, conductivity_path + ["hidden_dim"], args.hidden_dim)
    _set_nested(overrides, potential_path + ["activation"], args.activation)
    _set_nested(overrides, conductivity_path + ["activation"], args.activation)

    _set_nested(overrides, potential_path + ["num_hidden_layers"], args.potential_hidden_layers)
    _set_nested(overrides, potential_path + ["hidden_dim"], args.potential_hidden_dim)
    _set_nested(overrides, potential_path + ["activation"], args.potential_activation)
    _set_nested(overrides, conductivity_path + ["num_hidden_layers"], args.conductivity_hidden_layers)
    _set_nested(overrides, conductivity_path + ["hidden_dim"], args.conductivity_hidden_dim)
    _set_nested(overrides, conductivity_path + ["activation"], args.conductivity_activation)
    _set_nested(overrides, conductivity_path + ["sigma_floor"], args.sigma_floor)

    _set_nested(overrides, loss_weights_path + ["data"], args.loss_data_weight)
    _set_nested(overrides, loss_weights_path + ["pde"], args.loss_pde_weight)
    _set_nested(overrides, loss_weights_path + ["bc"], args.loss_bc_weight)
    _set_nested(overrides, loss_weights_path + ["reg"], args.loss_reg_weight)
    _set_nested(overrides, loss_weights_path + ["flux"], args.loss_flux_weight)

    warm_start_checkpoint = getattr(args, "warm_start_checkpoint", None)
    _set_nested(overrides, ["inverse", "inversion", "warm_start_checkpoint"], warm_start_checkpoint)

    return overrides
