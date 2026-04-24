"""Minimal inverse PINN with modified PDE and flux conservation.

This file intentionally contains the full essential implementation:
- model definitions (u_theta, sigma_phi)
- autograd operators (grad, div)
- Gaussian dipole source
- full loss block (data + PDE + BC + TV + flux)
- training loop
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


FACE_SPECS: dict[str, tuple[int, float]] = {
    "x_min": (0, -1.0),
    "x_max": (0, 1.0),
    "y_min": (1, -1.0),
    "y_max": (1, 1.0),
    "z_min": (2, -1.0),
    "z_max": (2, 1.0),
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ):
        super().__init__()
        act_name = activation.lower()
        if act_name == "tanh":
            act = nn.Tanh
        elif act_name == "relu":
            act = nn.ReLU
        elif act_name == "silu":
            act = nn.SiLU
        elif act_name == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(max(0, hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PotentialNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, activation: str):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ConductivityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
        sigma_floor: float = 1e-6,
    ):
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(self.model(x)) + self.sigma_floor


def gradient(scalar: Tensor, inputs: Tensor) -> Tensor:
    return torch.autograd.grad(
        outputs=scalar,
        inputs=inputs,
        grad_outputs=torch.ones_like(scalar),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def divergence(vector: Tensor, inputs: Tensor) -> Tensor:
    terms = []
    for axis in range(vector.shape[1]):
        comp = vector[:, axis : axis + 1]
        grad_comp = torch.autograd.grad(
            outputs=comp,
            inputs=inputs,
            grad_outputs=torch.ones_like(comp),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        terms.append(grad_comp[:, axis : axis + 1])
    return torch.stack(terms, dim=0).sum(dim=0)


def gaussian_delta(points: Tensor, center: Tensor, epsilon: float) -> Tensor:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    diff = points - center.view(1, -1)
    sq_norm = torch.sum(diff * diff, dim=1, keepdim=True)
    dim = points.shape[1]
    norm_const = (math.sqrt(2.0 * math.pi) * float(epsilon)) ** dim
    return torch.exp(-0.5 * sq_norm / (float(epsilon) ** 2)) / norm_const


def gaussian_source(points: Tensor, source_center: Tensor, sink_center: Tensor, current: float, epsilon: float) -> Tensor:
    current_tensor = torch.as_tensor(float(current), device=points.device, dtype=points.dtype)
    return current_tensor * (
        gaussian_delta(points, source_center, epsilon) - gaussian_delta(points, sink_center, epsilon)
    )


def sample_uniform(
    bounds: dict[str, tuple[float, float]],
    n_points: int,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    mins = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64)
    maxs = np.array([bounds["x"][1], bounds["y"][1], bounds["z"][1]], dtype=np.float64)
    points = mins + rng.random((n_points, 3), dtype=np.float64) * (maxs - mins)
    return torch.tensor(points, device=device, dtype=dtype)


def sample_face(
    bounds: dict[str, tuple[float, float]],
    face: str,
    n_points: int,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if face not in FACE_SPECS:
        raise ValueError(f"Unknown face: {face}")

    points = sample_uniform(bounds, n_points, rng, device, dtype)
    axis, sign = FACE_SPECS[face]

    fixed_value = bounds["xyz"[axis]][0] if sign < 0 else bounds["xyz"[axis]][1]
    points[:, axis] = torch.as_tensor(fixed_value, device=device, dtype=dtype)

    normals = torch.zeros((n_points, 3), device=device, dtype=dtype)
    normals[:, axis] = sign
    return points, normals


def sample_sphere(
    center: Tensor,
    radius: float,
    n_points: int,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    directions = rng.normal(size=(n_points, 3)).astype(np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.clip(norms, 1e-12, None)
    points = center.detach().cpu().numpy().reshape(1, 3) + float(radius) * directions
    return (
        torch.tensor(points, device=device, dtype=dtype),
        torch.tensor(directions, device=device, dtype=dtype),
    )


def build_electrodes(count: int, radius: float, z_value: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    angles = torch.linspace(0.0, 2.0 * torch.pi, steps=count + 1, device=device, dtype=dtype)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = torch.full((count,), z_value, device=device, dtype=dtype)
    return torch.stack([x, y, z], dim=1)


def clamp_center(center: Tensor, bounds: dict[str, tuple[float, float]], margin: float) -> Tensor:
    mins = torch.tensor([bounds["x"][0], bounds["y"][0], bounds["z"][0]], device=center.device, dtype=center.dtype)
    maxs = torch.tensor([bounds["x"][1], bounds["y"][1], bounds["z"][1]], device=center.device, dtype=center.dtype)

    if margin < 0.0:
        raise ValueError("margin must be >= 0")

    lo = mins + margin
    hi = maxs - margin
    if torch.any(lo > hi):
        raise ValueError("Source smoothing radius / control radius is too large for domain bounds")

    return torch.minimum(torch.maximum(center, lo), hi)


def load_observations(
    path: Path | None,
    delimiter: str,
    skiprows: int,
    point_columns: list[int],
    value_column: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor | None, Tensor | None]:
    if path is None:
        return None, None

    if len(point_columns) != 3:
        raise ValueError("point_columns must contain exactly 3 indices")

    table = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, comments="#")
    table = np.atleast_2d(np.asarray(table, dtype=np.float64))
    if table.shape[1] < 4:
        raise ValueError("Observation file must have at least 4 columns: x,y,z,u")

    points = torch.tensor(table[:, point_columns], device=device, dtype=dtype)
    values = torch.tensor(table[:, value_column : value_column + 1], device=device, dtype=dtype)
    return points, values


def _scalar(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_losses(
    u_theta: PotentialNet,
    sigma_phi: ConductivityNet,
    bounds: dict[str, tuple[float, float]],
    n_interior: int,
    n_boundary_face: int,
    dirichlet_face: str,
    dirichlet_value: float,
    neumann_faces: list[str],
    neumann_target_flux: float,
    source_center: Tensor,
    sink_center: Tensor,
    current: float,
    gaussian_epsilon: float,
    flux_radius: float,
    flux_surface_points: int,
    tv_eps: float,
    obs_points: Tensor | None,
    obs_values: Tensor | None,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, Tensor]]:
    x_int = sample_uniform(bounds, n_interior, rng, device, dtype)
    x_dir, _ = sample_face(bounds, dirichlet_face, n_boundary_face, rng, device, dtype)

    if neumann_faces:
        x_neu_list: list[Tensor] = []
        n_neu_list: list[Tensor] = []
        for face in neumann_faces:
            face_points, face_normals = sample_face(bounds, face, n_boundary_face, rng, device, dtype)
            x_neu_list.append(face_points)
            n_neu_list.append(face_normals)
        x_neu = torch.cat(x_neu_list, dim=0)
        n_neu = torch.cat(n_neu_list, dim=0)
    else:
        x_neu = torch.zeros((0, 3), device=device, dtype=dtype)
        n_neu = torch.zeros((0, 3), device=device, dtype=dtype)

    x_src, n_src = sample_sphere(source_center, flux_radius, flux_surface_points, rng, device, dtype)
    x_sink, n_sink = sample_sphere(sink_center, flux_radius, flux_surface_points, rng, device, dtype)

    n_data = 0 if obs_points is None else int(obs_points.shape[0])
    all_blocks = [x_int, x_dir, x_neu, x_src, x_sink]
    if n_data > 0:
        all_blocks.append(obs_points)

    x_all = torch.cat(all_blocks, dim=0).detach().requires_grad_(True)

    u_all = u_theta(x_all)
    sigma_all = sigma_phi(x_all)
    grad_u_all = gradient(u_all, x_all)
    grad_sigma_all = gradient(sigma_all, x_all)
    laplace_u_all = divergence(grad_u_all, x_all)
    # Explicit product rule for div(sigma * grad(u)) to ensure operator-level correctness.
    div_sigma_grad_u_all = torch.sum(grad_sigma_all * grad_u_all, dim=1, keepdim=True) + sigma_all * laplace_u_all

    cursor = 0
    s_int = slice(cursor, cursor + x_int.shape[0])
    cursor += x_int.shape[0]
    s_dir = slice(cursor, cursor + x_dir.shape[0])
    cursor += x_dir.shape[0]
    s_neu = slice(cursor, cursor + x_neu.shape[0])
    cursor += x_neu.shape[0]
    s_src = slice(cursor, cursor + x_src.shape[0])
    cursor += x_src.shape[0]
    s_sink = slice(cursor, cursor + x_sink.shape[0])
    cursor += x_sink.shape[0]
    s_data = None if n_data == 0 else slice(cursor, cursor + n_data)

    source_term = gaussian_source(
        points=x_all[s_int],
        source_center=source_center,
        sink_center=sink_center,
        current=current,
        epsilon=gaussian_epsilon,
    )
    pde_residual = -div_sigma_grad_u_all[s_int] - source_term
    loss_pde = torch.mean(pde_residual**2)

    dirichlet_target = torch.as_tensor(dirichlet_value, device=device, dtype=dtype)
    loss_dirichlet = torch.mean((u_all[s_dir] - dirichlet_target) ** 2)

    if x_neu.shape[0] > 0:
        target_flux = torch.as_tensor(neumann_target_flux, device=device, dtype=dtype)
        normal_flux = torch.sum(sigma_all[s_neu] * grad_u_all[s_neu] * n_neu, dim=1, keepdim=True)
        loss_neumann = torch.mean((normal_flux - target_flux) ** 2)
    else:
        loss_neumann = torch.zeros((), device=device, dtype=dtype)

    loss_bc = loss_dirichlet + loss_neumann

    grad_sigma_int = grad_sigma_all[s_int]
    loss_reg = torch.mean(torch.sqrt(torch.sum(grad_sigma_int * grad_sigma_int, dim=1) + tv_eps))

    src_flux = torch.sum(sigma_all[s_src] * grad_u_all[s_src] * n_src, dim=1, keepdim=True)
    sink_flux = torch.sum(sigma_all[s_sink] * grad_u_all[s_sink] * n_sink, dim=1, keepdim=True)

    control_area = 4.0 * math.pi * (float(flux_radius) ** 2)
    flux_target = torch.as_tensor(float(current) / control_area, device=device, dtype=dtype)
    loss_flux = (torch.mean(src_flux) - flux_target) ** 2 + (torch.mean(sink_flux) + flux_target) ** 2

    if s_data is None or obs_values is None:
        loss_data = torch.zeros((), device=device, dtype=dtype)
    else:
        loss_data = torch.mean((u_all[s_data] - obs_values) ** 2)

    loss_total = loss_data + loss_pde + loss_bc + loss_reg + loss_flux

    return loss_total, {
        "data": loss_data,
        "pde": loss_pde,
        "bc": loss_bc,
        "reg": loss_reg,
        "flux": loss_flux,
        "total": loss_total,
        "bc_dirichlet": loss_dirichlet,
        "bc_neumann": loss_neumann,
        "sigma_mean": torch.mean(sigma_all[s_int]),
    }


def run_minimal_inverse(config: dict, output_root: Path, mode: str = "invert") -> dict:
    base_cfg = config["base"]
    data_cfg = config["data"]
    model_cfg = config["model"].get("model", config["model"])
    inverse_cfg = config["inverse"].get("inversion", config["inverse"])
    training_cfg = config["training"].get("training", config["training"])
    electrodes_cfg = config["electrodes"].get("electrodes", config["electrodes"])

    project_cfg = base_cfg.get("project", {})
    runtime_cfg = base_cfg.get("runtime", {})

    requested_device = str(project_cfg.get("device", "auto")).lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    dtype_name = str(runtime_cfg.get("dtype", "float32")).lower()
    dtype = torch.float64 if dtype_name == "float64" else torch.float32

    seed = int(project_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    bounds = {
        "x": tuple(data_cfg["domain"]["bounds"]["x"]),
        "y": tuple(data_cfg["domain"]["bounds"]["y"]),
        "z": tuple(data_cfg["domain"]["bounds"]["z"]),
    }

    u_theta = PotentialNet(
        input_dim=int(model_cfg.get("input_dim", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        hidden_layers=int(model_cfg.get("num_hidden_layers", 6)),
        activation=str(model_cfg.get("activation", "tanh")),
    ).to(device=device, dtype=dtype)

    sigma_phi = ConductivityNet(
        input_dim=3,
        hidden_dim=int(inverse_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 128))),
        hidden_layers=int(inverse_cfg.get("num_hidden_layers", model_cfg.get("num_hidden_layers", 3))),
        activation=str(inverse_cfg.get("activation", model_cfg.get("activation", "tanh"))),
        sigma_floor=float(inverse_cfg.get("sigma_floor", 1e-6)),
    ).to(device=device, dtype=dtype)

    optimizer_cfg = inverse_cfg.get("optimizer", config["training"].get("optimizer", {}))
    optimizer = torch.optim.Adam(
        list(u_theta.parameters()) + list(sigma_phi.parameters()),
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    sampling_cfg = data_cfg.get("sampling", {})
    n_interior = int(sampling_cfg.get("interior_points_per_epoch", 20000))
    n_boundary_face = int(sampling_cfg.get("boundary_points_per_face_per_epoch", 2000))
    n_prediction = int(sampling_cfg.get("measurement_points", 512))

    epochs = int(inverse_cfg.get("epochs", training_cfg.get("epochs", 1000)))
    log_every = int(inverse_cfg.get("log_every", training_cfg.get("log_every", 20)))

    source_cfg = inverse_cfg.get("source_model", {})
    current = float(source_cfg.get("current", 1.0))
    gaussian_epsilon = float(source_cfg.get("gaussian_epsilon", 0.05))
    flux_radius = float(source_cfg.get("flux_control_radius", gaussian_epsilon))
    flux_surface_points = int(source_cfg.get("flux_surface_points", 256))

    electrode_count = int(electrodes_cfg.get("count", 16))
    electrode_radius = float(electrodes_cfg.get("radius", 1.0))
    electrode_z = float(electrodes_cfg.get("z", 0.0))
    electrodes = build_electrodes(electrode_count, electrode_radius, electrode_z, device, dtype)

    fixed_pair = source_cfg.get("fixed_electrode_pair", [0, 1])
    if not isinstance(fixed_pair, (list, tuple)) or len(fixed_pair) != 2:
        raise ValueError("source_model.fixed_electrode_pair must be [source_idx, sink_idx]")

    source_idx, sink_idx = int(fixed_pair[0]), int(fixed_pair[1])
    if source_idx == sink_idx:
        raise ValueError("source and sink electrodes must be different")

    center_margin = max(gaussian_epsilon, flux_radius)
    source_center = clamp_center(electrodes[source_idx], bounds, center_margin)
    sink_center = clamp_center(electrodes[sink_idx], bounds, center_margin)

    bc_cfg = inverse_cfg.get("boundary_conditions", {})
    dirichlet_face = str(bc_cfg.get("dirichlet_face", "z_max"))
    dirichlet_value = float(bc_cfg.get("dirichlet_value", 0.0))
    raw_neumann_faces = bc_cfg.get("neumann_faces")
    if raw_neumann_faces is None:
        neumann_faces = [face for face in FACE_SPECS if face != dirichlet_face]
    else:
        neumann_faces = [str(face) for face in raw_neumann_faces]
        neumann_faces = [face for face in neumann_faces if face != dirichlet_face]

    for face in neumann_faces:
        if face not in FACE_SPECS:
            raise ValueError(f"Unknown Neumann face: {face}")

    neumann_target_flux = float(bc_cfg.get("neumann_target_flux", 0.0))

    tv_eps = float(inverse_cfg.get("regularization", {}).get("tv_eps", 1e-8))

    project_root = Path(config.get("_meta", {}).get("project_root", Path(output_root).resolve().parent.parent))
    obs_cfg = inverse_cfg.get("observations", {})
    obs_raw = obs_cfg.get("path")
    obs_path = None if obs_raw in (None, "", "null") else Path(str(obs_raw))
    if obs_path is not None and not obs_path.is_absolute():
        obs_path = project_root / obs_path

    obs_delimiter = str(obs_cfg.get("delimiter", ","))
    obs_skiprows = int(obs_cfg.get("skiprows", 0))
    obs_point_columns = [int(i) for i in obs_cfg.get("point_columns", [0, 1, 2])]
    obs_value_column = int(obs_cfg.get("value_column", 3))

    obs_points, obs_values = load_observations(
        path=obs_path,
        delimiter=obs_delimiter,
        skiprows=obs_skiprows,
        point_columns=obs_point_columns,
        value_column=obs_value_column,
        device=device,
        dtype=dtype,
    )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        total_loss, losses = compute_losses(
            u_theta=u_theta,
            sigma_phi=sigma_phi,
            bounds=bounds,
            n_interior=n_interior,
            n_boundary_face=n_boundary_face,
            dirichlet_face=dirichlet_face,
            dirichlet_value=dirichlet_value,
            neumann_faces=neumann_faces,
            neumann_target_flux=neumann_target_flux,
            source_center=source_center,
            sink_center=sink_center,
            current=current,
            gaussian_epsilon=gaussian_epsilon,
            flux_radius=flux_radius,
            flux_surface_points=flux_surface_points,
            tv_eps=tv_eps,
            obs_points=obs_points,
            obs_values=obs_values,
            rng=rng,
            device=device,
            dtype=dtype,
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")

        total_loss.backward()
        optimizer.step()

        row = {"epoch": float(epoch)}
        for key, value in losses.items():
            row[key] = _scalar(value)
        history.append(row)

        if epoch == 1 or epoch % max(1, log_every) == 0:
            print(
                "epoch={:5d} total={:.6e} pde={:.6e} data={:.6e} bc={:.6e} reg={:.6e} flux={:.6e}".format(
                    epoch,
                    row["total"],
                    row["pde"],
                    row["data"],
                    row["bc"],
                    row["reg"],
                    row["flux"],
                )
            )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / "minimal_inverse_model.pt"
    torch.save(
        {
            "u_theta": u_theta.state_dict(),
            "sigma_phi": sigma_phi.state_dict(),
            "device": str(device),
            "dtype": str(dtype),
        },
        checkpoint_path,
    )

    pred_points = sample_uniform(bounds, n_prediction, rng, device, dtype)
    with torch.no_grad():
        pred_u = u_theta(pred_points)
        pred_sigma = sigma_phi(pred_points)

    prediction_payload = {
        "points": pred_points.detach().cpu().numpy(),
        "potential": pred_u.detach().cpu().numpy(),
        "conductivity": pred_sigma.detach().cpu().numpy(),
    }

    np.savez(output_root / "inversion_predictions.npz", **prediction_payload)
    if mode == "train":
        np.savez(output_root / "training_predictions.npz", points=prediction_payload["points"], potential=prediction_payload["potential"])

    summary = {
        "mode": mode,
        "epochs": len(history),
        "final_metrics": history[-1] if history else {},
        "checkpoint": str(checkpoint_path),
        "observations": None if obs_path is None else str(obs_path),
        "source_electrode": source_idx,
        "sink_electrode": sink_idx,
    }

    summary_name = "training_summary.json" if mode == "train" else "inversion_summary.json"
    _save_json(summary, output_root / summary_name)
    return summary
