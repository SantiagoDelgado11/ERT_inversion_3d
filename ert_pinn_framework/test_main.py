# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
import json
from src.main import run_minimal_inverse

config = {
    "base": {
        "project": {"device": "cpu", "seed": 42},
        "runtime": {"dtype": "float32"}
    },
    "data": {
        "domain": {
            "bounds": {
                "x": [-1.0, 1.0],
                "y": [-1.0, 1.0],
                "z": [-1.0, 1.0]
            }
        },
        "sampling": {
            "interior_points_per_epoch": 100,
            "boundary_points_per_face_per_epoch": 100,
            "measurement_points": 10,
            "flux_source_points": 50,
            "flux_sink_points": 50
        }
    },
    "model": {
        "potential": {
            "input_dim": 22,
            "hidden_dim": 16,
            "num_hidden_layers": 2,
            "activation": "tanh"
        }
    },
    "inverse": {
        "conductivity": {
            "input_dim": 3,
            "hidden_dim": 16,
            "num_hidden_layers": 2,
            "activation": "tanh",
            "sigma_floor": 1e-6
        },
        "optimizer": {
            "lr": 1e-3
        },
        "epochs": 2,
        "log_every": 1,
        "loss_weights": {
            "data": 1.0,
            "pde": 1.0,
            "bc": 1.0,
            "reg": 1.0,
            "flux": 1.0
        },
        "source_model": {
            "current": 1.0,
            "gaussian_epsilon": 0.05,
            "flux_control_radius": 0.05,
            "fixed_electrode_pair": [0, 1]
        },
        "boundary_conditions": {
            "dirichlet_face": "z_max",
            "dirichlet_value": 0.0,
            "neumann_target_flux": 0.0
        },
        "regularization": {
            "tv_eps": 1e-8
        }
    },
    "training": {
        "optimizer": {
            "lr": 1e-3
        },
        "epochs": 2,
        "log_every": 1,
        "loss_weights": {
            "pde": 1.0,
            "bc": 1.0,
            "flux": 1.0
        }
    },
    "electrodes": {
        "count": 4,
        "radius": 0.5,
        "z": 0.0
    }
}

try:
    print("Running minimal inverse...")
    run_minimal_inverse(config, Path("test_out"), mode="train")
    print("Train mode successful!")
    run_minimal_inverse(config, Path("test_out"), mode="invert")
    print("Invert mode successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
