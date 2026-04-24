"""Physics operators, PDE residuals and boundary conditions."""

from .bcs import box_face_normals, dirichlet_loss, flux_conservation_loss, neumann_loss
from .operators import divergence, gradient, laplacian
from .pde import conductivity_pde_residual, gaussian_dipole_source, gaussian_smoothed_delta

__all__ = [
    "gradient",
    "divergence",
    "laplacian",
    "conductivity_pde_residual",
    "gaussian_smoothed_delta",
    "gaussian_dipole_source",
    "dirichlet_loss",
    "neumann_loss",
    "flux_conservation_loss",
    "box_face_normals",
]
