"""relativistic_jax: Numerical verification suite for relativistic kinematics."""

from .constants import PhysicalConstants, NumericalConfig, PHYSICS, CONFIG
from .kinematics import (
    relativistic_lagrangian,
    canonical_momentum_ad,
    hamiltonian_energy_ad,
    four_momentum_ad,
    closed_form_lorentz_gamma,
    closed_form_four_momentum,
    minkowski_contraction,
)
from .quadrature import roots_and_weights_golub_welsch, integrate_work_quadrature
from .suite import NumericalVerificationSuite, TestRecord

__version__ = "1.0.0"
__all__ = [
    "PhysicalConstants",
    "NumericalConfig",
    "PHYSICS",
    "CONFIG",
    "relativistic_lagrangian",
    "canonical_momentum_ad",
    "hamiltonian_energy_ad",
    "four_momentum_ad",
    "closed_form_lorentz_gamma",
    "closed_form_four_momentum",
    "minkowski_contraction",
    "roots_and_weights_golub_welsch",
    "integrate_work_quadrature",
    "NumericalVerificationSuite",
    "TestRecord",
]
