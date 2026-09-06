"""Physical constants (CODATA 2022 / SI exact) and numerical acceptance budgets."""

from dataclasses import dataclass, field
from typing import Tuple
import jax
import jax.numpy as jnp

# Enforce binary64 across the module
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants with explicit provenance."""
    # Speed of light in vacuum (SI 2019 exact definition)
    c: float = 299_792_458.0  # m/s
    
    # Proton rest mass (CODATA 2022 recommended value: 1.67262192595(52)e-27 kg)
    m0: float = 1.672_621_925_95e-27  # kg
    
    # Minkowski metric tensor eta_mu_nu: diag(+1, -1, -1, -1)
    eta: jnp.ndarray = field(
        default_factory=lambda: jnp.diag(jnp.array([1.0, -1.0, -1.0, -1.0], dtype=jnp.float64))
    )


@dataclass(frozen=True)
class NumericalConfig:
    """Operational numerical acceptance budgets."""
    eps_mach: float = float(jnp.finfo(jnp.float64).eps)  # ~2.22e-16
    
    tol_rest_energy: float = 10.0 * eps_mach                # ~2.22e-15
    tol_hessian_rel: float = 100.0 * eps_mach               # ~2.22e-14
    tol_hessian_sym: float = 10.0 * eps_mach                # ~2.22e-15
    tol_four_momentum_cross: float = 1e-13
    tol_mass_scaling: float = 50.0 * eps_mach               # ~1.11e-14
    tol_golub_welsch_spectral: float = 50.0 * eps_mach      # ~1.11e-14
    tol_quadrature_exactness: float = 1e-14
    tol_minkowski_condition_scaled: float = 100.0 * eps_mach  # ~2.22e-14
    tol_work_energy_convergence: float = 1e-14
    
    n_random_samples: int = 1000
    random_seeds: Tuple[int, ...] = (42, 101, 2024)
    beta_min_log: float = -6.0                              # beta = 1e-6
    beta_max_log: float = -0.000001                         # beta ~ 0.9999977
    quad_nodes_list: Tuple[int, ...] = (4, 8, 16, 32, 64)


PHYSICS = PhysicalConstants()
CONFIG = NumericalConfig()
