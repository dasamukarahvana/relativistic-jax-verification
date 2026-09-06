"""Relativistic kinematic formulations: Variational AD and closed-form algebra."""

import jax
import jax.numpy as jnp


def relativistic_lagrangian(v: jnp.ndarray, m0: float, c: float) -> jnp.ndarray:
    r"""L(v) = -m0 * c^2 * sqrt(1 - ||v||^2 / c^2)"""
    v_sq = jnp.sum(v ** 2, axis=-1)
    gamma_inv = jnp.sqrt(1.0 - v_sq / (c ** 2))
    return -m0 * (c ** 2) * gamma_inv


# Canonical conjugate momentum p = \nabla_v L via automatic differentiation
canonical_momentum_ad = jax.grad(relativistic_lagrangian, argnums=0)


def hamiltonian_energy_ad(v: jnp.ndarray, m0: float, c: float) -> jnp.ndarray:
    r"""Legendre transformation: H(v) = p(v) . v - L(v)"""
    p = canonical_momentum_ad(v, m0, c)
    l_val = relativistic_lagrangian(v, m0, c)
    return jnp.dot(p, v) - l_val


def four_momentum_ad(v: jnp.ndarray, m0: float, c: float) -> jnp.ndarray:
    r"""Constructs four-momentum P^\mu = (H/c, p_x, p_y, p_z) via the AD pipeline."""
    E = hamiltonian_energy_ad(v, m0, c)
    p = canonical_momentum_ad(v, m0, c)
    return jnp.concatenate([jnp.array([E / c]), p], axis=-1)


def minkowski_contraction(P: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    r"""Minkowski metric contraction: P^\mu \eta_{\mu\nu} P^\nu (Supports 1D and batched 2D)."""
    return jnp.einsum('...i,ij,...j->...', P, eta, P)


def closed_form_lorentz_gamma(v: jnp.ndarray, c: float) -> jnp.ndarray:
    r"""Lorentz factor \gamma = 1 / sqrt(1 - ||v||^2 / c^2) (Supports 1D and batched 2D)."""
    v_sq = jnp.sum(v ** 2, axis=-1)
    return 1.0 / jnp.sqrt(1.0 - v_sq / (c ** 2))


def closed_form_four_momentum(v: jnp.ndarray, m0: float, c: float) -> jnp.ndarray:
    r"""
    Closed-form four-momentum: P^\mu_closed = (\gamma m0 c, \gamma m0 v).
    Maps shape (3,) -> (4,) and batch shape (N, 3) -> (N, 4).
    """
    gamma = closed_form_lorentz_gamma(v, c)
    E = gamma * m0 * (c ** 2)
    p = gamma[..., None] * m0 * v
    E_over_c = (E / c)[..., None]
    return jnp.concatenate([E_over_c, p], axis=-1)
