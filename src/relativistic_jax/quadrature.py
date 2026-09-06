"""Gauss-Legendre quadrature engine via pure JAX spectral decomposition."""

from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
from .constants import PHYSICS


@partial(jax.jit, static_argnums=(0,))
def roots_and_weights_golub_welsch(n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Computes Gauss-Legendre quadrature nodes, weights, and orthogonal eigenvector matrix
    via symmetric tridiagonal Jacobi matrix spectral decomposition (Golub-Welsch 1969).
    """
    k = jnp.arange(1, n_points, dtype=jnp.float64)
    beta = k / jnp.sqrt(4.0 * (k ** 2) - 1.0)
    J = jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    eigenvalues, eigenvectors = jnp.linalg.eigh(J)
    nodes = eigenvalues
    weights = 2.0 * (eigenvectors[0, :] ** 2)
    return nodes, weights, eigenvectors


@partial(jax.jit, static_argnums=(2,))
def integrate_work_quadrature(p_final: float, m0: float, n_points: int) -> jnp.ndarray:
    r"""Evaluates W = \int_0^{p_final} v(p) dp using Gauss-Legendre quadrature."""
    c = PHYSICS.c
    nodes, weights, _ = roots_and_weights_golub_welsch(n_points)
    p_nodes = 0.5 * p_final * (nodes + 1.0)
    scaled_weights = 0.5 * p_final * weights

    c2 = c ** 2
    c4 = c ** 4
    v_integrand = (p_nodes * c2) / jnp.sqrt((p_nodes ** 2) * c2 + (m0 ** 2) * c4)
    return jnp.sum(scaled_weights * v_integrand)
