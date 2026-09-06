"""Execution orchestrator for the 8-stage numerical verification suite."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math
import jax
import jax.numpy as jnp

from .constants import PhysicalConstants, NumericalConfig, PHYSICS, CONFIG
from .kinematics import (
    canonical_momentum_ad,
    hamiltonian_energy_ad,
    four_momentum_ad,
    minkowski_contraction,
    closed_form_four_momentum,
)
from .quadrature import roots_and_weights_golub_welsch, integrate_work_quadrature


@dataclass
class TestRecord:
    test_id: str
    name: str
    status: str  # "PASS", "FAIL", or "INVALID"
    metrics: Dict[str, Any]
    details: str


class NumericalVerificationSuite:
    def __init__(self, phys: PhysicalConstants = PHYSICS, cfg: NumericalConfig = CONFIG):
        self.phys = phys
        self.cfg = cfg
        self.records: List[TestRecord] = []

    def _log(self, record: TestRecord):
        self.records.append(record)
        print(f"[{record.status}] {record.name}")
        print(f"       -> {record.details}")

    def run_all_tests(self) -> Tuple[bool, str]:
        print("=" * 90)
        print("  JAX NUMERICAL VERIFICATION SUITE: SPECIAL-RELATIVISTIC KINEMATICS (FLOAT64)")
        print(f"  IEEE-754 Machine Epsilon (Float64): {self.cfg.eps_mach:.2e}")
        print("=" * 90)

        self.test_a_golub_welsch_spectral()
        self.test_b_rest_energy()
        self.test_c_hessian_curvature()
        self.test_d_four_momentum_cross_check()
        self.test_e_minkowski_invariance_conditioning_stress()
        self.test_f_work_energy_convergence()
        self.test_g_gross_fault_injection()
        self.test_h_mass_scaling_homogeneity()

        statuses = [r.status for r in self.records]
        if any(s == "INVALID" for s in statuses):
            final_status = "INVALID (One or more tests encountered invalid/non-finite execution)"
            all_passed = False
        elif any(s == "FAIL" for s in statuses):
            final_status = "FAIL (One or more numerical acceptance budgets were exceeded)"
            all_passed = False
        else:
            final_status = "PASS (All numerical verification consistency criteria satisfied)"
            all_passed = True
        
        print("\n" + "=" * 90)
        print(f"  FINAL VERIFICATION VERDICT: {final_status}")
        print("=" * 90)
        return all_passed, final_status

    # (Seluruh method test_a sampai test_h dari kode final turn sebelumnya dimuat di sini)
    def test_a_golub_welsch_spectral(self) -> bool:
        n_pts = 16
        nodes, weights, V = roots_and_weights_golub_welsch(n_pts)
        if not (jnp.all(jnp.isfinite(nodes)) and jnp.all(jnp.isfinite(weights)) and jnp.all(jnp.isfinite(V))):
            self._log(TestRecord("TEST_A", "Golub-Welsch Spectral Validation", "INVALID", {}, "Non-finite outputs."))
            return False
        I_mat = jnp.eye(n_pts, dtype=jnp.float64)
        ortho_err = float(jnp.linalg.norm(V.T @ V - I_mat) / jnp.sqrt(n_pts))
        sum_w_err = abs(float(jnp.sum(weights)) - 2.0)
        nodes_ok = bool(jnp.all(nodes > -1.0) and jnp.all(nodes < 1.0))
        weights_ok = bool(jnp.all(weights > 0.0))
        max_poly_err = 0.0
        for k in range(2 * n_pts):
            exact = 0.0 if (k % 2 == 1) else (2.0 / (k + 1.0))
            approx = float(jnp.sum(weights * (nodes ** k)))
            max_poly_err = max(max_poly_err, abs(approx - exact))
        spectral_passed = (nodes_ok and weights_ok and ortho_err <= self.cfg.tol_golub_welsch_spectral and sum_w_err <= self.cfg.tol_golub_welsch_spectral)
        quadrature_passed = max_poly_err <= self.cfg.tol_quadrature_exactness
        passed = spectral_passed and quadrature_passed
        metrics = {"orthogonality_error": ortho_err, "weight_sum_error": sum_w_err, "max_polynomial_exactness_error": max_poly_err}
        self._log(TestRecord("TEST_A", f"Golub-Welsch Spectral Validation (N={n_pts})", "PASS" if passed else "FAIL", metrics, f"V^T V Ortho: {ortho_err:.2e}, Sum(w) Err: {sum_w_err:.2e}, Poly Exactness: {max_poly_err:.2e}"))
        return passed

    def test_b_rest_energy(self) -> bool:
        v_zero = jnp.zeros(3, dtype=jnp.float64)
        E_zero = hamiltonian_energy_ad(v_zero, self.phys.m0, self.phys.c)
        E_expected = self.phys.m0 * (self.phys.c ** 2)
        abs_res = abs(float(E_zero) - E_expected)
        rel_res = abs_res / E_expected
        passed = rel_res <= self.cfg.tol_rest_energy
        self._log(TestRecord("TEST_B", "Rest-Energy Consistency H(0) vs m0*c^2", "PASS" if passed else "FAIL", {"rel_residual": rel_res}, f"Abs: {abs_res:.2e} J, Rel: {rel_res:.2e}"))
        return passed

    def test_c_hessian_curvature(self) -> bool:
        v_zero = jnp.zeros(3, dtype=jnp.float64)
        energy_fn = lambda v: hamiltonian_energy_ad(v, self.phys.m0, self.phys.c)
        H_mat = jax.hessian(energy_fn)(v_zero)
        I_mat = jnp.eye(3, dtype=jnp.float64)
        H_normalized = H_mat / self.phys.m0
        rel_frobenius_err = float(jnp.linalg.norm(H_normalized - I_mat) / jnp.sqrt(3.0))
        sym_norm = float(jnp.linalg.norm(H_mat))
        rel_sym_err = float(jnp.linalg.norm(H_mat - H_mat.T)) / sym_norm if sym_norm > 0 else 0.0
        eigenvals_norm = jnp.linalg.eigvalsh(H_normalized)
        max_eig_err = float(jnp.max(jnp.abs(eigenvals_norm - 1.0)))
        passed = (rel_frobenius_err <= self.cfg.tol_hessian_rel and rel_sym_err <= self.cfg.tol_hessian_sym and max_eig_err <= self.cfg.tol_hessian_rel)
        self._log(TestRecord("TEST_C", "Mass Hessian Tensor Curvature at Rest", "PASS" if passed else "FAIL", {"rel_frobenius": rel_frobenius_err}, f"Frobenius Rel (H/m0 - I): {rel_frobenius_err:.2e}, Symmetry: {rel_sym_err:.2e}, Eigenval: {max_eig_err:.2e}"))
        return passed

    def test_d_four_momentum_cross_check(self) -> bool:
        test_betas = jnp.array([1e-4, 0.1, 0.5, 0.9, 0.99, 0.999], dtype=jnp.float64)
        raw_dirs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype=jnp.float64)
        directions = raw_dirs / jnp.linalg.norm(raw_dirs, axis=-1, keepdims=True)
        max_rel_diff = 0.0
        for b in test_betas:
            for d in directions:
                v = d * (float(b) * self.phys.c)
                p_ad = four_momentum_ad(v, self.phys.m0, self.phys.c)
                p_closed = closed_form_four_momentum(v, self.phys.m0, self.phys.c)
                rel_diff = float(jnp.linalg.norm(p_ad - p_closed) / jnp.linalg.norm(p_closed))
                max_rel_diff = max(max_rel_diff, rel_diff)
        passed = max_rel_diff <= self.cfg.tol_four_momentum_cross
        self._log(TestRecord("TEST_D", "Cross-Validation: AD vs Closed-Form (5 Directions)", "PASS" if passed else "FAIL", {"max_rel_diff": max_rel_diff}, f"Max Rel Difference: {max_rel_diff:.2e}"))
        return passed

    def test_e_minkowski_invariance_conditioning_stress(self) -> bool:
        c, m0, expected = self.phys.c, self.phys.m0, (self.phys.m0 * self.phys.c) ** 2
        stress_betas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 0.999999]
        raw_dirs = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype=jnp.float64)
        dirs = raw_dirs / jnp.linalg.norm(raw_dirs, axis=-1, keepdims=True)
        v_stress = jnp.stack([d * (float(b) * c) for b in stress_betas for d in dirs])
        vmap_minkowski = jax.jit(jax.vmap(lambda v: minkowski_contraction(four_momentum_ad(v, m0, c), self.phys.eta)))
        all_raw_errors, all_scaled_errors, all_betas, all_kappas = [], [], [], []
        casimir_stress = vmap_minkowski(v_stress)
        betas_stress = jnp.linalg.norm(v_stress, axis=-1) / c
        kappa_stress = (1.0 + betas_stress ** 2) / (1.0 - betas_stress ** 2)
        raw_err_stress = jnp.abs(casimir_stress - expected) / expected
        all_raw_errors.extend([float(e) for e in raw_err_stress])
        all_scaled_errors.extend([float(e) for e in (raw_err_stress / kappa_stress)])
        all_betas.extend([float(b) for b in betas_stress])
        all_kappas.extend([float(k) for k in kappa_stress])
        for seed in self.cfg.random_seeds:
            key = jax.random.PRNGKey(seed)
            k_dir, k_spd = jax.random.split(key)
            r_dirs = jax.random.normal(k_dir, shape=(self.cfg.n_random_samples, 3))
            r_dirs = r_dirs / jnp.linalg.norm(r_dirs, axis=1, keepdims=True)
            log_b = jax.random.uniform(k_spd, shape=(self.cfg.n_random_samples, 1), minval=self.cfg.beta_min_log, maxval=self.cfg.beta_max_log)
            betas_rand = 10.0 ** log_b
            v_rand = r_dirs * (betas_rand * c)
            speeds = jnp.linalg.norm(v_rand, axis=1)
            casimir_rand = vmap_minkowski(v_rand)
            b_rand = speeds / c
            kappa_rand = (1.0 + b_rand ** 2) / (1.0 - b_rand ** 2)
            raw_err_rand = jnp.abs(casimir_rand - expected) / expected
            all_raw_errors.extend([float(e) for e in raw_err_rand])
            all_scaled_errors.extend([float(e) for e in (raw_err_rand / kappa_rand)])
            all_betas.extend([float(b) for b in b_rand])
            all_kappas.extend([float(k) for k in kappa_rand])
        scaled_arr = jnp.array(all_scaled_errors)
        max_idx = int(jnp.argmax(scaled_arr))
        max_scaled_err = float(scaled_arr[max_idx])
        passed = max_scaled_err <= self.cfg.tol_minkowski_condition_scaled
        self._log(TestRecord("TEST_E", rf"On-Shell Minkowski Invariant P^\mu P_\mu ({len(all_raw_errors)} samples)", "PASS" if passed else "FAIL", {"max_scaled_error": max_scaled_err}, rf"Max Raw Err: {float(jnp.max(jnp.array(all_raw_errors))):.2e} (kappa={float(all_kappas[max_idx]):.1e}), Max Scaled: {max_scaled_err:.2e}"))
        return passed

    def test_f_work_energy_convergence(self) -> bool:
        c, m0 = self.phys.c, self.phys.m0
        v_test = jnp.array([0.5, 0.4, 0.3], dtype=jnp.float64) * c
        v_norm = float(jnp.linalg.norm(v_test))
        gamma_scalar = 1.0 / math.sqrt(1.0 - (v_norm / c) ** 2)
        p_final_closed = gamma_scalar * m0 * v_norm
        delta_E_ref = (gamma_scalar - 1.0) * m0 * (c ** 2)
        errors = [abs(float(integrate_work_quadrature(p_final_closed, m0, n_points=n)) - delta_E_ref) / delta_E_ref for n in self.cfg.quad_nodes_list]
        final_err = errors[-1]
        reduction = errors[0] / max(final_err, self.cfg.eps_mach)
        p_order = float(jnp.log2(errors[0] / max(errors[1], self.cfg.eps_mach)))
        passed = (final_err <= self.cfg.tol_work_energy_convergence) and (reduction > 1e4)
        self._log(TestRecord("TEST_F", "Work-Energy Quadrature Convergence vs Scalar Reference", "PASS" if passed else "FAIL", {"final_error": final_err}, f"Error N=4: {errors[0]:.2e} -> N=64: {final_err:.2e} (Reduction: {reduction:.1e}x, Slope: ~{p_order:.1f})"))
        return passed

    def test_g_gross_fault_injection(self) -> bool:
        c, m0, expected = self.phys.c, self.phys.m0, (self.phys.m0 * self.phys.c) ** 2
        v_sample = jnp.array([0.5 * c, 0.0, 0.0], dtype=jnp.float64)
        p_corrupted = four_momentum_ad(v_sample, m0, c) * jnp.array([1.001, 1.0, 1.0, 1.0])
        raw_err = abs(float(minkowski_contraction(p_corrupted, self.phys.eta)) - expected) / expected
        detected_p = (raw_err / ((1.0 + 0.25) / (1.0 - 0.25))) > self.cfg.tol_minkowski_condition_scaled
        _, weights, _ = roots_and_weights_golub_welsch(16)
        detected_w = abs(float(jnp.sum(weights * 1.01)) - 2.0) > self.cfg.tol_golub_welsch_spectral
        passed = detected_p and detected_w
        self._log(TestRecord("TEST_G", "Sensitivity Audit: Gross-Fault Injection Detection", "PASS" if passed else "FAIL", {"detected_p": detected_p, "detected_w": detected_w}, f"Detected P0 Fault (+0.1%): {detected_p}, Detected Quadrature Fault (+1.0%): {detected_w}"))
        return passed

    def test_h_mass_scaling_homogeneity(self) -> bool:
        c, m0 = self.phys.c, self.phys.m0
        v_sample = jnp.array([0.6 * c, -0.3 * c, 0.2 * c], dtype=jnp.float64)
        E_base, p_base = hamiltonian_energy_ad(v_sample, m0, c), canonical_momentum_ad(v_sample, m0, c)
        max_scaling_err = 0.0
        for s in [0.5, 2.0, 10.0]:
            E_scaled, p_scaled = hamiltonian_energy_ad(v_sample, s * m0, c), canonical_momentum_ad(v_sample, s * m0, c)
            err_E = abs(float(E_scaled) - s * float(E_base)) / (s * float(E_base))
            err_p = float(jnp.linalg.norm(p_scaled - s * p_base) / (s * jnp.linalg.norm(p_base)))
            max_scaling_err = max(max_scaling_err, err_E, err_p)
        passed = max_scaling_err <= self.cfg.tol_mass_scaling
        self._log(TestRecord("TEST_H", "Degree-1 Mass-Scaling Homogeneity", "PASS" if passed else "FAIL", {"max_scaling_error": max_scaling_err}, f"Max Homogeneity Error: {max_scaling_err:.2e}"))
        return passed


def main():
    suite = NumericalVerificationSuite()
    success, _ = suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
