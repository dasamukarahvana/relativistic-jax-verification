
# relativistic-jax-verification


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Float64%20Enabled-green.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A machine-checked **numerical consistency and conditioning verification suite** for special-relativistic kinematic formulations implemented in **Google JAX (IEEE-754 binary64)**.

---

## 🎯 Scope & Epistemic Boundaries

To maintain scientific rigor and avoid overclaiming, this repository defines strict epistemic boundaries:

| What This Repository **IS** | What This Repository **IS NOT** |
| :--- | :--- |
| **A Numerical Consistency Suite:** Audits floating-point propagation, Legendre transform variational consistency, and high-order quadrature in JAX Float64. | **NOT a Formal Proof:** It does not use interactive theorem provers (e.g., Lean 4, Coq) or axiomatic formal logic. |
| **A Conditioning-Aware Audit:** Evaluates on-shell invariant residuals against exact subtraction loss-of-precision factor $\kappa(\beta) = \frac{1+\beta^2}{1-\beta^2}$. | **NOT an Empirical Physics Validation:** It does not fit or compare against experimental particle accelerator data. |
| **A Dual-Path Cross-Check:** Verifies agreement between automatic differentiation (AD) pipelines and closed-form vector kinematics. | **NOT a Proof of Special Relativity:** It verifies the software implementation of relativistic equations, not nature itself. |

---

## 🔬 The 8-Stage Verification Architecture

The test suite evaluates 8 distinct mathematical and numerical properties:

```text
                  ┌────────────────────────────────────────────────────────┐
                  │ 1. Variational Action Principle: S = -m0 c ∫ dτ        │
                  │    Lagrangian: L(v) = -m0 c² √(1 - ||v||²/c²)          │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
       ┌────────────────────────────────┐            ┌────────────────────────────────┐
       │     Path A: Automatic Diff     │            │    Path B: Independent Ref     │
       │  p = ∇_v L                     │            │  γ = 1 / √(1 - β²)             │
       │  H = p · v - L (Legendre)      │            │  P_closed = (γ m0 c, γ m0 v)   │
       │  P_AD = (H/c, p)               │            │  E_ref = (γ - 1) m0 c²         │
       └───────────────┬────────────────┘            └───────────────┬────────────────┘
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              │
                      Cross-Validation & Invariant Consistency Audits:
                      ├── Test A: Golub-Welsch Spectral Eigensolver (N=16)
                      ├── Test B: Rest-Energy H(0) == m0 c²
                      ├── Test C: Mass Hessian Curvature Tensor at Rest
                      ├── Test D: Four-Momentum Cross-Path Agreement (5 Spatial Directions)
                      ├── Test E: On-Shell Minkowski Contraction & Conditioning Stress
                      ├── Test F: Work-Energy Quadrature Convergence (N=4 → 64)
                      ├── Test G: Gross-Fault Sensitivity Detection (Negative Testing)
                      └── Test H: Degree-1 Positive Mass-Scaling Homogeneity
```

---

## 📊 Summary of Verification Criteria

| Test ID | Objective | Primary Metric | Acceptance Budget | Observed Residual | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **TEST_A** | Quadrature Backend Validity | $V^T V \text{ Ortho} + \text{Poly Exactness}$ | $\le 50\,\epsilon_{\text{mach}} / 10^{-14}$ | $1.94 \times 10^{-15} / 4.11 \times 10^{-15}$ | **PASS** |
| **TEST_B** | Rest-Energy Equivalence | $\frac{\|H(0) - m_0 c^2\|}{m_0 c^2}$ | $\le 10\,\epsilon_{\text{mach}}$ | $0.00 \times 10^{0}$ | **PASS** |
| **TEST_C** | Mass Hessian Curvature | $\frac{\|\mathcal{H}/m_0 - I\|_F}{\sqrt{3}}$ | $\le 100\,\epsilon_{\text{mach}}$ | $0.00 \times 10^{0}$ | **PASS** |
| **TEST_D** | Dual-Path Cross-Validation | $\frac{\|P_{\text{AD}} - P_{\text{closed}}\|}{\|P_{\text{closed}}\|}$ | $\le 1.0 \times 10^{-13}$ | $1.94 \times 10^{-16}$ | **PASS** |
| **TEST_E** | Minkowski Invariant Stress | $\max \left( \frac{\text{Raw Err}}{\kappa(\beta)} \right)$ | $\le 100\,\epsilon_{\text{mach}}$ | $1.59 \times 10^{-16}$ | **PASS** |
| **TEST_F** | Work-Energy Convergence | $\|W_{N=64} - \Delta E_{\text{ref}}\| / \Delta E_{\text{ref}}$ | $\le 1.0 \times 10^{-14}$ | $8.30 \times 10^{-16}$ | **PASS** |
| **TEST_G** | Fault Sensitivity Audit | Injected $+0.1\% P^0$ & $+1.0\% w_i$ | Detection $\equiv \text{True}$ | Detected $\equiv \text{True}$ | **PASS** |
| **TEST_H** | Mass-Scaling Homogeneity | $\frac{\|E(s\cdot m) - s\cdot E(m)\|}{s\cdot E(m)}$ | $\le 50\,\epsilon_{\text{mach}}$ | $0.00 \times 10^{0}$ | **PASS** |

*Note: $\epsilon_{\text{mach}} = 2^{-52} \approx 2.2204 \times 10^{-16}$ (IEEE-754 binary64 double precision).*

---

## 🚀 Quickstart

### Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/your-username/relativistic-jax-verification.git
cd relativistic-jax-verification
pip install -e ".[dev]"
```

### Run Verification Suite via CLI

```bash
python -m relativistic_jax.suite
```

### Run with Pytest

```bash
pytest -v
```

---

## 📘 Conditioning Analysis in Test E

When computing the Minkowski norm $P^\mu \eta_{\mu\nu} P^\nu = (P^0)^2 - \|\mathbf{p}\|^2$ near the speed of light ($\beta \to 1$), catastrophic cancellation occurs. The condition number for subtracting two nearly identical large floating-point quantities is:

$$\kappa(\beta) = \frac{(P^0)^2 + \|\mathbf{p}\|^2}{|(P^0)^2 - \|\mathbf{p}\|^2|} = \frac{1 + \beta^2}{1 - \beta^2} = (1 + \beta^2)\gamma^2$$

At $\beta = 0.999999$, $\kappa \approx 2.0 \times 10^6$. The raw absolute error naturally scales up to $\approx 1.59 \times 10^{-10}$. When normalized against the exact condition factor:

$$\rho_{\text{scaled}} = \frac{\text{Raw Relative Error}}{\kappa(\beta)} \approx \mathbf{1.59 \times 10^{-16}} \le 1.0 \cdot \epsilon_{\text{mach}}$$

This proves the floating-point calculation is backward-stable to within a single machine epsilon.

---

## 📂 Repository Structure

```text
relativistic-jax-verification/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated multi-OS, multi-Python CI Pipeline
├── src/
│   └── relativistic_jax/
│       ├── __init__.py            # Clean top-level package exports
│       ├── constants.py           # CODATA 2022 & SI exact physical constants
│       ├── kinematics.py          # Variational AD & closed-form kinematics
│       ├── quadrature.py          # Golub-Welsch JAX eigensolver & work integral
│       └── suite.py               # 8-Stage test suite orchestrator & CLI runner
├── tests/
│   ├── __init__.py
│   └── test_verification.py       # Automated Pytest test wrapper
├── .gitignore
├── CITATION.cff                   # Standard academic citation metadata
├── LICENSE                        # Apache-2.0 License
├── pyproject.toml                 # Modern PEP 517/621 packaging
└── README.md                      # Comprehensive documentation
```

---

## 📜 Citation

If you reference or use this verification suite in scientific software or publications, please cite:

```bibtex
@software{relativistic_jax_verification,
  author = {Your Name / Research Group},
  title = {relativistic-jax-verification: A JAX Float64 Numerical Consistency Suite for Special-Relativistic Kinematics},
  year = {2026},
  url = {https://github.com/dasamukarahvana/relativistic-jax-verification/tree/main
}
```

## 📄 License

Distributed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.
```
