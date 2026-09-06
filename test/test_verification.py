"""Automated Pytest runner wrapper."""

from relativistic_jax.suite import NumericalVerificationSuite


def test_full_numerical_verification_suite():
    suite = NumericalVerificationSuite()
    all_passed, verdict = suite.run_all_tests()
    assert all_passed, f"Verification suite failed with verdict: {verdict}"
