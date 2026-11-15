import pathlib
import sys

import numpy as np
import pytest
from scipy.optimize import fsolve

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from core import DIV, EXH, PIV, ROOTS, F, grid, run
from diagnostics import local_rate_summary, nr_order_by_root, w4_spectral_radius


@pytest.fixture(scope="module")
def results_400():
    x, y = grid(400)
    return {method: run(method, x, y) for method in ("NR", "DN", "W4")}


def test_roots_are_polished_and_satisfy_system():
    residual = np.hypot(*F(ROOTS[:, 0], ROOTS[:, 1]))
    assert np.max(residual) < 1e-14
    for root in ROOTS:
        assert np.allclose(fsolve(lambda z: F(z[0], z[1]), root), root, atol=1e-12)


def test_udl_reproduces_jacobian_and_inverse_action():
    x, y = 1.3, -0.7
    delta = x * x - 2 * y * y
    u = np.array([[1.0, 2 * y / x**2], [0.0, 1.0]])
    d = np.diag([2 * delta / x, x**2])
    l = np.array([[1.0, 0.0], [2 * y / x, 1.0]])
    jac = np.array([[2 * x, 2 * y], [2 * x * y, x**2]])
    assert np.allclose(u @ d @ l, jac)
    assert np.allclose(np.linalg.solve(l, np.linalg.solve(d, np.linalg.inv(u))), np.linalg.inv(jac))


@pytest.mark.parametrize(
    "method,success,root_shares",
    [
        ("NR", 70.945625, [20.3475, 20.349375, 15.125625, 15.123125]),
        ("DN", 70.489375, [20.159375, 20.15625, 15.08875, 15.085]),
        ("W4", 99.65, [20.475625, 20.478125, 29.355625, 29.340625]),
    ],
)
def test_basin_shares_match_report(results_400, method, success, root_shares):
    states = results_400[method][0]
    assert 100 * (states < 4).mean() == pytest.approx(success, abs=1e-10)
    observed = [100 * (states == root).mean() for root in range(4)]
    assert observed == pytest.approx(root_shares, abs=1e-10)


def test_terminal_failure_counts_match_report(results_400):
    expected = {"NR": (15, 46472, 0), "DN": (5, 47212, 0), "W4": (548, 0, 12)}
    for method, (diverged, exhausted, pivots) in expected.items():
        states = results_400[method][0]
        assert int((states == DIV).sum()) == diverged
        assert int((states == EXH).sum()) == exhausted
        assert int((states == PIV).sum()) == pivots


def test_failure_overlap_matches_report(results_400):
    w4 = results_400["W4"][0]
    r34 = (w4 == 2) | (w4 == 3)
    expected = {"NR": (90.3392, 41996), "DN": (89.7071, 42357)}
    for method, (pct, count) in expected.items():
        failed = results_400[method][0] >= 4
        overlap = failed & r34
        assert int(overlap.sum()) == count
        assert 100 * overlap.sum() / failed.sum() == pytest.approx(pct, abs=0.1)


def test_w4_divergence_distance_matches_report(results_400):
    states, _, minimum_distance = results_400["W4"]
    median = float(np.median(minimum_distance[states == DIV]))
    assert median == pytest.approx(4.2400739758129136e-4, rel=1e-4)


def test_common_successes_and_iteration_means_match_report(results_400):
    common = np.logical_and.reduce([results_400[m][0] < 4 for m in ("NR", "DN", "W4")])
    assert int(common.sum()) == pytest.approx(106405, abs=5)
    means = [float(results_400[m][1][common].mean()) for m in ("NR", "DN", "W4")]
    assert means == pytest.approx([5.82, 25.48, 37.49], abs=0.02)


def test_nr_local_order_is_quadratic_at_all_roots():
    rows = nr_order_by_root()
    medians = [row["median"] for row in rows]
    assert len(medians) == 4
    assert medians == pytest.approx([1.9920, 2.0055, 1.9729, 1.9962], abs=0.015)


def test_w4_local_rate_and_spectral_radius_match_report():
    rate = local_rate_summary("W4")
    assert rate["median"] == pytest.approx(0.5157, abs=0.001)
    assert w4_spectral_radius(0) == pytest.approx(0.500001, abs=2e-6)
    assert w4_spectral_radius(2) == pytest.approx(0.500002, abs=2e-6)
