import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core import _w4_step
from general_solver import ExpressionError, compare_methods, compile_expression, compile_system, solve_one


def test_all_methods_solve_a_user_defined_polynomial_system():
    results = compare_methods("x^2 + y^2 - 1", "x - y", 0.8, 0.6)

    assert [result.method for result in results] == ["NR", "DN", "W4"]
    assert all(result.status == "Convergió" for result in results)
    for result in results:
        assert result.residual <= 1e-6
        assert result.x == pytest.approx(1 / np.sqrt(2), abs=1e-5)
        assert result.y == pytest.approx(1 / np.sqrt(2), abs=1e-5)
        assert result.path[0].tolist() == [0.8, 0.6]


def test_trigonometric_custom_system_can_be_evaluated_on_a_grid():
    sine = compile_expression("sin(x) + y - 1")
    inverse_sine = compile_expression("asin(sin(x))")
    x = np.array([0.0, 0.5])
    y = np.array([1.0, 0.5])
    assert sine(x, y) == pytest.approx([0.0, np.sin(0.5) - 0.5])
    assert inverse_sine(x, y) == pytest.approx(x)


def test_general_w4_step_matches_the_project_udl_formula():
    x0, y0, p0, h = 1.3, -0.7, (0.2, -0.1), 0.5
    functions = compile_system("x^2+y^2-4", "x^2*y-1")
    result = solve_one("W4", functions, x0, y0, p0=p0, h=h, max_iterations=1)
    expected = _w4_step(
        np.array([x0]), np.array([y0]), np.array([p0[0]]), np.array([p0[1]]), h,
    )
    assert result.path[1] == pytest.approx([expected[0][0], expected[1][0]], abs=1e-6)


@pytest.mark.parametrize("expression", [
    "__import__('os').system('whoami')",
    "x.__class__",
    "sin",
    "'not a number'",
])
def test_expression_parser_rejects_unsupported_or_unsafe_python(expression):
    with pytest.raises(ExpressionError):
        compile_expression(expression)
