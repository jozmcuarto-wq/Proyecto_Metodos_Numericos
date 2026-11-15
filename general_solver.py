"""Safe, two-variable solver used by the interactive system-comparison page.

The report-specific experiment in ``core.py`` keeps its analytic Jacobian and
vectorized implementation. This module supports user-entered systems using a
centered finite-difference Jacobian so that the same three iterations can be
explored on other functions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np

FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
CONSTANTS = {"pi": np.pi, "e": np.e}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.UAdd, ast.USub,
)
MAX_EXPRESSION_LENGTH = 240
MAX_AST_NODES = 80
SINGULAR_TOLERANCE = 1e-12
DIVERGENCE_LIMIT = 1e8


class ExpressionError(ValueError):
    """A user expression is invalid or uses an unsupported operation."""


@dataclass
class SolveResult:
    method: str
    status: str
    iterations: int
    x: float
    y: float
    residual: float
    path: np.ndarray


def compile_expression(expression: str):
    """Compile a restricted arithmetic expression in variables ``x`` and ``y``."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Cada expresión debe tener como máximo {MAX_EXPRESSION_LENGTH} caracteres.")
    normalized = expression.strip().replace("×", "*").replace("^", "**")
    if not normalized:
        raise ExpressionError("La expresión no puede estar vacía.")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"No pude interpretar la expresión: {exc.msg}.") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ExpressionError("La expresión es demasiado compleja.")
    call_names = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            raise ExpressionError("Usa solo operaciones aritméticas y funciones matemáticas permitidas.")
        if isinstance(node, ast.Name) and node.id not in {"x", "y", *CONSTANTS, *FUNCTIONS}:
            raise ExpressionError(f"Nombre no permitido: {node.id}.")
        if isinstance(node, ast.Name) and node.id in FUNCTIONS and id(node) not in call_names:
            raise ExpressionError(f"La función {node.id} requiere argumentos, por ejemplo {node.id}(x).")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError("Usa solo constantes numéricas, x, y, pi y e.")
            try:
                numeric_value = float(node.value)
            except (OverflowError, ValueError) as exc:
                raise ExpressionError("El valor numérico escrito es demasiado grande.") from exc
            if not np.isfinite(numeric_value) or abs(numeric_value) > 1e100:
                raise ExpressionError("Los números escritos en la expresión deben tener magnitud menor que 1e100.")
            if isinstance(node.value, int):
                node.value = numeric_value
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ExpressionError("Solo se permiten funciones matemáticas como sin, cos, exp, log y sqrt.")
            if node.keywords:
                raise ExpressionError("Las funciones deben escribirse con argumentos posicionales.")

    code = compile(tree, "<expresión del usuario>", "eval")
    scope = {**FUNCTIONS, **CONSTANTS}

    def evaluate(x: float, y: float):
        with np.errstate(all="ignore"):
            value = eval(code, {"__builtins__": {}},
                         {**scope, "x": np.asarray(x, dtype=float),
                          "y": np.asarray(y, dtype=float)})
        value_array = np.asarray(value)
        if np.iscomplexobj(value_array):
            raise ExpressionError("Cada función debe devolver valores reales.")
        if value_array.ndim == 0:
            return float(value_array)
        return value_array.astype(float, copy=False)

    return evaluate


def compile_system(f1_expression: str, f2_expression: str):
    """Return the two scalar functions for a system F(x, y) = 0."""
    return compile_expression(f1_expression), compile_expression(f2_expression)


def _evaluate(functions, x: float, y: float) -> np.ndarray:
    return np.asarray([functions[0](x, y), functions[1](x, y)], dtype=float)


def _jacobian(functions, x: float, y: float) -> np.ndarray:
    """Centered finite-difference Jacobian, scaled to the current coordinates."""
    step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(x), abs(y))
    dx_plus = _evaluate(functions, x + step, y)
    dx_minus = _evaluate(functions, x - step, y)
    dy_plus = _evaluate(functions, x, y + step)
    dy_minus = _evaluate(functions, x, y - step)
    return np.column_stack(((dx_plus - dx_minus) / (2 * step),
                            (dy_plus - dy_minus) / (2 * step)))


def _singular_jacobian(jacobian: np.ndarray) -> tuple[bool, float, float]:
    a, b = jacobian[0]
    c, d = jacobian[1]
    determinant = a * d - b * c
    scale = float(np.max(np.sum(np.abs(jacobian), axis=1)))
    singular = (not np.isfinite(scale) or scale == 0.0
                or not np.isfinite(determinant)
                or abs(determinant) <= SINGULAR_TOLERANCE * scale**2)
    return singular, determinant, scale


def solve_one(
    method: str,
    functions,
    x0: float,
    y0: float,
    *,
    alpha: float = 0.5,
    h: float = 0.5,
    p0: tuple[float, float] = (0.0, 0.0),
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> SolveResult:
    """Run NR, damped Newton (DN), or W4 from one shared starting point."""
    if method not in {"NR", "DN", "W4"}:
        raise ValueError("El método debe ser NR, DN o W4.")
    if not 0.0 < alpha <= 1.0 or not 0.0 < h < 1.0:
        raise ValueError("Se requiere 0 < alpha <= 1 y 0 < h < 1.")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("La tolerancia y el máximo de iteraciones deben ser positivos.")

    x, y = float(x0), float(y0)
    p = np.asarray(p0, dtype=float)
    path = [(x, y)]

    for iteration in range(max_iterations + 1):
        if (not np.isfinite(x) or not np.isfinite(y)
                or abs(x) > DIVERGENCE_LIMIT or abs(y) > DIVERGENCE_LIMIT):
            residual = float("inf")
            status = "Divergencia"
            break

        with np.errstate(all="ignore"):
            value = _evaluate(functions, x, y)
            residual = float(np.linalg.norm(value))
        if not np.isfinite(residual):
            status = "Divergencia"
            break
        if residual <= tolerance:
            status = "Convergió"
            break
        if iteration == max_iterations:
            status = "Agotó el máximo de iteraciones"
            break

        with np.errstate(all="ignore"):
            jacobian = _jacobian(functions, x, y)
        if not np.all(np.isfinite(jacobian)):
            status = "Jacobiano no finito"
            break
        singular, determinant, scale = _singular_jacobian(jacobian)

        if method == "W4":
            # For J=[[a,b],[c,d]], J=U D L with unit triangular U,L,
            # D=diag(det(J)/d, d). This is the same right-pivot UDL
            # convention used by the project's specialized implementation.
            a, b = jacobian[0]
            c, d = jacobian[1]
            pivot1 = determinant / d if d != 0.0 else 0.0
            if (singular or abs(d) <= SINGULAR_TOLERANCE * scale
                    or abs(pivot1) <= SINGULAR_TOLERANCE * scale):
                status = "Pivote UDL nulo"
                break

            with np.errstate(all="ignore"):
                transformed_p = np.array([p[0], p[1] - (c / d) * p[0]])
                transformed_f = np.array([
                    (d * value[0] - b * value[1]) / determinant,
                    value[1] / d,
                ])
                x += h * transformed_p[0]
                y += h * transformed_p[1]
                p = (1.0 - 2.0 * h) * p - h * transformed_f
        else:
            if singular:
                status = "Jacobiano singular"
                break
            try:
                correction = np.linalg.solve(jacobian, value)
            except np.linalg.LinAlgError:
                status = "Jacobiano singular"
                break
            factor = 1.0 if method == "NR" else alpha
            x -= factor * correction[0]
            y -= factor * correction[1]

        path.append((float(x), float(y)))
    else:  # pragma: no cover - the loop exits through a status above
        status = "Agotó el máximo de iteraciones"

    return SolveResult(method, status, iteration, float(x), float(y),
                       float(residual), np.asarray(path, dtype=float))


def compare_methods(
    f1_expression: str,
    f2_expression: str,
    x0: float,
    y0: float,
    *,
    alpha: float = 0.5,
    h: float = 0.5,
    p0: tuple[float, float] = (0.0, 0.0),
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> list[SolveResult]:
    """Compare all three methods under a shared system and initial point."""
    functions = compile_system(f1_expression, f2_expression)
    return [
        solve_one(method, functions, x0, y0, alpha=alpha, h=h, p0=p0,
                  tolerance=tolerance, max_iterations=max_iterations)
        for method in ("NR", "DN", "W4")
    ]
