"""Diagnósticos locales con las mismas definiciones del informe final."""

from __future__ import annotations

import numpy as np

from core import ROOTS, _w4_step, trayectoria


RATE_BANDS = (
    (1e-4, 1e-5), (1e-5, 1e-6), (1e-6, 1e-7), (1e-7, 1e-8),
    (1e-8, 1e-9), (1e-9, 1e-10), (1e-10, 1e-11), (1e-11, 1e-12),
)
ORDER_OFFSETS = ((0.25, -0.175), (0.30, -0.21), (0.35, -0.245), (0.40, -0.28))


def local_rate_summary(method: str, alpha: float = 0.5, h: float = 0.5,
                       p0: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Return report-compatible error ratios over four roots and error bands."""
    name = method.upper()
    if name not in {"DN", "W4"}:
        raise ValueError("local_rate_summary supports DN and W4")
    step = alpha if name == "DN" else h
    ratios: list[float] = []
    by_band: dict[tuple[float, float], list[float]] = {band: [] for band in RATE_BANDS}

    for root in ROOTS:
        points = trayectoria(
            name, float(root[0] + 0.25), float(root[1] - 0.175),
            alpha=alpha, h=h, maxit=600, p0=p0, tol=1e-15,
        )
        errors = np.hypot(points[:, 0] - root[0], points[:, 1] - root[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            values = errors[1:] / errors[:-1]
        for k, value in enumerate(values):
            if 1e-11 < errors[k] < 1e-7 and np.isfinite(value):
                ratios.append(float(value))
            for upper, lower in RATE_BANDS:
                if lower < errors[k] <= upper and np.isfinite(value):
                    by_band[(upper, lower)].append(float(value))
                    break

    band_medians = {
        band: float(np.median(values)) if values else np.nan
        for band, values in by_band.items()
    }
    vals = np.asarray(ratios, dtype=float)
    return {
        "method": name,
        "theoretical": 1.0 - step,
        "count": int(vals.size),
        "median": float(np.median(vals)) if vals.size else np.nan,
        "minimum": float(np.min(vals)) if vals.size else np.nan,
        "maximum": float(np.max(vals)) if vals.size else np.nan,
        "bands": band_medians,
    }


def nr_order_by_root() -> list[dict[str, float]]:
    """Estimate q with the four fixed perturbations used by the report."""
    rows = []
    for root in ROOTS:
        estimates: list[float] = []
        for offset in ORDER_OFFSETS:
            points = trayectoria(
                "NR", float(root[0] + offset[0]), float(root[1] + offset[1]),
                maxit=80, tol=1e-15,
            )
            errors = np.hypot(points[:, 0] - root[0], points[:, 1] - root[1])
            for a, b, c in zip(errors[:-2], errors[1:-1], errors[2:]):
                if 1e-14 < c < b < a and b < 1e-2:
                    denominator = np.log(b / a)
                    if denominator != 0.0:
                        estimates.append(float(np.log(c / b) / denominator))
        q = np.asarray(estimates, dtype=float)
        rows.append({
            "count": int(q.size),
            "median": float(np.median(q)),
            "minimum": float(np.min(q)),
            "maximum": float(np.max(q)),
        })
    return rows


def w4_spectral_radius(root_index: int = 0, h: float = 0.5, step_size: float = 1e-5) -> float:
    """Estimate the W4 map's local spectral radius by centered differences."""
    root = ROOTS[root_index]

    def update(z: np.ndarray) -> np.ndarray:
        x, y, p, q, _ = _w4_step(
            np.array([z[0]]), np.array([z[1]]),
            np.array([z[2]]), np.array([z[3]]), h,
        )
        return np.array([x[0], y[0], p[0], q[0]])

    z0 = np.r_[root, 0.0, 0.0]
    jac = np.empty((4, 4), dtype=float)
    for j in range(4):
        offset = np.zeros(4)
        offset[j] = step_size
        jac[:, j] = (update(z0 + offset) - update(z0 - offset)) / (2 * step_size)
    return float(np.max(np.abs(np.linalg.eigvals(jac))))
