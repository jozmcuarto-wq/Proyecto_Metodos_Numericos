"""Núcleo numérico: sistema F, métodos NR / DN / W4 vectorizados y clasificación."""
import numpy as np

ROOTS = np.array([[1.9837924, 0.25410169], [-1.9837924, 0.25410169],
                  [0.73307679, 1.8608059], [-0.73307679, 1.8608059]])
def _pulir(R):
    """Refina las raíces tabuladas (8 dígitos) con Newton hasta precisión de máquina."""
    R = R.copy()
    for _ in range(8):
        x, y = R[:, 0], R[:, 1]
        a, b, c, d = 2 * x, 2 * y, 2 * x * y, x**2
        f1, f2 = x**2 + y**2 - 4, x**2 * y - 1
        det = a * d - b * c
        R[:, 0], R[:, 1] = x - (d * f1 - b * f2) / det, y - (a * f2 - c * f1) / det
    return R


ROOTS = _pulir(ROOTS)
# Estados: 0-3 raíz R1-R4; 4 divergencia; 5 agotó tope; 6 J singular; 7 pivote UDL nulo
DIV, EXH, SING, PIV = 4, 5, 6, 7
NOMBRES = ["R1", "R2", "R3", "R4", "Divergencia", "Agotó el tope", "J singular", "Pivote UDL nulo"]
COLORES = ["#ef6a7d", "#3f74a8", "#cdbb44", "#2a8a3a", "#222222", "#222222", "#222222", "#222222"]
TOL, MAXIT, SING_TOL, DIV_X = 1e-6, 1000, 1e-12, 1e6


def F(x, y):
    return np.stack([x**2 + y**2 - 4, x**2 * y - 1])


def Fnorm(x, y):
    f = F(x, y)
    return np.hypot(f[0], f[1])


def nearest_root(x, y):
    d = (x[:, None] - ROOTS[:, 0]) ** 2 + (y[:, None] - ROOTS[:, 1]) ** 2
    return np.argmin(d, axis=1)


def _newton_dir(x, y):
    """Devuelve la corrección explícita (dx, dy) y detecta J casi singular."""
    f1, f2 = F(x, y)
    a, b, c, d = 2 * x, 2 * y, 2 * x * y, x**2
    det = a * d - b * c
    scale = np.maximum(np.abs(a) + np.abs(b), np.abs(c) + np.abs(d))
    sing = (scale == 0.0) | (np.abs(det) <= SING_TOL * scale**2)
    delta = x**2 - 2 * y**2
    # Keep the same explicit operation order as the validated notebook.  The
    # formulas are algebraically equivalent to Cramer's rule, but their
    # floating-point rounding can change a basin-boundary classification.
    x_safe = np.where(sing, 1.0, x)
    delta_safe = np.where(sing, 1.0, delta)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dx = x_safe / (2.0 * delta_safe) * f1 - y / (x_safe * delta_safe) * f2
        dy = -y / delta_safe * f1 + f2 / delta_safe
    return dx, dy, sing


def _udl_pivot_failure(x, y):
    """Use the report's relative Jacobian and UDL-pivot tolerance tests."""
    a, b, c, d = 2 * x, 2 * y, 2 * x * y, x**2
    scale = np.maximum(np.abs(a) + np.abs(b), np.abs(c) + np.abs(d))
    det = a * d - b * c
    delta = x**2 - 2 * y**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = np.divide(2.0 * delta, x, out=np.full_like(x, np.inf), where=x != 0.0)
        d2 = x**2
        return (
            (scale == 0.0)
            | (np.abs(det) <= SING_TOL * scale**2)
            | (x == 0.0)
            | (np.abs(d1) <= SING_TOL * scale)
            | (np.abs(d2) <= SING_TOL * scale)
        )


def _w4_step(x, y, p1, p2, h):
    """Un paso de W4 con J = U D L (sin formar inversas). Devuelve nuevos estados y pivote nulo."""
    f1, f2 = F(x, y)
    delta = x**2 - 2 * y**2
    piv = _udl_pivot_failure(x, y)
    xs = np.where(piv, 1.0, x)
    ds = np.where(piv, 1.0, delta)
    # X p = L^-1 p; Y F = D^-1 U^-1 F.  Preserve the operation order
    # used by the report's vectorized implementation near basin boundaries.
    Xp1, Xp2 = p1, p2 - (2.0 * y / xs) * p1
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Yf1 = xs / (2.0 * ds) * f1 - y / (xs * ds) * f2
        Yf2 = f2 / (xs * xs)
    nx, ny = x + h * Xp1, y + h * Xp2
    np1, np2 = (1 - 2 * h) * p1 - h * Yf1, (1 - 2 * h) * p2 - h * Yf2
    return nx, ny, np1, np2, piv


def dist_sing(x, y):
    """Distancia a la unión singular: x=0 o delta=x^2-2y^2=0 (rectas x=±sqrt2 y)."""
    r = np.sqrt(2.0)
    return np.minimum(np.abs(x), np.minimum(np.abs(x - r * y), np.abs(x + r * y)) / np.sqrt(3.0))


def run(method, x0, y0, alpha=0.5, h=0.5, maxit=MAXIT, tol=TOL, p0=(0.0, 0.0)):
    """Ejecuta el método sobre arreglos 1-D de inicios. Devuelve (estado, iteraciones)."""
    x, y = np.array(x0, float), np.array(y0, float)
    p1, p2 = np.full_like(x, p0[0]), np.full_like(x, p0[1])
    n = x.size
    dmin = np.full(n, np.inf)
    state = np.full(n, -1)
    its = np.zeros(n, int)
    act = np.arange(n)
    for k in range(maxit + 1):
        if act.size == 0:
            break
        xa, ya = x[act], y[act]
        with np.errstate(all="ignore"):
            nf = Fnorm(xa, ya)
            dmin[act] = np.minimum(dmin[act], dist_sing(xa, ya))
        ok = nf < tol
        bad = ~np.isfinite(nf) | (np.abs(xa) > DIV_X) | (np.abs(ya) > DIV_X)
        state[act[ok]] = nearest_root(xa[ok], ya[ok])
        state[act[bad & ~ok]] = DIV
        its[act] = k
        keep = ~ok & ~bad
        if k == maxit:
            state[act[keep]] = EXH
            break
        act, xa, ya = act[keep], xa[keep], ya[keep]
        with np.errstate(all="ignore"):
            if method == "W4":
                nx, ny, np1, np2, piv = _w4_step(xa, ya, p1[act], p2[act], h)
                state[act[piv]] = PIV
                p1[act], p2[act] = np1, np2
                x[act], y[act] = nx, ny
                act = act[~piv]
            else:
                dx, dy, sing = _newton_dir(xa, ya)
                a = 1.0 if method == "NR" else alpha
                state[act[sing]] = SING
                x[act], y[act] = xa - a * dx, ya - a * dy
                act = act[~sing]
    return state, its, dmin


def grid(N, centros=False):
    if centros:
        e = -5 + 10 * (np.arange(N) + 0.5) / N
    else:
        e = np.linspace(-5, 5, N)
    X, Y = np.meshgrid(e, e)
    return X.ravel(), Y.ravel()


def resumen(state):
    n = state.size
    return {NOMBRES[i]: int((state == i).sum()) for i in range(8)} | {
        "exito_%": 100 * float((state < 4).sum()) / n}


def trayectoria(method, x0, y0, alpha=0.5, h=0.5, maxit=60, p0=(0.0, 0.0), tol=TOL):
    """Trayectoria de un solo punto (para la demo paso a paso)."""
    x, y, p1, p2 = np.array([x0], float), np.array([y0], float), np.array([p0[0]], float), np.array([p0[1]], float)
    pts = [(x0, y0)]
    for _ in range(maxit):
        with np.errstate(all="ignore"):
            if (Fnorm(x, y)[0] < tol or not np.isfinite(x[0] + y[0])
                    or abs(x[0]) > DIV_X or abs(y[0]) > DIV_X):
                break
            if method == "W4":
                x, y, p1, p2, piv = _w4_step(x, y, p1, p2, h)
                if piv[0]:
                    break
            else:
                dx, dy, s = _newton_dir(x, y)
                if s[0]:
                    break
                a = 1.0 if method == "NR" else alpha
                x, y = x - a * dx, y - a * dy
        pts.append((float(x[0]), float(y[0])))
    return np.array(pts)
