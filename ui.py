"""Shared Streamlit layout, report comparisons, caching, and Plotly helpers."""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = pathlib.Path(__file__).resolve().parent


def _project_root(app_dir: pathlib.Path) -> pathlib.Path:
    """Support both the nested ``app/`` layout and a flattened repo-root app."""
    if (app_dir / "data").is_dir() or (app_dir / "results").is_dir():
        return app_dir
    return app_dir.parent


PROJECT_DIR = _project_root(APP_DIR)
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core import (  # noqa: E402
    COLORES, DIV, EXH, Fnorm, MAXIT, NOMBRES, PIV, ROOTS, SING, TOL,
    grid, run, trayectoria,
)

REPORT_DEFAULTS = {
    "N": 400, "alpha": 0.5, "h": 0.5, "p0x": 0.0, "p0y": 0.0, "maxit": 1000,
}
SIDEBAR_DEFAULTS = {
    "global_n": REPORT_DEFAULTS["N"], "global_alpha": REPORT_DEFAULTS["alpha"],
    "global_h": REPORT_DEFAULTS["h"], "global_p0x": REPORT_DEFAULTS["p0x"],
    "global_p0y": REPORT_DEFAULTS["p0y"], "global_maxit": REPORT_DEFAULTS["maxit"],
    "presentation_mode": False,
}
REPORT_COVERAGE = {"NR": 70.945625, "DN": 70.489375, "W4": 99.65}
REPORT_COST_PER_UPDATE = {"NR": 0.045262899994850156, "DN": 0.04801649998407811,
                          "W4": 0.06685787998139858}
REPORT_FAILURES = {
    "NR": {"Divergencia": 15, "Agotó tope": 46472, "J singular": 0, "Pivote UDL": 0},
    "DN": {"Divergencia": 5, "Agotó tope": 47212, "J singular": 0, "Pivote UDL": 0},
    "W4": {"Divergencia": 548, "Agotó tope": 0, "J singular": 0, "Pivote UDL": 12},
}
STATUS_TEXT = ["Raíz R₁", "Raíz R₂", "Raíz R₃", "Raíz R₄", "Divergencia",
               "Agotó el tope", "Jacobiano singular", "Pivote UDL nulo"]
METHOD_COLORS = {"NR": "#3f74a8", "DN": "#d9775a", "W4": "#2a8a3a"}
PAGE_FILES = [
    "1_Cuencas_y_cobertura.py", "2_Superposicion_de_fallos.py", "3_Trayectorias.py",
    "4_Fallos_y_singularidades.py", "5_Sensibilidad.py", "6_Convergencia_local.py",
    "7_Costo.py",
]
PAGE_LABELS = [
    "Cuencas y cobertura", "Superposición de fallos", "Trayectorias",
    "Fallos y singularidades", "Sensibilidad", "Convergencia local", "Costo",
]


@dataclass(frozen=True)
class PageContext:
    params: dict
    results: dict | None
    presentation: bool


def inject_style(presentation: bool = False) -> None:
    css_file = APP_DIR / "assets" / "style.css"
    css = css_file.read_text(encoding="utf-8") if css_file.exists() else ""
    if presentation:
        css += "\n" + "\n".join([
            "html, body, [class*='st-'] { font-size: 1.08rem !important; }",
            "h1 { font-size: 2.55rem !important; }",
            "[data-testid='stMetricValue'] { font-size: 2.25rem !important; }",
        ])
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _set_report_defaults() -> None:
    st.session_state["global_n"] = REPORT_DEFAULTS["N"]
    st.session_state["global_alpha"] = REPORT_DEFAULTS["alpha"]
    st.session_state["global_h"] = REPORT_DEFAULTS["h"]
    st.session_state["global_p0x"] = REPORT_DEFAULTS["p0x"]
    st.session_state["global_p0y"] = REPORT_DEFAULTS["p0y"]
    st.session_state["global_maxit"] = REPORT_DEFAULTS["maxit"]
    for key, value in current_params().items():
        state_key = f"_saved_{key}"
        st.session_state[state_key] = value
    st.session_state["calc_signature"] = None


def _parameter_signature(params: dict) -> tuple:
    return tuple(params[k] for k in ("N", "alpha", "h", "p0x", "p0y", "maxit"))


def current_params() -> dict:
    for key, default in SIDEBAR_DEFAULTS.items():
        if key == "presentation_mode":
            continue
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"_saved_{key}", default)
    params = {
        "N": int(st.session_state["global_n"]),
        "alpha": float(st.session_state["global_alpha"]),
        "h": float(st.session_state["global_h"]),
        "p0x": float(st.session_state["global_p0x"]),
        "p0y": float(st.session_state["global_p0y"]),
        "maxit": int(st.session_state["global_maxit"]),
    }
    for key, value in params.items():
        st.session_state[f"_saved_{key}"] = value
    return params


@st.cache_data(show_spinner=False, max_entries=24)
def _simulate_one(method: str, n: int, alpha: float, h: float, p0x: float,
                  p0y: float, maxit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = grid(n)
    return run(method, x, y, alpha=alpha, h=h, maxit=maxit, p0=(p0x, p0y))


@st.cache_data(show_spinner="Calculando la malla y las cuencas…", max_entries=8)
def _simulate_all(n: int, alpha: float, h: float, p0x: float,
                  p0y: float, maxit: int) -> dict:
    return {
        method: _simulate_one(method, n, alpha, h, p0x, p0y, maxit)
        for method in ("NR", "DN", "W4")
    }


@st.cache_data(show_spinner=False)
def _load_baseline(mtime_ns: int) -> dict:
    path = PROJECT_DIR / "data" / "baseline_400.npz"
    with np.load(path, allow_pickle=False) as raw:
        return {
            method: (
                raw[f"state_{method}"].copy(), raw[f"iterations_{method}"].copy(),
                raw[f"dmin_{method}"].copy(),
            )
            for method in ("NR", "DN", "W4")
        }


def baseline_available() -> bool:
    return (PROJECT_DIR / "data" / "baseline_400.npz").is_file()


def common_sidebar() -> tuple[dict, bool]:
    for key, value in SIDEBAR_DEFAULTS.items():
        st.session_state.setdefault(key, st.session_state.get(f"_saved_{key}", value))
    st.session_state.setdefault("calc_signature", None)

    with st.sidebar:
        st.header("Parámetros globales")
        st.selectbox("Malla N × N", (100, 200, 400), key="global_n")
        st.slider("Amortiguamiento α (DN)", 0.1, 0.9, step=0.05, key="global_alpha")
        st.slider("Paso Δτ (W4)", 0.1, 0.9, step=0.05, key="global_h")
        pcols = st.columns(2)
        pcols[0].number_input("p₀ x", -1.0, 1.0, step=0.05, key="global_p0x")
        pcols[1].number_input("p₀ y", -1.0, 1.0, step=0.05, key="global_p0y")
        st.number_input("Tope de iteraciones", 10, 2000, step=10, key="global_maxit")
        st.toggle("Vista de proyector", key="presentation_mode")
        st.button("Restablecer valores del informe", on_click=_set_report_defaults,
                  width="stretch")

    params = current_params()
    st.session_state["_saved_presentation_mode"] = bool(st.session_state["presentation_mode"])
    is_reference = params == REPORT_DEFAULTS
    st.sidebar.markdown("---")
    if is_reference and baseline_available():
        st.sidebar.success("Configuración idéntica a la malla del informe (400 × 400).")
    else:
        st.sidebar.info("Comparación visible con los valores del informe.")

    return params, bool(st.session_state.presentation_mode)


def _ensure_sidebar_state() -> None:
    """Restore the shared controls when Streamlit runs a page without its entrypoint."""
    if any(key not in st.session_state for key in SIDEBAR_DEFAULTS):
        common_sidebar()


def page_context(title: str, slide: int | str, require_results: bool = False) -> PageContext:
    _ensure_sidebar_state()
    params = current_params()
    presentation = bool(st.session_state.get("presentation_mode", False))
    st.title(title)
    if not presentation:
        st.caption("Exploración de los resultados, trayectorias y diagnósticos del experimento del informe.")

    signature = _parameter_signature(params)
    report_config = params == REPORT_DEFAULTS
    results = None
    if require_results:
        if params["N"] == 400 and not (report_config and baseline_available()):
            with st.sidebar:
                calculate = st.button("Calcular malla 400 × 400", type="primary", width="stretch")
            if calculate:
                st.session_state["calc_signature"] = signature
            if st.session_state.get("calc_signature") != signature:
                st.info("La malla 400 × 400 requiere un cálculo explícito. Ajusta los parámetros y pulsa «Calcular malla 400 × 400» en la barra lateral.")
            else:
                results = _simulate_all(
                    params["N"], params["alpha"], params["h"], params["p0x"],
                    params["p0y"], params["maxit"],
                )
        elif params["N"] == 400 and report_config:
            mtime_ns = (PROJECT_DIR / "data" / "baseline_400.npz").stat().st_mtime_ns
            results = _load_baseline(mtime_ns)
        else:
            with st.spinner("Calculando las tres cuencas…"):
                results = _simulate_all(
                    params["N"], params["alpha"], params["h"], params["p0x"],
                    params["p0y"], params["maxit"],
                )
    elif params["N"] == 400 and report_config and baseline_available():
        mtime_ns = (PROJECT_DIR / "data" / "baseline_400.npz").stat().st_mtime_ns
        results = _load_baseline(mtime_ns)

    return PageContext(params=params, results=results, presentation=presentation)


def home_context() -> tuple[dict, bool]:
    _ensure_sidebar_state()
    params = current_params()
    presentation = bool(st.session_state.get("presentation_mode", False))
    st.session_state["_saved_presentation_mode"] = presentation
    return params, presentation


def metric_report(label: str, value: float, reference: float, fmt: str = ".2f",
                  suffix: str = "%", unit: str = "p.p.") -> None:
    difference = value - reference
    st.metric(label, f"{value:{fmt}}{suffix}", f"{difference:+.{2 if fmt == '.2f' else 3}f} {unit}")
    st.caption(f"Informe: {reference:{fmt}}{suffix}")


def show_insight(what: str, means: str) -> None:
    if not st.session_state.get("presentation_mode", False):
        st.markdown(f"**Qué se ve.** {what}")
        st.markdown(f"**Qué significa y qué no.** {means}")


def show_what(text: str) -> None:
    if not st.session_state.get("presentation_mode", False):
        st.markdown(f"**Qué se ve.** {text}")


def show_meaning(text: str) -> None:
    if not st.session_state.get("presentation_mode", False):
        st.markdown(f"**Qué significa y qué no.** {text}")


def _state_hover(states: np.ndarray, iterations: np.ndarray) -> np.ndarray:
    labels = np.asarray(STATUS_TEXT, dtype=object)
    state_text = labels[states]
    return np.stack((state_text, iterations), axis=-1)


def basin_figure(method: str, result: tuple, n: int, *, title: str | None = None,
                 overlay: np.ndarray | None = None) -> go.Figure:
    states, iterations, _ = result
    states = np.asarray(states).reshape(n, n)
    iterations = np.asarray(iterations).reshape(n, n)
    z = np.where(states < 4, states, 4)
    axis = np.linspace(-5.0, 5.0, n)
    colors = ["#ef6a7d", "#3f74a8", "#cdbb44", "#2a8a3a", "#222222"]
    colorscale = []
    for i, color in enumerate(colors):
        lo, hi = i / 5.0, (i + 1) / 5.0
        colorscale.extend([[lo, color], [hi - 1e-7, color]])
    custom = _state_hover(states, iterations)
    fig = go.Figure(go.Heatmap(
        x=axis, y=axis, z=z, zmin=0, zmax=5, colorscale=colorscale,
        customdata=custom, showscale=True,
        colorbar={"title": "Estado", "tickvals": [0.5, 1.5, 2.5, 3.5, 4.5],
                  "ticktext": ["R₁", "R₂", "R₃", "R₄", "Fallo"]},
        hovertemplate="x₀=%{x:.4f}<br>y₀=%{y:.4f}<br>%{customdata[0]}<br>Iteraciones: %{customdata[1]}<extra></extra>",
        name="Raíz o fallo",
    ))
    if overlay is not None:
        ys, xs = np.where(overlay.reshape(n, n))
        fig.add_trace(go.Scattergl(
            x=axis[xs], y=axis[ys], mode="markers", name="Fallos del método",
            marker={"color": "#111111", "size": 3, "symbol": "x"},
            hovertemplate="Fallo en (%{x:.4f}, %{y:.4f})<extra></extra>",
        ))
    fig.update_layout(
        title=title or f"Cuencas de {method}", height=560, margin={"l": 50, "r": 20, "t": 60, "b": 50},
        xaxis_title="x inicial", yaxis_title="y inicial", font={"size": 15},
        legend={"orientation": "h", "y": -0.16},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def trajectory_figure(paths: dict[str, np.ndarray], frame: int | None = None) -> go.Figure:
    fig = go.Figure()
    for method, points in paths.items():
        count = len(points) if frame is None else min(len(points), frame + 1)
        visible = points[:count]
        fig.add_trace(go.Scatter(
            x=visible[:, 0], y=visible[:, 1], mode="lines+markers", name=method,
            line={"color": METHOD_COLORS[method], "width": 2},
            marker={"size": 7},
            hovertemplate=f"{method}<br>x=%{{x:.6g}}<br>y=%{{y:.6g}}<extra></extra>",
        ))
        if len(visible):
            fig.add_trace(go.Scatter(
                x=[visible[-1, 0]], y=[visible[-1, 1]], mode="markers",
                name=f"{method}: paso actual", showlegend=False,
                marker={"symbol": "diamond", "size": 12, "color": METHOD_COLORS[method]},
            ))
    fig.update_layout(title="Trayectoria simultánea en el plano (x, y)", height=500,
                      xaxis_title="x", yaxis_title="y", font={"size": 14})
    return fig


def residual_figure(paths: dict[str, np.ndarray], frame: int | None = None) -> go.Figure:
    fig = go.Figure()
    for method, points in paths.items():
        count = len(points) if frame is None else min(len(points), frame + 1)
        residual = Fnorm(points[:count, 0], points[:count, 1])
        fig.add_trace(go.Scatter(
            x=np.arange(count), y=np.maximum(residual, 1e-18), mode="lines+markers",
            name=method, line={"color": METHOD_COLORS[method], "width": 2},
            hovertemplate=f"{method}<br>Iteración %{{x}}<br>‖F(x)‖₂=%{{y:.3e}}<extra></extra>",
        ))
    fig.add_hline(y=TOL, line_dash="dash", line_color="#222222",
                  annotation_text="Tolerancia 10⁻⁶")
    fig.update_yaxes(type="log", title="Residuo ‖F(xₙ)‖₂")
    fig.update_xaxes(title="Iteración")
    fig.update_layout(title="Residuo por paso", height=400, font={"size": 14})
    return fig


def download_figure_and_table(fig: go.Figure, dataframe: pd.DataFrame, stem: str) -> None:
    c1, c2 = st.columns(2)
    c2.download_button(f"Descargar tabla CSV · {stem}", dataframe.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{stem}.csv", mime="text/csv", width="stretch")
    try:
        png = fig.to_image(format="png", scale=2)
        c1.download_button(f"Descargar figura PNG · {stem}", png, file_name=f"{stem}.png",
                           mime="image/png", width="stretch")
    except Exception:
        c1.caption("La exportación PNG necesita Kaleido y Chrome disponibles en el entorno.")


def add_navigation(current: int) -> None:
    st.markdown("---")
    cols = st.columns(3)
    previous = current - 1
    following = current + 1
    if previous >= 0:
        cols[0].page_link(f"pages/{PAGE_FILES[previous]}", label=f"← Anterior · {PAGE_LABELS[previous]}")
    else:
        cols[0].page_link("pages/0_Inicio.py", label="← Inicio")
    cols[1].page_link("pages/0_Inicio.py", label="Análisis del proyecto")
    if following < len(PAGE_FILES):
        cols[2].page_link(f"pages/{PAGE_FILES[following]}", label=f"Siguiente · {PAGE_LABELS[following]} →")
    else:
        cols[2].page_link("pages/0_Inicio.py", label="Inicio →")


def empty_results_message() -> None:
    st.info("Todavía no hay resultados para esta configuración. Para N=400, calcula la malla desde la barra lateral.")
