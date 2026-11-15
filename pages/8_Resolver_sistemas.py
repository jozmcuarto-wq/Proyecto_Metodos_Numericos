"""Comparación guiada de NR, Newton amortiguado y W4."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import Fnorm, NOMBRES, run, trayectoria
from general_solver import compare_methods, compile_system


TOL = 1e-6
MAXIT = 1000
ALPHA = 0.5
H = 0.5
P0 = (0.0, 0.0)

SYSTEMS = {
    "Proyecto del informe": {
        "f1": "x^2 + y^2 - 4",
        "f2": "x^2*y - 1",
        "start": (1.5, 1.0),
        "report": True,
    },
    "Ejemplo polinómico": {
        "f1": "x^2 + y^2 - 1",
        "f2": "x - y",
        "start": (0.8, 0.6),
        "report": False,
    },
    "Ejemplo trigonométrico": {
        "f1": "sin(x) + y - 1",
        "f2": "x + sin(y) - 1",
        "start": (0.5, 0.5),
        "report": False,
    },
}

REPORT_STARTS = {
    "P1 del informe": (2.66917293, -2.56892231),
    "P2 del informe": (-2.44360902, -5.0),
    "P3 del informe": (4.47368421, -3.17042607),
    "Escribir otro punto": (0.0, 0.0),
}


st.title("Comparar métodos")
st.write("Elige un sistema y un punto inicial. Los tres métodos empiezan desde el mismo punto.")
st.page_link("pages/0_Inicio.py", label="Ver resultados principales del informe", icon="📊")

system_name = st.selectbox("1. Sistema", list(SYSTEMS), key="simple_system")
system = SYSTEMS[system_name]

if system["report"]:
    point_name = st.selectbox("2. Punto inicial", list(REPORT_STARTS), key="simple_project_point")
    default_x, default_y = REPORT_STARTS[point_name]
else:
    point_name = "Punto de ejemplo"
    default_x, default_y = system["start"]

x_key = f"simple_x_{system_name}_{point_name}"
y_key = f"simple_y_{system_name}_{point_name}"
st.session_state.setdefault(x_key, float(default_x))
st.session_state.setdefault(y_key, float(default_y))

coord_cols = st.columns(2)
x0 = coord_cols[0].number_input(
    "x₀", min_value=-5.0, max_value=5.0, step=0.00000001,
    format="%.8f", key=x_key,
)
y0 = coord_cols[1].number_input(
    "y₀", min_value=-5.0, max_value=5.0, step=0.00000001,
    format="%.8f", key=y_key,
)

st.caption(f"Sistema: F₁ = {system['f1']}  ·  F₂ = {system['f2']}")
st.caption("Configuración común: tolerancia 10⁻⁶ · máximo 1000 iteraciones · α = 0.5 · h = 0.5 · p₀ = (0, 0).")


def project_comparison(x_start: float, y_start: float):
    """Use the validated, specialized implementation from the report."""
    rows = []
    paths = {}
    for method in ("NR", "DN", "W4"):
        states, iterations, _ = run(
            method,
            np.array([x_start]),
            np.array([y_start]),
            alpha=ALPHA,
            h=H,
            maxit=MAXIT,
            tol=TOL,
            p0=P0,
        )
        path = trayectoria(
            method, x_start, y_start, alpha=ALPHA, h=H,
            maxit=MAXIT, p0=P0, tol=TOL,
        )
        paths[method] = path
        state = int(states[0])
        rows.append({
            "Método": method,
            "Resultado": f"Convergió ({NOMBRES[state]})" if state < 4 else NOMBRES[state],
            "Iteraciones": int(iterations[0]),
            "x final": float(path[-1, 0]),
            "y final": float(path[-1, 1]),
            "Residuo final": float(Fnorm(path[-1:, 0], path[-1:, 1])[0]),
        })
    return rows, paths


if system["report"]:
    rows, paths = project_comparison(float(x0), float(y0))
    st.info("Este es el sistema del informe y usa su implementación analítica original.")
else:
    try:
        results = compare_methods(
            system["f1"], system["f2"], float(x0), float(y0),
            alpha=ALPHA, h=H, p0=P0, tolerance=TOL,
            max_iterations=MAXIT,
        )
        rows = [{
            "Método": result.method,
            "Resultado": result.status,
            "Iteraciones": result.iterations,
            "x final": result.x,
            "y final": result.y,
            "Residuo final": result.residual,
        } for result in results]
        paths = {result.method: result.path for result in results}
        st.info("Ejemplo adicional: usa Jacobiano por diferencias finitas; sus resultados no son cifras del informe.")
    except (ValueError, ArithmeticError, FloatingPointError) as error:
        st.error(f"No se pudo resolver este ejemplo: {error}")
        st.stop()


st.subheader("3. Comparación")
st.caption("«Convergió» significa que el residuo quedó por debajo de 10⁻⁶. R1–R4 son las cuatro raíces etiquetadas en el informe.")
table = pd.DataFrame(rows)
st.dataframe(
    table.style.format({"x final": "{:.8f}", "y final": "{:.8f}", "Residuo final": "{:.3e}"}),
    hide_index=True,
    width="stretch",
)

fig = go.Figure()
for method in ("NR", "DN", "W4"):
    path = paths[method]
    if system["report"]:
        with np.errstate(all="ignore"):
            residuals = Fnorm(path[:, 0], path[:, 1])
    else:
        functions = compile_system(system["f1"], system["f2"])
        residuals = np.array([
            np.linalg.norm([function(float(point[0]), float(point[1])) for function in functions])
            for point in path
        ])
    residuals = np.asarray(residuals, dtype=float)
    residuals[~np.isfinite(residuals) | (residuals <= 0.0)] = np.nan
    fig.add_trace(go.Scatter(
        x=np.arange(len(residuals)), y=residuals, mode="lines+markers",
        name=method, marker={"size": 5},
    ))
fig.add_hline(y=TOL, line_dash="dash", line_color="#555555", annotation_text="tolerancia 10⁻⁶")
fig.update_layout(
    title="¿Qué tan cerca está cada paso de una solución?",
    xaxis_title="Iteración",
    yaxis_title="Residuo ‖F(x, y)‖ (menor es mejor)",
    yaxis_type="log",
    legend_title="Método",
    margin={"l": 20, "r": 20, "t": 65, "b": 20},
)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
st.caption("Cuando una curva queda debajo de la línea punteada, el residuo está dentro de la tolerancia.")

with st.expander("¿Qué significa cada método?"):
    st.markdown(
        "- **NR:** Newton-Raphson, usa el paso completo.\n"
        "- **DN:** Newton amortiguado, usa la mitad del paso (α = 0.5).\n"
        "- **W4:** método W4, actualiza también el vector auxiliar p."
    )
