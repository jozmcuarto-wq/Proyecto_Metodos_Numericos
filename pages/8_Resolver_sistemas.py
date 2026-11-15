import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from general_solver import ExpressionError, compare_methods, compile_system


PRESETS = {
    "Sistema del proyecto": {
        "key": "proyecto",
        "f1": "x^2 + y^2 - 4",
        "f2": "x^2*y - 1",
        "x0": 1.5,
        "y0": 1.0,
    },
    "Sistema polinómico · círculo y recta": {
        "key": "circulo",
        "f1": "x^2 + y^2 - 1",
        "f2": "x - y",
        "x0": 0.8,
        "y0": 0.6,
    },
    "Sistema trigonométrico": {
        "key": "trigonometrico",
        "f1": "sin(x) + y - 1",
        "f2": "x + sin(y) - 1",
        "x0": 0.5,
        "y0": 0.5,
    },
    "Escribir mi propio sistema": {
        "key": "personalizado",
        "f1": "x^2 + y^2 - 1",
        "f2": "x - y",
        "x0": 0.8,
        "y0": 0.6,
    },
}


st.title("Resolver y comparar sistemas no lineales")
st.write(
    "Define dos ecuaciones en **x** e **y** y ejecuta NR, Newton amortiguado (DN) "
    "y W4 desde el mismo punto inicial. Puedes usar los ejemplos o escribir otro sistema."
)

preset_name = st.selectbox("Sistema de ejemplo", list(PRESETS), key="general_solver_preset")
preset = PRESETS[preset_name]

with st.form("general_solver_form"):
    st.markdown("#### Define el sistema F(x, y) = 0")
    expression_cols = st.columns(2)
    f1_expression = expression_cols[0].text_input(
        "F₁(x, y)", value=preset["f1"], key=f"general_f1_{preset['key']}",
    )
    f2_expression = expression_cols[1].text_input(
        "F₂(x, y)", value=preset["f2"], key=f"general_f2_{preset['key']}",
    )
    st.caption(
        "Variables: x, y. Operaciones: +, -, *, /, ^ o **. Funciones: sin, cos, tan, "
        "exp, log, sqrt, abs, sinh, cosh, tanh, asin/arcsin, acos/arccos, atan/arctan; constantes pi y e."
    )

    initial_cols = st.columns(2)
    x0 = initial_cols[0].number_input("x₀", value=preset["x0"], step=0.1, key=f"general_x0_{preset['key']}")
    y0 = initial_cols[1].number_input("y₀", value=preset["y0"], step=0.1, key=f"general_y0_{preset['key']}")

    with st.expander("Parámetros de comparación"):
        parameter_cols = st.columns(4)
        alpha = parameter_cols[0].slider("Amortiguamiento α (DN)", 0.05, 1.0, 0.5, 0.05)
        h = parameter_cols[1].slider("Paso Δτ (W4)", 0.05, 0.95, 0.5, 0.05)
        tolerance = parameter_cols[2].select_slider(
            "Tolerancia ‖F‖₂", options=[1e-4, 1e-6, 1e-8, 1e-10], value=1e-6,
            format_func=lambda value: f"{value:.0e}",
        )
        max_iterations = parameter_cols[3].number_input(
            "Máximo de iteraciones", min_value=1, max_value=1000, value=100, step=10,
        )
        p_cols = st.columns(2)
        p0x = p_cols[0].number_input("p₀ x (W4)", value=0.0, step=0.1)
        p0y = p_cols[1].number_input("p₀ y (W4)", value=0.0, step=0.1)

    submitted = st.form_submit_button("Comparar NR, DN y W4", type="primary", width="stretch")

signature = (
    f1_expression.strip(), f2_expression.strip(), float(x0), float(y0), float(alpha),
    float(h), float(tolerance), int(max_iterations), float(p0x), float(p0y),
)

if submitted:
    try:
        results = compare_methods(
            f1_expression, f2_expression, x0, y0, alpha=alpha, h=h,
            p0=(p0x, p0y), tolerance=tolerance, max_iterations=int(max_iterations),
        )
        st.session_state["general_solver_output"] = {
            "signature": signature,
            "results": [
                {
                    "method": result.method,
                    "status": result.status,
                    "iterations": result.iterations,
                    "x": result.x,
                    "y": result.y,
                    "residual": result.residual,
                    "path": result.path.tolist(),
                }
                for result in results
            ],
        }
    except (ExpressionError, ValueError, OverflowError, ZeroDivisionError) as error:
        st.session_state.pop("general_solver_output", None)
        st.error(str(error))

output = st.session_state.get("general_solver_output")
if output and output["signature"] != signature:
    st.info("Cambiaste el sistema o sus parámetros. Pulsa «Comparar NR, DN y W4» para actualizar los resultados.")
elif output:
    rows = [
        {
            "Método": item["method"],
            "Estado": item["status"],
            "Iteraciones": item["iterations"],
            "x final": item["x"],
            "y final": item["y"],
            "Residuo ‖F(x)‖₂": item["residual"],
        }
        for item in output["results"]
    ]
    table = pd.DataFrame(rows)
    converged = sum(item["status"] == "Convergió" for item in output["results"])
    metric_cols = st.columns(3)
    metric_cols[0].metric("Métodos que convergieron", f"{converged}/3")
    metric_cols[1].metric("Tolerancia residual", f"{tolerance:.0e}")
    metric_cols[2].metric("Inicio compartido", f"({x0:g}, {y0:g})")
    st.markdown("#### Resultado de la comparación")
    st.dataframe(
        table.style.format({"x final": "{:.8g}", "y final": "{:.8g}", "Residuo ‖F(x)‖₂": "{:.3e}"}),
        width="stretch", hide_index=True,
    )

    view_limit = st.slider(
        "Extensión visible del plano para ecuaciones y trayectorias",
        min_value=2.0, max_value=20.0, value=5.0, step=1.0,
    )
    axis = np.linspace(-view_limit, view_limit, 121)
    xx, yy = np.meshgrid(axis, axis)
    try:
        functions = compile_system(f1_expression, f2_expression)
        f1_values = np.asarray(functions[0](xx, yy), dtype=float)
        f2_values = np.asarray(functions[1](xx, yy), dtype=float)
        f1_values[~np.isfinite(f1_values)] = np.nan
        f2_values[~np.isfinite(f2_values)] = np.nan

        fig = go.Figure()
        for name, values, color in (
            ("F₁ = 0", f1_values, "#8795a5"),
            ("F₂ = 0", f2_values, "#d4a33c"),
        ):
            fig.add_trace(go.Contour(
                x=axis, y=axis, z=values,
                contours={"type": "constraint", "operation": "=", "value": 0,
                          "coloring": "none"},
                fillcolor="rgba(255,255,255,0)",
                line={"color": color, "width": 1.5}, showscale=False,
                name=name, hoverinfo="skip",
            ))
        for item in output["results"]:
            path = np.asarray(item["path"], dtype=float)
            in_view = np.all(np.abs(path) <= view_limit, axis=1)
            plotted = np.where(in_view[:, None], path, np.nan)
            fig.add_trace(go.Scatter(
                x=plotted[:, 0], y=plotted[:, 1], mode="lines+markers",
                name=f"{item['method']} · {item['status']}",
                line={"color": {"NR": "#3f74a8", "DN": "#d9775a", "W4": "#2a8a3a"}[item["method"]], "width": 2},
                marker={"size": 6},
                hovertemplate=f"{item['method']}<br>x=%{{x:.6g}}<br>y=%{{y:.6g}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0], mode="markers", name="Inicio compartido",
            marker={"symbol": "x", "size": 12, "color": "#16314d"},
        ))
        fig.update_layout(
            title="Curvas F₁ = 0 y F₂ = 0 con las trayectorias de los métodos",
            xaxis_title="x", yaxis_title="y", height=620,
            xaxis={"range": [-view_limit, view_limit]},
            yaxis={"range": [-view_limit, view_limit], "scaleanchor": "x", "scaleratio": 1},
            font={"size": 14}, legend={"orientation": "h", "y": -0.18},
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    except (ExpressionError, ValueError, TypeError) as error:
        st.warning(f"No se pudieron dibujar las curvas para este sistema: {error}")

    path_rows = []
    for item in output["results"]:
        for step, (px, py) in enumerate(item["path"]):
            path_rows.append({"Método": item["method"], "Iteración": step, "x": px, "y": py})
    downloads = st.columns(2)
    downloads[0].download_button(
        "Descargar comparación CSV", table.to_csv(index=False).encode("utf-8-sig"),
        file_name="comparacion_sistema_no_lineal.csv", mime="text/csv", width="stretch",
    )
    downloads[1].download_button(
        "Descargar trayectorias CSV", pd.DataFrame(path_rows).to_csv(index=False).encode("utf-8-sig"),
        file_name="trayectorias_sistema_no_lineal.csv", mime="text/csv", width="stretch",
    )

    st.caption(
        "Los tres métodos usan el mismo sistema, inicio y tolerancia. En esta pantalla el Jacobiano se "
        "aproxima con diferencias finitas centradas; por eso se comparan iteraciones y residuos, no "
        "tiempos frente a la implementación vectorizada y analítica del experimento del informe."
    )
