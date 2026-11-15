import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import (
    METHOD_COLORS, NOMBRES, PAGE_FILES, REPORT_COVERAGE, add_navigation,
    basin_figure, download_figure_and_table, empty_results_message,
    metric_report, page_context, show_meaning, show_what,
)


ctx = page_context("Cuencas de convergencia y cobertura", 9, require_results=True)
if ctx.results is None:
    empty_results_message()
    add_navigation(0)
    st.stop()

p = ctx.params
results = ctx.results
show_what("Cada celda representa un inicio y el color indica a qué raíz llega; negro agrupa los estados terminales fallidos.")

metric_cols = st.columns(3)
for col, method in zip(metric_cols, ("W4", "NR", "DN")):
    states = results[method][0]
    current = 100.0 * np.count_nonzero(states < 4) / states.size
    with col:
        metric_report(f"Cobertura {method}", current, REPORT_COVERAGE[method])

view = st.radio("Comparación", ["Tres métodos", "Antes / Después · NR y W4"],
                horizontal=True, key="basin_view")
method_options = ["NR", "DN", "W4"]
if view.startswith("Antes"):
    visible_methods = ["NR", "W4"]
else:
    visible_methods = st.multiselect("Métodos visibles", method_options, default=method_options,
                                     key="basin_methods")

table_rows = []
for method in method_options:
    states = results[method][0]
    table_rows.append({
        "Método": method,
        **{f"R{i + 1} (%)": 100.0 * np.count_nonzero(states == i) / states.size for i in range(4)},
        "Éxito (%)": 100.0 * np.count_nonzero(states < 4) / states.size,
        "Divergencias": int(np.count_nonzero(states == 4)),
        "Agotó tope": int(np.count_nonzero(states == 5)),
        "J singular": int(np.count_nonzero(states == 6)),
        "Pivote UDL": int(np.count_nonzero(states == 7)),
    })
summary = pd.DataFrame(table_rows).set_index("Método")

chart_cols = st.columns(len(visible_methods))
for col, method in zip(chart_cols, visible_methods):
    with col:
        fig = basin_figure(method, results[method], p["N"],
                           title=f"{method} · {100 * (results[method][0] < 4).mean():.2f}% converge")
        event = st.plotly_chart(
            fig, key=f"basin_{method}", width="stretch",
            on_select="rerun", selection_mode=("points", "box", "lasso"),
            alt=f"Mapa interactivo de cuencas de {method}; el color identifica la raíz final y el hover muestra el estado y las iteraciones.",
            config={"scrollZoom": True, "displaylogo": False},
        )
        selection = getattr(getattr(event, "selection", None), "points", [])
        if selection:
            point = selection[-1]
            if "x" in point and "y" in point:
                st.session_state["basin_initial_x"] = float(point["x"])
                st.session_state["basin_initial_y"] = float(point["y"])

st.markdown("#### Selecciona un inicio para seguir los tres métodos")
axis_step = 10.0 / max(p["N"] - 1, 1)
if "basin_initial_x" not in st.session_state:
    st.session_state["basin_initial_x"] = 0.0
if "basin_initial_y" not in st.session_state:
    st.session_state["basin_initial_y"] = 0.0
select_cols = st.columns(2)
x0 = select_cols[0].slider("x₀", -5.0, 5.0, step=axis_step, key="basin_initial_x")
y0 = select_cols[1].slider("y₀", -5.0, 5.0, step=axis_step, key="basin_initial_y")
st.caption(f"Inicio seleccionado: ({x0:.5f}, {y0:.5f}). También puedes usar el selector de punto del mapa.")
st.page_link("pages/3_Trayectorias.py", label="Ver su trayectoria en NR, DN y W4 →")

st.markdown("#### Cobertura y tipos terminales")
st.dataframe(summary.style.format("{:.4f}"), width="stretch")
download_figure_and_table(basin_figure("W4", results["W4"], p["N"]), summary.reset_index(),
                          "cobertura_cuencas")

show_meaning("La comparación usa los mismos puntos iniciales, la misma función y el mismo criterio de parada. Los porcentajes dependen de la región, resolución, paso y momento inicial; no se extrapolan a todo sistema no lineal.")
add_navigation(0)
