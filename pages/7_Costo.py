import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import (
    REPORT_COST_PER_UPDATE, add_navigation, download_figure_and_table,
    empty_results_message, metric_report, page_context, show_meaning, show_what,
)


ctx = page_context("Costo computacional por actualización y por raíz", 16,
                   require_results=True)
if ctx.results is None:
    empty_results_message()
    add_navigation(6)
    st.stop()

results = ctx.results
methods = ("NR", "DN", "W4")
common = np.logical_and.reduce([results[method][0] < 4 for method in methods])
common_count = int(common.sum())
rows = []
for method in methods:
    states, iterations, _ = results[method]
    successes = int(np.count_nonzero(states < 4))
    common_mean = float(np.mean(iterations[common])) if common_count else np.nan
    updates = int(np.sum(iterations, dtype=np.int64))
    unit = REPORT_COST_PER_UPDATE[method]
    cost = unit * updates / successes if successes else np.nan
    rows.append({
        "Método": method,
        "Iteraciones medias · éxitos comunes": common_mean,
        "µs por actualización · informe": unit,
        "µs estimados por raíz": cost,
        "Informe · µs por raíz": {"NR": 18.803857793217926,
                                  "DN": 21.33806035950454,
                                  "W4": 3.38217887417064}[method],
        "Actualizaciones totales": updates,
        "Éxitos": successes,
        "Agotó tope": int(np.count_nonzero(states == 5)),
    })
cost_table = pd.DataFrame(rows).set_index("Método")

show_what("El costo amortizado multiplica las actualizaciones de los intentos por el tiempo mediano por actualización y divide por las raíces encontradas.")

metric_cols = st.columns(3)
for col, method in zip(metric_cols, methods):
    row = cost_table.loc[method]
    with col:
        metric_report(f"Costo estimado por raíz · {method}",
                      float(row["µs estimados por raíz"]),
                      float(row["Informe · µs por raíz"]),
                      fmt=".3f", suffix=" µs", unit="µs")
st.metric("Inicios resueltos por los tres métodos", f"{common_count:,}".replace(",", " "),
          f"vs informe {common_count - 106405:+d}")
st.caption("Informe: 106 405 inicios comunes.")

choice = st.radio("Comparación", ["Los tres métodos", "Antes / Después · NR y W4"],
                  horizontal=True, key="cost_comparison")
visible = methods if choice.startswith("Los") else ("NR", "W4")
fig = go.Figure()
for method in visible:
    row = cost_table.loc[method]
    fig.add_trace(go.Bar(
        name=method, x=["Por actualización", "Por raíz"],
        y=[row["µs por actualización · informe"], row["µs estimados por raíz"]],
        marker_color={"NR": "#3f74a8", "DN": "#d9775a", "W4": "#2a8a3a"}[method],
        text=[f"{row['µs por actualización · informe']:.4f}",
              f"{row['µs estimados por raíz']:.3f}"],
        textposition="outside",
        hovertemplate=f"{method}<br>%{{x}}<br>%{{y:.4f}} µs<extra></extra>",
    ))
fig.update_layout(
    title="W4 requiere más por actualización y menos por raíz en esta malla",
    barmode="group", height=470, yaxis_type="log",
    yaxis_title="Microsegundos (escala logarítmica)", font={"size": 14},
)
st.plotly_chart(fig, width="stretch", alt="Comparación del costo estimado por actualización y por raíz, en escala logarítmica.", config={"displaylogo": False})
st.caption("La escala logarítmica permite comparar medidas de distinto orden; el costo por raíz incluye los intentos fallidos.")
st.dataframe(cost_table.style.format({
    "Iteraciones medias · éxitos comunes": "{:.2f}",
    "µs por actualización · informe": "{:.4f}",
    "µs estimados por raíz": "{:.3f}",
    "Informe · µs por raíz": "{:.3f}",
}), width="stretch")

st.latex(r"C_m = c_m \frac{\sum_i k_i}{S_m}")
download_figure_and_table(fig, cost_table.reset_index(), "costo_computacional")

show_meaning(f"Los tres métodos coinciden en {common_count:,} éxitos; allí se comparan iteraciones medias, y para el costo amortizado también cuentan los fallos. Los tiempos dependen del equipo y protocolo.".replace(",", " "))
add_navigation(6)
