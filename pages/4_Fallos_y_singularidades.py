import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import (
    add_navigation, download_figure_and_table, empty_results_message,
    page_context, show_meaning, show_what,
)


ctx = page_context("Fallos y proximidad a las curvas singulares", 12,
                   require_results=True)
if ctx.results is None:
    empty_results_message()
    add_navigation(3)
    st.stop()

rows = []
for method in ("NR", "DN", "W4"):
    states = ctx.results[method][0]
    rows.append({
        "Método": method,
        "Divergencia": int(np.count_nonzero(states == 4)),
        "Agotó tope": int(np.count_nonzero(states == 5)),
        "J singular": int(np.count_nonzero(states == 6)),
        "Pivote UDL nulo": int(np.count_nonzero(states == 7)),
    })
failure_table = pd.DataFrame(rows).set_index("Método")

metric_cols = st.columns(3)
for col, label, state, report in (
    (metric_cols[0], "Divergencias W4", 4, 548),
    (metric_cols[1], "Pivotes UDL nulos", 7, 12),
    (metric_cols[2], "Agotamientos W4", 5, 0),
):
    value = int(np.count_nonzero(ctx.results["W4"][0] == state))
    with col:
        st.metric(label, f"{value:,}".replace(",", " "), f"vs informe {value - report:+d}")
        st.caption(f"Informe: {report:,}".replace(",", " "))

show_what("La tabla separa divergencia, agotamiento, Jacobiano singular y pivote UDL nulo; la curva muestra la distancia mínima de cada trayectoria W4 a x=0 o δ=0.")

fig = go.Figure()
for state, label, key in ((4, "Divergencia", "Divergencia"),
                          (5, "Agotó tope", "Agotó tope"),
                          (6, "Jacobiano singular", "J singular"),
                          (7, "Pivote UDL nulo", "Pivote UDL nulo")):
    fig.add_trace(go.Bar(
        name=label, x=[row["Método"] for row in rows],
        y=[row[key] for row in rows],
        marker_color={"Divergencia": "#222222", "Agotó tope": "#6a7480",
                      "Jacobiano singular": "#9b3d98", "Pivote UDL nulo": "#d9775a"}[label],
        text=[row[key] for row in rows], textposition="auto",
        hovertemplate=f"{label}<br>%{{x}}: %{{y:,}}<extra></extra>",
    ))
fig.update_layout(
    title="Estados terminales en la malla activa", barmode="group", height=440,
    xaxis_title="Método", yaxis_title="Número de inicios",
    font={"size": 15}, legend={"orientation": "h", "y": -0.2},
)

states, _, dmin = ctx.results["W4"]
groups = [
    ("Éxito", states < 4, "#2a8a3a"),
    ("Divergencia", states == 4, "#222222"),
    ("Pivote UDL", states == 7, "#d9775a"),
]
ecdf = go.Figure()
medians = {}
for label, mask, color in groups:
    values = np.sort(np.asarray(dmin)[mask])
    if values.size == 0:
        medians[label] = np.nan
        continue
    medians[label] = float(np.median(values))
    ecdf.add_trace(go.Scatter(
        x=np.maximum(values, 1e-15), y=100 * np.arange(1, len(values) + 1) / len(values),
        mode="lines", name=label, line={"color": color, "width": 2},
        hovertemplate=f"{label}<br>dₘᵢₙ=%{{x:.3e}}<br>Fracción=%{{y:.1f}}%<extra></extra>",
    ))
ecdf.update_layout(
    title="Distribución acumulada de dₘᵢₙ en trayectorias W4",
    height=500, xaxis_type="log", xaxis_title="Distancia mínima al conjunto singular",
    yaxis_title="Trayectorias acumuladas (%)", font={"size": 14},
)

charts = st.columns([1, 1.25])
with charts[0]:
    st.plotly_chart(fig, width="stretch", alt="Conteo de divergencias, agotamientos, singularidades del Jacobiano y fallos de pivote UDL.", config={"displaylogo": False})
with charts[1]:
    st.plotly_chart(ecdf, width="stretch", alt="Distribución acumulada de la distancia mínima a las curvas singulares, separada por estado terminal.", config={"displaylogo": False})
    st.caption("Descripción: fracción acumulada de las trayectorias agrupadas por su estado final.")

st.dataframe(failure_table, width="stretch")
dist_table = pd.DataFrame([{"Grupo": key, "Mediana d_min": val} for key, val in medians.items()])
st.dataframe(dist_table.style.format({"Mediana d_min": "{:.3e}"}), width="stretch")
download_figure_and_table(fig, failure_table.reset_index(), "fallos_y_singularidades")

show_meaning("La escala horizontal de dₘᵢₙ es logarítmica. La tolerancia de singularidad influye en la clasificación de puntos cercanos; las frecuencias pertenecen a este dominio y configuración.")
add_navigation(3)
