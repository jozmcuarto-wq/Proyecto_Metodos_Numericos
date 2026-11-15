import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import (
    add_navigation, download_figure_and_table, empty_results_message,
    page_context, show_meaning, show_what,
)


ctx = page_context("Superposición de fallos sobre las cuencas de W4", 10,
                   require_results=True)
if ctx.results is None:
    empty_results_message()
    add_navigation(1)
    st.stop()

n = ctx.params["N"]
states_w4 = ctx.results["W4"][0]
r34 = (states_w4 == 2) | (states_w4 == 3)
data = []
for method in ("NR", "DN"):
    failed = ctx.results[method][0] >= 4
    inside = failed & r34
    count = int(inside.sum())
    total = int(failed.sum())
    percentage = 100.0 * count / total if total else 0.0
    data.append({"Método": method, "Fallos en cuencas W4 R₃/R₄": count,
                 "Fallos totales": total, "Proporción (%)": percentage})

cols = st.columns(2)
for col, method, reference in zip(cols, ("NR", "DN"), (90.3392, 89.7071)):
    current = next(row["Proporción (%)"] for row in data if row["Método"] == method)
    with col:
        st.metric(f"Fallos de {method} en W4 R₃/R₄", f"{current:.2f}%",
                  f"{current - reference:+.2f} p.p.")
        st.caption(f"Informe: {reference:.2f}%")

show_what("Los puntos negros son inicios donde NR o DN falla; el fondo muestra la raíz de W4 para esos mismos inicios.")

fig = go.Figure()
for method in ("NR", "DN"):
    failed = ctx.results[method][0] >= 4
    fig.add_trace(go.Heatmap(
        x=np.linspace(-5, 5, n), y=np.linspace(-5, 5, n),
        z=np.where(states_w4.reshape(n, n) < 4, states_w4.reshape(n, n), 4),
        zmin=0, zmax=5,
        colorscale=[[0, "#ef6a7d"], [0.2 - 1e-7, "#ef6a7d"],
                    [0.2, "#3f74a8"], [0.4 - 1e-7, "#3f74a8"],
                    [0.4, "#cdbb44"], [0.6 - 1e-7, "#cdbb44"],
                    [0.6, "#2a8a3a"], [0.8 - 1e-7, "#2a8a3a"],
                    [0.8, "#222222"], [1, "#222222"]],
        showscale=(method == "NR"), name=f"Cuenca W4 · fallos {method}",
        legendgroup=method,
        visible=(method == "NR"),
        colorbar={"title": "Raíz W4", "tickvals": [0.5, 1.5, 2.5, 3.5, 4.5],
                  "ticktext": ["R₁", "R₂", "R₃", "R₄", "Fallo"]},
        hovertemplate="x₀=%{x:.4f}<br>y₀=%{y:.4f}<br>Cuenca W4<extra></extra>",
    ))
    yy, xx = np.where(failed.reshape(n, n))
    axis = np.linspace(-5, 5, n)
    fig.add_trace(go.Scattergl(
        x=axis[xx], y=axis[yy], mode="markers", name=f"Fallos de {method}",
        visible=(method == "NR"), legendgroup=method,
        marker={"color": "#111111", "size": 4, "symbol": "x"},
        hovertemplate=f"Fallo de {method}<br>x₀=%{{x:.4f}}<br>y₀=%{{y:.4f}}<extra></extra>",
    ))
fig.update_layout(
    title="Misma malla y escala · selecciona NR o DN en la leyenda",
    height=610, xaxis_title="x inicial", yaxis_title="y inicial",
    font={"size": 15}, legend={"orientation": "h", "groupclick": "togglegroup"},
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(fig, width="stretch", alt="Mapas de las mismas cuencas W4 con fallos de NR o DN en negro; usa la leyenda para cambiar de método.", config={"scrollZoom": True, "displaylogo": False})
table = pd.DataFrame(data)
st.dataframe(table.style.format({"Proporción (%)": "{:.4f}"}), width="stretch")
download_figure_and_table(fig, table, "superposicion_fallos")

show_meaning("La proporción usa como denominador todos los fallos del método, en la malla activa. El informe de referencia corresponde a N=400 y los parámetros base; la asociación no prueba causalidad.")
add_navigation(1)
