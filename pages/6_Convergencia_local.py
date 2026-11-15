import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import math

from diagnostics import RATE_BANDS, local_rate_summary, nr_order_by_root, w4_spectral_radius
from ui import (
    add_navigation, download_figure_and_table, empty_results_message,
    page_context, show_meaning, show_what,
)


ctx = page_context("Diagnósticos locales y geometría singular", 15,
                   require_results=True)
if ctx.results is None:
    empty_results_message()
    add_navigation(5)
    st.stop()

p = ctx.params
dn_rate = local_rate_summary("DN", p["alpha"], p["h"], (p["p0x"], p["p0y"]))
w4_rate = local_rate_summary("W4", p["alpha"], p["h"], (p["p0x"], p["p0y"]))
order = nr_order_by_root()
rho = w4_spectral_radius(0, p["h"])

def show_rate(value: float, digits: int) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "Sin muestra local"

metrics = st.columns(4)
metrics[0].metric("Orden empírico NR", f"{min(row['median'] for row in order):.4f}–{max(row['median'] for row in order):.4f}")
metrics[0].caption("Informe: 1,973–2,006")
metrics[1].metric("Razón local DN", show_rate(dn_rate["median"], 6),
                  f"{dn_rate['median'] - 0.5:+.6f}" if math.isfinite(dn_rate["median"]) else None)
metrics[1].caption("Informe: 0,500000")
metrics[2].metric("Razón observada W4", show_rate(w4_rate["median"], 4),
                  f"{w4_rate['median'] - 0.5157:+.4f}" if math.isfinite(w4_rate["median"]) else None)
metrics[2].caption("Informe: 0,5157")
metrics[3].metric("Radio espectral FD", f"{rho:.6f}", f"{rho - (1 - p['h']):+.6f}")
metrics[3].caption(f"Predicción local: 1 − Δτ = {1 - p['h']:.4f}")

show_what("NR presenta orden cercano a dos; DN y W4 muestran razones locales acordes con el amortiguamiento en torno a una raíz.")

band_x = [np.sqrt(upper * lower) for upper, lower in RATE_BANDS]
rate_fig = go.Figure()
for method, summary, color in (("DN", dn_rate, "#3f74a8"), ("W4", w4_rate, "#2a8a3a")):
    values = [summary["bands"][band] for band in RATE_BANDS]
    rate_fig.add_trace(go.Scatter(
        x=band_x, y=values, mode="lines+markers", name=method,
        line={"color": color, "width": 2}, marker={"size": 8},
        hovertemplate=f"{method}<br>Error previo=%{{x:.2e}}<br>Razón=%{{y:.6f}}<extra></extra>",
    ))
rate_fig.add_hline(y=1 - p["alpha"], line_dash="dash", line_color="#3f74a8",
                   annotation_text="1 − α (DN)")
rate_fig.add_hline(y=1 - p["h"], line_dash="dot", line_color="#2a8a3a",
                   annotation_text="1 − Δτ (W4)")
rate_fig.update_xaxes(type="log", title="Error de posición antes de la actualización")
rate_fig.update_yaxes(title="Razón eₙ₊₁ / eₙ")
rate_fig.update_layout(title="Razón de error por escala de proximidad", height=430,
                       font={"size": 14})

order_fig = go.Figure(go.Scatter(
    x=[f"R{i + 1}" for i in range(4)], y=[row["median"] for row in order],
    error_y={
        "type": "data", "symmetric": False,
        "array": [row["maximum"] - row["median"] for row in order],
        "arrayminus": [row["median"] - row["minimum"] for row in order],
        "visible": True,
    },
    mode="markers", marker={"color": "#16314d", "size": 12},
    hovertemplate="%{x}<br>Orden q=%{y:.6f}<extra></extra>",
))
order_fig.add_hline(y=2, line_dash="dash", line_color="#d9775a",
                    annotation_text="Orden cuadrático")
order_fig.update_layout(title="Orden empírico de NR por raíz", height=430,
                        xaxis_title="Raíz", yaxis_title="Orden estimado q",
                        font={"size": 14})

states, _, distances = ctx.results["W4"]
proximity_rows = []
for label, mask in (("Éxito", states < 4), ("Divergencia", states == 4),
                    ("Pivote UDL", states == 7)):
    values = np.asarray(distances)[mask]
    proximity_rows.append({
        "Grupo": label,
        "Trayectorias": int(values.size),
        "Mediana d_min": float(np.median(values)) if values.size else np.nan,
        "d_min ≤ 0,025 (%)": 100 * float(np.mean(values <= 0.025)) if values.size else np.nan,
        "d_min ≤ 0,1 (%)": 100 * float(np.mean(values <= 0.1)) if values.size else np.nan,
    })
proximity = pd.DataFrame(proximity_rows)
proximity_fig = go.Figure()
for _, row in proximity.iterrows():
    proximity_fig.add_trace(go.Bar(
        name=row["Grupo"], x=["d_min ≤ 0,025", "d_min ≤ 0,1"],
        y=[row["d_min ≤ 0,025 (%)"], row["d_min ≤ 0,1 (%)"]],
        marker_color={"Éxito": "#2a8a3a", "Divergencia": "#222222",
                      "Pivote UDL": "#d9775a"}[row["Grupo"]],
        text=[f"{row['d_min ≤ 0,025 (%)']:.1f}%", f"{row['d_min ≤ 0,1 (%)']:.1f}%"],
        textposition="outside",
    ))
proximity_fig.update_layout(title="Proximidad de trayectorias a la unión singular",
                            barmode="group", height=430, yaxis_title="Trayectorias (%)",
                            yaxis_range=[0, 115], font={"size": 14})

cols = st.columns(2)
with cols[0]:
    st.plotly_chart(rate_fig, width="stretch", alt="Razón de error local de DN y W4 por escala de proximidad a las raíces.", config={"displaylogo": False})
with cols[1]:
    st.plotly_chart(order_fig, width="stretch", alt="Orden empírico de convergencia de Newton para cada raíz.", config={"displaylogo": False})
st.plotly_chart(proximity_fig, width="stretch", alt="Porcentaje de trayectorias exitosas, divergentes y con pivote nulo cercanas a las curvas singulares.", config={"displaylogo": False})
st.caption("Descripción: comparación por método de razones de error, orden local de NR y fracción de trayectorias cercanas a las curvas singulares.")
st.dataframe(proximity.style.format({
    "Mediana d_min": "{:.3e}", "d_min ≤ 0,025 (%)": "{:.2f}", "d_min ≤ 0,1 (%)": "{:.2f}",
}), width="stretch")

combined_rows = []
for index, row in enumerate(order, 1):
    combined_rows.append({"Diagnóstico": f"Orden NR · R{index}", **row})
combined_rows.extend([
    {"Diagnóstico": "Razón mediana DN", "median": dn_rate["median"],
     "minimum": dn_rate["minimum"], "maximum": dn_rate["maximum"], "count": dn_rate["count"]},
    {"Diagnóstico": "Razón mediana W4", "median": w4_rate["median"],
     "minimum": w4_rate["minimum"], "maximum": w4_rate["maximum"], "count": w4_rate["count"]},
    {"Diagnóstico": "Radio espectral FD W4", "median": rho, "minimum": np.nan,
     "maximum": np.nan, "count": np.nan},
])
download_figure_and_table(rate_fig, pd.DataFrame(combined_rows), "diagnosticos_locales")

show_meaning("La distancia mínima se calcula respecto a x=0 o δ=x²−2y²=0. Acercarse a esas curvas no garantiza un fallo: también hay trayectorias exitosas cercanas y la asociación no es causal.")
add_navigation(5)
