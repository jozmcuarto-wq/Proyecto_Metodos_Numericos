import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import METHOD_COLORS, REPORT_COVERAGE


st.title("Resultados principales del informe")
st.write("Resumen del experimento con 160 000 puntos iniciales en el sistema estudiado.")
st.page_link("pages/8_Resolver_sistemas.py", label="← Volver a comparar métodos", icon="🧮")

st.subheader("Cobertura observada")
coverage_cols = st.columns(3)
for col, method in zip(coverage_cols, ("NR", "DN", "W4")):
    col.metric(method, f"{REPORT_COVERAGE[method]:.2f}%")

fig = go.Figure(go.Bar(
    x=[REPORT_COVERAGE[method] for method in ("NR", "DN", "W4")],
    y=["NR", "DN", "W4"],
    orientation="h",
    marker_color=[METHOD_COLORS[method] for method in ("NR", "DN", "W4")],
    text=[f"{REPORT_COVERAGE[method]:.2f}%" for method in ("NR", "DN", "W4")],
    textposition="outside",
    hovertemplate="%{y}: %{x:.2f}% de los inicios<extra></extra>",
))
fig.update_layout(
    xaxis={"title": "Inicios que convergen (%)", "range": [0, 110]},
    yaxis={"autorange": "reversed"},
    height=260,
    margin={"l": 30, "r": 40, "t": 15, "b": 45},
    showlegend=False,
)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
st.info("En esta malla y con esta configuración, W4 resolvió más puntos iniciales. El resultado describe este experimento; no garantiza superioridad para cualquier sistema.")

st.subheader("Costo estimado por raíz")
root_cost = {"NR": 18.804, "DN": 21.338, "W4": 3.382}
cost_cols = st.columns(3)
for col, method in zip(cost_cols, ("NR", "DN", "W4")):
    col.metric(method, f"{root_cost[method]:.3f} µs")
st.caption("Son estimaciones del entorno del informe; los tiempos cambian según el equipo y la ejecución.")

with st.expander("¿Qué pasa si se cambia el paso de W4?"):
    st.caption("Barrido de sensibilidad con una malla de 200 × 200 puntos iniciales.")
    sensitivity = pd.DataFrame({
        "Paso Δτ": [0.50, 0.90],
        "Cobertura": [99.65, 71.545],
    })
    st.dataframe(sensitivity.style.format({"Paso Δτ": "{:.2f}", "Cobertura": "{:.3f}%"}),
                 hide_index=True, width="stretch")
    st.caption("La cobertura depende del paso. Para estos datos, Δτ = 0.50 dio mayor cobertura que Δτ = 0.90.")

with st.expander("Configuración usada en el informe"):
    st.markdown(
        "- Malla de 400 × 400 puntos iniciales en [−5, 5]².\n"
        "- Tolerancia residual: 10⁻⁶; máximo: 1000 iteraciones.\n"
        "- Newton amortiguado: α = 0.5. W4: Δτ = 0.5 y p₀ = (0, 0)."
    )
