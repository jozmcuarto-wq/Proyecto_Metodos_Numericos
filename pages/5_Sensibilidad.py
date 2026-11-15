import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import grid, run
from ui import (
    PROJECT_DIR, REPORT_DEFAULTS, REPORT_COST_PER_UPDATE, add_navigation,
    download_figure_and_table, page_context, show_meaning, show_what,
)


ctx = page_context("Sensibilidad a parámetros y presupuesto de iteraciones", "13–14")
p = ctx.params
base_config = (p["alpha"] == 0.5 and p["h"] == 0.5 and p["p0x"] == 0.0
               and p["p0y"] == 0.0 and p["maxit"] == 1000)
show_what("Los barridos muestran cómo varían cobertura y costo con el tope, α, Δτ y p₀.")
reference_cards = st.columns(3)
for col, label, value in zip(
    reference_cards,
    ("W4 · Δτ=0,50 · N=200", "W4 · Δτ=0,90 · N=200", "DN · α=0,10 · N=200"),
    (99.65, 71.545, 74.19),
):
    col.metric(label, f"{value:.2f}%", "Referencia del informe")

def load_report_csv(name: str) -> pd.DataFrame:
    path = PROJECT_DIR / "results" / "colab" / name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def coverage(method: str, n: int, alpha: float, h: float,
             p0x: float, p0y: float, maxit: int) -> tuple[float, int, int]:
    x, y = grid(n)
    states, iterations, _ = run(method, x, y, alpha=alpha, h=h,
                                maxit=maxit, p0=(p0x, p0y))
    successes = int(np.count_nonzero(states < 4))
    cost = int(np.sum(iterations, dtype=np.int64))
    return 100.0 * successes / states.size, successes, cost


@st.cache_data(show_spinner="Calculando sensibilidad al tope…", max_entries=8)
def calculate_caps(n: int, alpha: float, h: float, p0x: float, p0y: float) -> pd.DataFrame:
    values = []
    for cap in (10, 20, 50, 100, 200, 500, 1000):
        for method in ("NR", "DN", "W4"):
            rate, successes, work = coverage(method, n, alpha, h, p0x, p0y, cap)
            estimated = REPORT_COST_PER_UPDATE[method] * work / successes if successes else np.nan
            values.append({"tope": cap, "método": method, "éxito (%)": rate,
                           "éxitos": successes, "µs/raíz amortizado": estimated})
    return pd.DataFrame(values)


@st.cache_data(show_spinner="Calculando barrido de α y Δτ…", max_entries=8)
def calculate_steps(n: int, alpha: float, h: float, p0x: float, p0y: float,
                    maxit: int) -> pd.DataFrame:
    rows = []
    for method, values in (("DN", (0.1, 0.25, 0.5, 0.75, 0.9)),
                           ("W4", (0.1, 0.25, 0.5, 0.75, 0.9))):
        for step in values:
            rate, successes, work = coverage(
                method, n, step if method == "DN" else alpha,
                step if method == "W4" else h, p0x, p0y, maxit,
            )
            rows.append({"Método": method, "Paso": step, "Éxito (%)": rate,
                         "Éxitos": successes, "Actualizaciones": work})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Calculando sensibilidad al momento inicial…", max_entries=8)
def calculate_momenta(n: int, alpha: float, h: float, p0x: float, p0y: float,
                      maxit: int, perturbation: float) -> pd.DataFrame:
    base_rate, base_success, _ = coverage("W4", n, alpha, h, p0x, p0y, maxit)
    x, y = grid(n)
    base_states = run("W4", x, y, alpha=alpha, h=h, maxit=maxit, p0=(p0x, p0y))[0]
    rows = []
    for ox, oy in ((0.0, 0.0), (perturbation, 0.0), (-perturbation, 0.0),
                   (0.0, perturbation), (0.0, -perturbation),
                   (perturbation, perturbation), (-perturbation, -perturbation)):
        px, py = p0x + ox, p0y + oy
        states = run("W4", x, y, alpha=alpha, h=h, maxit=maxit, p0=(px, py))[0]
        rows.append({
            "p₀": f"({px:+.2f}, {py:+.2f})",
            "Éxito (%)": 100.0 * np.count_nonzero(states < 4) / states.size,
            "Puntos con estado distinto": int(np.count_nonzero(states != base_states)),
        })
    return pd.DataFrame(rows)


tab_cap, tab_step, tab_p0 = st.tabs(["Tope de iteraciones", "α y Δτ", "Momento inicial p₀"])

with tab_cap:
    st.markdown("#### ¿Cuántos intentos resuelve cada tope?")
    if base_config:
        cap_data = load_report_csv("colab_iteration_cap_sensitivity.csv").rename(
            columns={"tope": "tope", "método": "método", "éxito (%)": "éxito (%)",
                     "µs/raíz amortizado": "µs/raíz amortizado"}
        )
        if cap_data.empty:
            cap_data = calculate_caps(400, p["alpha"], p["h"], p["p0x"], p["p0y"])
    else:
        if st.button("Calcular barrido de topes para esta configuración", key="calc_caps",
                     type="primary"):
            st.session_state["caps_signature"] = (p["N"], p["alpha"], p["h"], p["p0x"],
                                                   p["p0y"], p["maxit"])
        signature = (p["N"], p["alpha"], p["h"], p["p0x"], p["p0y"], p["maxit"])
        if st.session_state.get("caps_signature") == signature:
            cap_data = calculate_caps(p["N"], p["alpha"], p["h"], p["p0x"], p["p0y"])
        else:
            cap_data = pd.DataFrame()
            st.info("El barrido completo tiene 21 corridas. Pulsa el botón para calcularlo.")

    if not cap_data.empty:
        measure = st.radio("Mostrar", ["Cobertura", "Costo estimado por raíz"],
                           horizontal=True, key="cap_measure")
        selected_methods = st.multiselect("Métodos", ["NR", "DN", "W4"],
                                          default=["NR", "DN", "W4"], key="cap_methods")
        y_key = "éxito (%)" if measure == "Cobertura" else "µs/raíz amortizado"
        fig = go.Figure()
        for method in selected_methods:
            part = cap_data[cap_data["método"] == method]
            fig.add_trace(go.Scatter(
                x=part["tope"], y=part[y_key], mode="lines+markers", name=method,
                line={"color": {"NR": "#3f74a8", "DN": "#d9775a", "W4": "#2a8a3a"}[method]},
                hovertemplate=f"{method}<br>Tope=%{{x}}<br>{measure}=%{{y:.4g}}<extra></extra>",
            ))
        fig.update_xaxes(type="log", title="Tope de iteraciones")
        if measure.startswith("Costo"):
            fig.update_yaxes(type="log", title="µs estimados por raíz")
        else:
            fig.update_yaxes(title="Inicios que convergen (%)")
        fig.update_layout(title=f"{measure} según el tope · N={400 if base_config else p['N']}",
                          height=460, font={"size": 14})
        st.plotly_chart(fig, width="stretch", alt="Cobertura y costo estimado por raíz según el tope de iteraciones.", config={"displaylogo": False})
        st.dataframe(cap_data, width="stretch")
        download_figure_and_table(fig, cap_data, "sensibilidad_tope")

with tab_step:
    st.markdown("#### Cobertura al cambiar el factor de paso")
    if base_config:
        step_data = load_report_csv("colab_tau_sensitivity.csv")
        step_data = step_data.rename(columns={"method": "Método", "step": "Paso",
                                              "success_pct": "Éxito (%)"})
        step_data = step_data[step_data["Método"].isin(["DN", "W4"])]
    else:
        if st.button("Calcular barrido de pasos para esta configuración", key="calc_steps",
                     type="primary"):
            st.session_state["steps_signature"] = (p["N"], p["alpha"], p["h"], p["p0x"],
                                                    p["p0y"], p["maxit"])
        signature = (p["N"], p["alpha"], p["h"], p["p0x"], p["p0y"], p["maxit"])
        if st.session_state.get("steps_signature") == signature:
            step_data = calculate_steps(p["N"], p["alpha"], p["h"], p["p0x"],
                                        p["p0y"], p["maxit"])
        else:
            step_data = pd.DataFrame()
            st.info("El barrido usa cinco valores para cada método. Pulsa el botón para calcularlo.")

    if not step_data.empty:
        before_after = st.toggle("Antes / Después · comparar NR con W4", key="step_before_after")
        methods = ["W4"] if before_after else ["DN", "W4"]
        fig = go.Figure()
        for method in methods:
            part = step_data[step_data["Método"] == method]
            fig.add_trace(go.Bar(
                x=part["Paso"], y=part["Éxito (%)"], name=method,
                marker_color={"DN": "#3f74a8", "W4": "#d9775a"}[method],
                text=[f"{v:.2f}%" for v in part["Éxito (%)"]], textposition="outside",
                hovertemplate=f"{method}<br>Paso=%{{x}}<br>Éxito=%{{y:.4f}}%<extra></extra>",
            ))
        if before_after:
            nr_rate = 71.0325 if base_config else coverage(
                "NR", p["N"], p["alpha"], p["h"], p["p0x"], p["p0y"], p["maxit"]
            )[0]
            w4_steps = step_data[step_data["Método"] == "W4"]["Paso"]
            fig.add_trace(go.Scatter(
                x=w4_steps, y=[nr_rate] * len(w4_steps), mode="lines+markers",
                name="NR · referencia sin amortiguamiento",
                line={"color": "#16314d", "dash": "dash", "width": 2},
                hovertemplate="NR<br>Paso=%{x}<br>Éxito=%{y:.4f}%<extra></extra>",
            ))
        fig.update_layout(title="DN según α y W4 según Δτ", barmode="group",
                          xaxis_title="α (DN) o Δτ (W4)", yaxis_title="Cobertura (%)",
                          yaxis_range=[0, 110], height=460, font={"size": 14})
        st.plotly_chart(fig, width="stretch", alt="Barras de cobertura de DN y W4 según el paso; el modo antes/después superpone la cobertura NR en la misma malla.", config={"displaylogo": False})
        st.caption("Referencia del informe: N=200. En configuración base se muestran sus datos publicados.")
        st.dataframe(step_data[["Método", "Paso", "Éxito (%)"]], width="stretch")
        download_figure_and_table(fig, step_data, "sensibilidad_pasos")

with tab_p0:
    st.markdown("#### Efecto de perturbar el momento inicial")
    perturbation = st.slider("Magnitud de la perturbación de p₀", 0.01, 0.2, 0.05, 0.01,
                             key="momentum_perturbation")
    report_view = base_config and p["N"] in (100, 200, 400) and perturbation == 0.05
    if report_view:
        p0_data = load_report_csv("colab_p0_sensitivity.csv").rename(
            columns={"p0x": "p₀ x", "p0y": "p₀ y", "success_pct": "Éxito (%)"}
        )
        p0_data["p₀"] = p0_data.apply(lambda row: f"({row['p₀ x']:+.2f}, {row['p₀ y']:+.2f})", axis=1)
        p0_data = p0_data[["p₀", "Éxito (%)"]]
    else:
        if st.button("Calcular perturbaciones de p₀", key="calc_p0", type="primary"):
            st.session_state["p0_signature"] = (p["N"], p["alpha"], p["h"], p["p0x"],
                                                 p["p0y"], p["maxit"], perturbation)
        signature = (p["N"], p["alpha"], p["h"], p["p0x"], p["p0y"], p["maxit"], perturbation)
        if st.session_state.get("p0_signature") == signature:
            p0_data = calculate_momenta(p["N"], p["alpha"], p["h"], p["p0x"],
                                        p["p0y"], p["maxit"], perturbation)
        else:
            p0_data = pd.DataFrame()
            st.info("La referencia usa N=100. Pulsa el botón para comparar con la configuración actual.")

    if not p0_data.empty:
        fig = go.Figure(go.Bar(
            x=p0_data["p₀"], y=p0_data["Éxito (%)"],
            marker_color="#2a8a3a", text=[f"{x:.2f}%" for x in p0_data["Éxito (%)"]],
            textposition="outside",
            hovertemplate="p₀=%{x}<br>Éxito=%{y:.4f}%<extra></extra>",
        ))
        fig.update_layout(title="Cobertura de W4 según p₀", yaxis_title="Cobertura (%)",
                          yaxis_range=[0, 110], height=450, font={"size": 14})
        st.plotly_chart(fig, width="stretch", alt="Cobertura de W4 bajo perturbaciones del momento inicial p cero.", config={"displaylogo": False})
        st.caption("Al cambiar p₀ puede cambiar la asignación a raíces aunque la cobertura total sea similar.")
        st.dataframe(p0_data, width="stretch")
        download_figure_and_table(fig, p0_data, "sensibilidad_momento_inicial")

show_meaning("Cada barrido describe los valores muestreados y la malla indicada; no se extrapola a pasos, momentos o sistemas fuera de la prueba.")
add_navigation(4)
