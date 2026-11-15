import numpy as np
import pandas as pd
import streamlit as st

from core import DIV, EXH, NOMBRES, PIV, SING, Fnorm, run, trayectoria
from ui import (
    add_navigation, download_figure_and_table, page_context, residual_figure,
    show_meaning, show_what, trajectory_figure,
)


ctx = page_context("Trayectorias de NR, DN y W4", 11)
p = ctx.params
if "basin_initial_x" not in st.session_state:
    st.session_state["basin_initial_x"] = 0.0
if "basin_initial_y" not in st.session_state:
    st.session_state["basin_initial_y"] = 0.0
st.session_state.setdefault("trajectory_frame", 0)
st.session_state.setdefault("trajectory_playing", False)

show_what("El punto inicial seleccionado se sigue simultáneamente con NR, DN y W4; el residuo usa escala logarítmica.")

coordinates = st.columns(2)
x0 = coordinates[0].slider("x₀ inicial", -5.0, 5.0, step=0.001, key="basin_initial_x")
y0 = coordinates[1].slider("y₀ inicial", -5.0, 5.0, step=0.001, key="basin_initial_y")

paths = {
    method: trayectoria(
        method, x0, y0, alpha=p["alpha"], h=p["h"], maxit=p["maxit"],
        p0=(p["p0x"], p["p0y"]),
    )
    for method in ("NR", "DN", "W4")
}
terminal_rows = []
for method in ("NR", "DN", "W4"):
    state, iterations, _ = run(
        method, np.array([x0]), np.array([y0]), alpha=p["alpha"], h=p["h"],
        maxit=p["maxit"], p0=(p["p0x"], p["p0y"]),
    )
    last = paths[method][-1]
    terminal_rows.append({
        "Método": method,
        "Estado": NOMBRES[int(state[0])],
        "Iteraciones": int(iterations[0]),
        "Residuo final": float(Fnorm(np.array([last[0]]), np.array([last[1]]))[0]),
        "Raíz": f"R{int(state[0]) + 1}" if state[0] < 4 else "—",
    })

max_frame = max(1, max(len(points) for points in paths.values()) - 1)
st.session_state["trajectory_frame"] = min(int(st.session_state["trajectory_frame"]), max_frame)

def draw_frame(frame: int) -> None:
    cols = st.columns([1, 1.2])
    with cols[0]:
        path_fig = trajectory_figure(paths, frame)
        st.plotly_chart(path_fig, width="stretch",
                        alt="Trayectorias simultáneas en el plano de las variables x e y para NR, DN y W4.",
                        config={"scrollZoom": True, "displaylogo": False})
    with cols[1]:
        residual_fig = residual_figure(paths, frame)
        st.plotly_chart(residual_fig, width="stretch",
                        alt="Norma del residuo por iteración en escala logarítmica, con línea de tolerancia.",
                        config={"displaylogo": False})
    st.caption(
        "Descripción: recorrido de NR, DN y W4 en el plano y norma del residuo por iteración. "
        f"Se muestran hasta el paso {frame} desde el inicio ({x0:.4f}, {y0:.4f})."
    )

buttons = st.columns(3)
buttons[0].button("▶ Reproducir", key="trajectory_play",
                  on_click=lambda: st.session_state.update(trajectory_playing=True))
buttons[1].button("⏸ Pausar", key="trajectory_pause",
                  on_click=lambda: st.session_state.update(trajectory_playing=False))
buttons[2].button("↺ Reiniciar", key="trajectory_restart",
                  on_click=lambda: st.session_state.update(trajectory_playing=False,
                                                           trajectory_frame=0))

if st.session_state["trajectory_playing"]:
    @st.fragment(run_every=0.5)
    def playback() -> None:
        frame = int(st.session_state.get("trajectory_frame", 0))
        if frame < max_frame:
            frame += 1
            st.session_state["trajectory_frame"] = frame
        else:
            st.session_state["trajectory_playing"] = False
        st.slider("Paso de animación", 0, max_frame, key="trajectory_frame")
        draw_frame(int(st.session_state["trajectory_frame"]))
        if not st.session_state["trajectory_playing"]:
            st.rerun()
    playback()
else:
    frame = st.slider("Paso de animación", 0, max_frame, key="trajectory_frame")
    draw_frame(frame)

summary = pd.DataFrame(terminal_rows)
st.markdown("#### Estado al finalizar")
st.dataframe(summary.style.format({"Residuo final": "{:.3e}"}), width="stretch")
csv_rows = []
for method, points in paths.items():
    residual = Fnorm(points[:, 0], points[:, 1])
    for step, (point, value) in enumerate(zip(points, residual)):
        csv_rows.append({"Método": method, "Iteración": step, "x": point[0],
                         "y": point[1], "Residuo": value})
download_figure_and_table(trajectory_figure(paths), pd.DataFrame(csv_rows), "trayectorias")

show_meaning("NR, DN y W4 parten exactamente del mismo par y comparten tolerancia, región y tope. Una trayectoria ilustra solo ese inicio; no permite generalizar el resultado a toda la malla.")
add_navigation(2)
