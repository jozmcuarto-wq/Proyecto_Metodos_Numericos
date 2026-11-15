from __future__ import annotations

import pathlib
import sys

from streamlit.testing.v1 import AppTest

APP_PATH = str(pathlib.Path(__file__).resolve().parents[1])
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)

from ui import _project_root


APP_DIR = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINT = APP_DIR / "app.py"
PAGES = (
    "0_Inicio.py",
    "1_Cuencas_y_cobertura.py",
    "2_Superposicion_de_fallos.py",
    "3_Trayectorias.py",
    "4_Fallos_y_singularidades.py",
    "5_Sensibilidad.py",
    "6_Convergencia_local.py",
    "7_Costo.py",
    "8_Resolver_sistemas.py",
)


def test_project_root_supports_nested_and_flat_repository_layouts(tmp_path):
    repo = tmp_path / "repository"
    nested_app = repo / "app"
    nested_app.mkdir(parents=True)
    assert _project_root(nested_app) == repo

    (repo / "data").mkdir()
    (repo / "results").mkdir()
    assert _project_root(repo) == repo


def test_all_registered_pages_render_without_streamlit_errors():
    for filename in PAGES:
        app = AppTest.from_file(str(ENTRYPOINT), default_timeout=90).run()
        app.switch_page(f"pages/{filename}").run(timeout=90)

        assert not app.exception, f"{filename}: {[error.message for error in app.exception]}"
        assert not app.warning, f"{filename}: {[warning.value for warning in app.warning]}"
        if filename not in {"0_Inicio.py", "8_Resolver_sistemas.py"}:
            assert app.get("plotly_chart"), f"{filename}: expected at least one interactive chart"


def test_page_context_restores_parameters_after_widget_state_is_cleared(tmp_path):
    probe = tmp_path / "sidebar_probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import streamlit as st\n"
        "st.session_state['_saved_global_n'] = 200\n"
        "from ui import page_context\n"
        "context = page_context('Probe', 1)\n"
        "st.metric('Resolución restaurada', context.params['N'])\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(probe), default_timeout=90).run()
    assert not app.exception
    assert app.session_state["global_n"] == 200
