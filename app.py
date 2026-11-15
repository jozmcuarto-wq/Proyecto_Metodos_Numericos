import streamlit as st

from ui import PAGE_FILES, PAGE_LABELS, common_sidebar, inject_style


st.set_page_config(page_title="Laboratorio de Métodos Numéricos", layout="wide", initial_sidebar_state="expanded")
pages = [
    st.Page("pages/8_Resolver_sistemas.py", title="Resolver sistemas", default=True, icon="🧮"),
    st.Page("pages/0_Inicio.py", title="Análisis del proyecto", icon="🏠"),
    *[
        st.Page(f"pages/{filename}", title=label)
        for filename, label in zip(PAGE_FILES, PAGE_LABELS)
    ],
]
selected_page = st.navigation(pages)
if selected_page.title == "Resolver sistemas":
    inject_style(False)
else:
    presentation_params, presentation_mode = common_sidebar()
    inject_style(presentation_mode)
selected_page.run()
