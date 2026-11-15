import streamlit as st

from ui import PAGE_FILES, PAGE_LABELS, inject_style


st.set_page_config(page_title="Métodos Numéricos · Comparador", layout="wide", initial_sidebar_state="collapsed")
pages = [
    st.Page("pages/8_Resolver_sistemas.py", title="Comparar métodos", default=True, icon="🧮"),
    st.Page("pages/0_Inicio.py", title="Resultados del informe", icon="📊"),
    *[
        st.Page(f"pages/{filename}", title=label)
        for filename, label in zip(PAGE_FILES, PAGE_LABELS)
    ],
]
selected_page = st.navigation(pages, position="hidden")
inject_style(False)
if selected_page.title in {"Comparar métodos", "Resultados del informe"}:
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='collapsedControl'] "
        "{display: none !important;}</style>",
        unsafe_allow_html=True,
    )
selected_page.run()
