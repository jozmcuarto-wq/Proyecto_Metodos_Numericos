import streamlit as st

from ui import PAGE_FILES, PAGE_LABELS, REPORT_COVERAGE, home_context


params, presentation = home_context()
st.title("Laboratorio de Métodos Numéricos")
st.subheader("Resolver sistemas no lineales y comparar métodos iterativos")
if not presentation:
    st.markdown(
        "Escribe dos funciones o elige un ejemplo para comparar NR, Newton amortiguado y W4 "
        "desde el mismo punto inicial. También puedes explorar los resultados del proyecto sobre "
        "**F₁ = x² + y² − 4, F₂ = x²y − 1**, en la región [−5, 5]²."
    )
    st.page_link("pages/8_Resolver_sistemas.py", label="Abrir el solucionador de sistemas", icon="🧮", width="stretch")
    st.markdown("### Valores de referencia del informe")
    cards = st.columns(3)
    for col, method in zip(cards, ("W4", "NR", "DN")):
        col.metric(f"Cobertura observada · {method}", f"{REPORT_COVERAGE[method]:.2f}%")
    st.caption("Proporciones sobre una malla finita y determinista de 400 × 400 puntos iniciales.")
    st.markdown("### Análisis del experimento documentado")

analysis_pages = list(zip(PAGE_LABELS, PAGE_FILES))
for start in range(0, len(analysis_pages), 3):
    group = analysis_pages[start:start + 3]
    cards = st.columns(len(group))
    for offset, (col, (label, filename)) in enumerate(zip(cards, group)):
        index = start + offset + 1
        col.markdown(f"**{index}. {label}**  \nExploración del experimento")
        col.page_link(f"pages/{filename}", label=f"Abrir · {label}", width="stretch")

if not presentation:
    st.markdown("---")
    st.markdown(
        "**Alcance.** Los porcentajes describen esta malla y esta configuración; "
        "las tasas son locales y la asociación entre fallos y singularidades es descriptiva, no causal. "
        "No se infiere superioridad universal de ningún método."
    )
