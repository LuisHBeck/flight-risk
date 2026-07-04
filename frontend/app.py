import streamlit as st

st.set_page_config(
    page_title="Flight Risk",
    page_icon="✈️",
    layout="wide",
)

st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 1rem; }
[data-testid="stPageLink"] > a {
    border-radius: 6px;
    padding: 4px 8px;
    font-weight: 500;
}
[data-testid="stPageLink"] > a:hover {
    background-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

pg1 = st.Page("pages/1_Previsao.py",  title="Previsão",         icon="✈️")
pg2 = st.Page("pages/2_Explorar.py",  title="Análise Histórica", icon="📊")
pg3 = st.Page("pages/3_Comparar.py",  title="Comparar Voos",     icon="⚖️")
pg4 = st.Page("pages/4_Sobre.py",     title="Sobre nós",         icon="👥")

pages = st.navigation([pg1, pg2, pg3, pg4], position="hidden")

c1, c2, c3, c4, *_ = st.columns([1, 1, 1, 1, 3])
with c1:
    st.page_link(pg1, label="Previsão",         icon="✈️", use_container_width=True)
with c2:
    st.page_link(pg2, label="Análise Histórica", icon="📊", use_container_width=True)
with c3:
    st.page_link(pg3, label="Comparar Voos",     icon="⚖️", use_container_width=True)
with c4:
    st.page_link(pg4, label="Sobre nós",         icon="👥", use_container_width=True)

st.divider()

pages.run()
