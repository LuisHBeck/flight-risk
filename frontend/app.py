import streamlit as st

st.set_page_config(
    page_title="Flight Risk",
    page_icon="✈️",
    layout="wide",
)

pages = st.navigation([
    st.Page("pages/1_Previsao.py",  title="Previsão",          icon="✈️"),
    st.Page("pages/2_Explorar.py",  title="Análise Histórica",  icon="📊"),
    st.Page("pages/3_Comparar.py",  title="Comparar Voos",      icon="⚖️"),
    st.Page("pages/4_Sobre.py",     title="Sobre nós",          icon="👥"),
])
pages.run()
