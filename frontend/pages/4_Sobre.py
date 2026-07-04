import streamlit as st

st.title("👥 Sobre nós")
st.caption("Conheça o time por trás do Flight Risk")

st.divider()

devs = [
    {
        "name": "Luis Beck",
        "role": "Cientista de Dados",
        "github_user": "luishbeck",
        "linkedin": "https://linkedin.com/in/luishbeck",
        "github": "https://github.com/luishbeck",
    },
    {
        "name": "Cliscia Fontoura",
        "role": "Cientista de Dados",
        "github_user": "Cliscia",
        "linkedin": "https://linkedin.com/in/cliscia-fontoura",
        "github": "https://github.com/Cliscia",
    },
    {
        "name": "Felipe Millani",
        "role": "Cientista de Dados",
        "github_user": "Felami10",
        "linkedin": "https://linkedin.com/in/felipe-millani",
        "github": "https://github.com/Felami10",
    },
]

cols = st.columns(len(devs), gap="large")

for col, dev in zip(cols, devs):
    with col:
        st.image(f"https://github.com/{dev['github_user']}.png", width=120)
        st.subheader(dev["name"])
        st.caption(dev["role"])
        st.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white)]({dev['linkedin']})")
        st.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)]({dev['github']})")

st.divider()
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:13px'>"
    "✈️ Flight Risk — Predição de atrasos em voos domésticos brasileiros"
    "</div>",
    unsafe_allow_html=True,
)
