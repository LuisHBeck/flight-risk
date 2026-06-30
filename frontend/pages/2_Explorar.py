from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_DIR = Path(__file__).parent.parent.parent / "model"


@st.cache_data
def load_airports():
    path = MODEL_DIR / "airports_reference.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=["ident"])
    return {
        row["ident"]: f"{row['ident']} — {row['municipality']}"
        if pd.notna(row.get("municipality")) else row["ident"]
        for _, row in df.iterrows()
    }

st.set_page_config(page_title="Explorar Dataset — Flight Risk", page_icon="📊", layout="wide")

st.title("📊 Explorar Dataset")
st.caption("Análise exploratória interativa de voos domésticos brasileiros (2022–2025)")

DATA_DIR = Path(__file__).parent.parent.parent / ".data"
CANDIDATES = ["flights_features.parquet", "flights_with_weather.parquet"]

DATA_PATH = next((DATA_DIR / f for f in CANDIDATES if (DATA_DIR / f).exists()), None)

@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)

    # normaliza nome do target
    if "is_delayed" not in df.columns:
        if "delayed_15_min_plus" in df.columns:
            df["is_delayed"] = df["delayed_15_min_plus"].astype(int)
        elif "arr_delay_min" in df.columns:
            df["is_delayed"] = (df["arr_delay_min"] >= 15).astype(int)
        elif "arr_actual" in df.columns and "arr_scheduled" in df.columns:
            arr_delay = (
                pd.to_datetime(df["arr_actual"]) - pd.to_datetime(df["arr_scheduled"])
            ).dt.total_seconds() / 60
            df["is_delayed"] = (arr_delay >= 15).astype(int)

    # garante coluna route
    if "route" not in df.columns:
        df["route"] = df["origin_icao"] + "_" + df["destination_icao"]

    # trata companhias sem código
    df["airline_icao"] = df["airline_icao"].fillna("N/D").str.strip()

    wx_cols  = [c for c in df.columns if c.startswith("origin_wx_") or c.startswith("destination_wx_")]
    geo_cols = [c for c in df.columns if c in ("origin_lat", "origin_lon")]
    keep = ["airline_icao", "origin_icao", "destination_icao",
            "dep_scheduled", "is_delayed", "route",
            "origin_region", "destination_region"] + wx_cols + geo_cols
    return df[[c for c in keep if c in df.columns]]

if DATA_PATH is None:
    st.warning("Dataset não encontrado. Coloque `flights_features.parquet` ou `flights_with_weather.parquet` em `.data/` para usar esta página.")
    st.stop()

airport_labels = load_airports()

with st.spinner("Carregando dataset..."):
    df = load_data(DATA_PATH)
    df["dep_scheduled"] = pd.to_datetime(df["dep_scheduled"])
    df["hour"]    = df["dep_scheduled"].dt.hour
    df["weekday"] = df["dep_scheduled"].dt.weekday
    df["month"]   = df["dep_scheduled"].dt.month
    df["year"]    = df["dep_scheduled"].dt.year

# ── Filtros ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros")

    airlines = sorted(df["airline_icao"].dropna().unique())
    sel_airlines = st.multiselect("Companhia aérea", airlines)

    if "origin_icao" in df.columns:
        origin_icao_list = sorted(df["origin_icao"].dropna().unique())
        origin_label_list = [airport_labels.get(c, c) for c in origin_icao_list]
        label_to_icao = dict(zip(origin_label_list, origin_icao_list))
        sel_origin_labels = st.multiselect("Aeroporto de origem", origin_label_list)
        sel_origins = [label_to_icao[l] for l in sel_origin_labels]
    else:
        sel_origins = []

    min_date = df["dep_scheduled"].min().date()
    max_date = df["dep_scheduled"].max().date()
    date_range = st.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df.copy()
if sel_airlines:
    filtered = filtered[filtered["airline_icao"].isin(sel_airlines)]
if sel_origins:
    filtered = filtered[filtered["origin_icao"].isin(sel_origins)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["dep_scheduled"].dt.date >= date_range[0]) &
        (filtered["dep_scheduled"].dt.date <= date_range[1])
    ]

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.subheader("Resumo")
k1, k2, k3, k4 = st.columns(4)

total     = len(filtered)
delay_rate = filtered["is_delayed"].mean() if total > 0 else 0

route_delay = (
    filtered.groupby("route")["is_delayed"].mean().sort_values(ascending=False)
    if total > 0 else pd.Series(dtype=float)
)
worst_route = route_delay.index[0] if len(route_delay) > 0 else "—"

airline_delay = (
    filtered.groupby("airline_icao")["is_delayed"].mean().sort_values()
    if total > 0 else pd.Series(dtype=float)
)
best_airline = airline_delay.index[0] if len(airline_delay) > 0 else "—"

k1.metric("Taxa de atraso",     f"{delay_rate*100:.1f}%")
k2.metric("Voos no período",    f"{total:,}")
k3.metric("Rota mais atrasada", worst_route)
k4.metric("Companhia mais pontual", best_airline)

st.divider()

# ── Visualizações ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Por horário", "Por rota", "Por companhia", "Clima", "Sazonalidade", "Mapa"
])

with tab1:
    st.markdown("#### Heatmap: hora × dia da semana")
    hm = (
        filtered.groupby(["hour", "weekday"])["is_delayed"]
        .mean()
        .reset_index()
    )
    hm.columns = ["Hora", "Dia da semana", "Taxa de atraso"]
    hm["Dia da semana"] = hm["Dia da semana"].map(
        {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
    )
    fig = px.density_heatmap(
        hm, x="Hora", y="Dia da semana", z="Taxa de atraso",
        color_continuous_scale="Reds", nbinsx=24,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Rotas com maior taxa de atraso (top 20)")
    top_routes = (
        filtered.groupby("route")["is_delayed"]
        .agg(["mean", "count"])
        .query("count >= 100")
        .sort_values("mean", ascending=False)
        .head(20)
        .reset_index()
    )
    top_routes.columns = ["Rota", "Taxa de atraso", "Voos"]
    fig = px.bar(
        top_routes, x="Taxa de atraso", y="Rota",
        orientation="h", color="Taxa de atraso",
        color_continuous_scale="Reds",
        text=top_routes["Taxa de atraso"].map(lambda x: f"{x*100:.1f}%"),
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### Taxa de atraso por companhia aérea")
    by_airline = (
        filtered.groupby("airline_icao")["is_delayed"]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_airline.columns = ["Companhia", "Taxa de atraso", "Voos"]
    fig = px.bar(
        by_airline.sort_values("Taxa de atraso", ascending=True),
        x="Taxa de atraso", y="Companhia",
        orientation="h", color="Taxa de atraso",
        color_continuous_scale="RdYlGn_r",
        text=by_airline.sort_values("Taxa de atraso", ascending=True)["Taxa de atraso"].map(lambda x: f"{x*100:.1f}%"),
    )
    st.plotly_chart(fig, use_container_width=True)

WX_CODE_LABELS = {
    0: "Céu limpo", 1: "Quase limpo", 2: "Parcialmente nublado", 3: "Nublado",
    45: "Neblina", 48: "Neblina com geada",
    51: "Chuvisco leve", 53: "Chuvisco mod.", 55: "Chuvisco intenso",
    61: "Chuva leve", 63: "Chuva mod.", 65: "Chuva forte",
    71: "Neve leve", 73: "Neve mod.", 75: "Neve intensa",
    80: "Pancadas leves", 81: "Pancadas mod.", 82: "Pancadas fortes",
    95: "Trovoada", 96: "Trovoada c/ granizo", 99: "Trovoada intensa",
}

with tab4:
    has_wx = "origin_wx_weathercode" in filtered.columns and filtered["origin_wx_weathercode"].notna().any()
    if not has_wx:
        st.info("Dataset atual não possui dados climáticos. Carregue `flights_with_weather.parquet`.")
    else:
        wx_df = filtered.dropna(subset=["origin_wx_weathercode", "is_delayed"]).copy()

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### Atraso por condição climática (origem)")
            wx_df["Condição"] = wx_df["origin_wx_weathercode"].astype(int).map(WX_CODE_LABELS).fillna("Outro")
            wx_grouped = (
                wx_df.groupby("Condição")["is_delayed"]
                .agg(["mean", "count"])
                .query("count >= 200")
                .sort_values("mean", ascending=True)
                .reset_index()
            )
            wx_grouped.columns = ["Condição", "Taxa de atraso", "Voos"]
            fig = px.bar(
                wx_grouped, x="Taxa de atraso", y="Condição",
                orientation="h", color="Taxa de atraso",
                color_continuous_scale="RdYlGn_r",
                text=wx_grouped["Taxa de atraso"].map(lambda x: f"{x*100:.1f}%"),
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown("#### Atraso por faixa de vento (origem)")
            wx_df["Vento (km/h)"] = pd.cut(
                wx_df["origin_wx_windspeed_10m"],
                bins=[0, 10, 20, 30, 40, 200],
                labels=["0–10", "10–20", "20–30", "30–40", "40+"],
            )
            wind_grouped = (
                wx_df.groupby("Vento (km/h)", observed=True)["is_delayed"]
                .agg(["mean", "count"])
                .reset_index()
            )
            wind_grouped.columns = ["Vento (km/h)", "Taxa de atraso", "Voos"]
            fig = px.bar(
                wind_grouped, x="Vento (km/h)", y="Taxa de atraso",
                color="Taxa de atraso", color_continuous_scale="RdYlGn_r",
                text=wind_grouped["Taxa de atraso"].map(lambda x: f"{x*100:.1f}%"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Atraso por precipitação (origem)")
        wx_df["Precipitação (mm)"] = pd.cut(
            wx_df["origin_wx_precipitation"],
            bins=[-0.01, 0, 1, 5, 10, 500],
            labels=["Sem chuva", "0–1 mm", "1–5 mm", "5–10 mm", "10+ mm"],
        )
        prec_grouped = (
            wx_df.groupby("Precipitação (mm)", observed=True)["is_delayed"]
            .agg(["mean", "count"])
            .reset_index()
        )
        prec_grouped.columns = ["Precipitação (mm)", "Taxa de atraso", "Voos"]
        fig = px.bar(
            prec_grouped, x="Precipitação (mm)", y="Taxa de atraso",
            color="Taxa de atraso", color_continuous_scale="RdYlGn_r",
            text=prec_grouped["Taxa de atraso"].map(lambda x: f"{x*100:.1f}%"),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("#### Taxa de atraso mensal (2022–2025)")
    by_month = (
        filtered.groupby(["year", "month"])["is_delayed"]
        .mean()
        .reset_index()
    )
    by_month["período"] = pd.to_datetime(
        by_month["year"].astype(str) + "-" + by_month["month"].astype(str)
    )
    fig = px.line(
        by_month.sort_values("período"),
        x="período", y="is_delayed",
        labels={"is_delayed": "Taxa de atraso", "período": ""},
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with tab6:
    has_geo = "origin_lat" in filtered.columns and "origin_lon" in filtered.columns
    if not has_geo:
        st.info("Dados de geolocalização não disponíveis no dataset atual.")
    else:
        airport_stats = (
            filtered.groupby(["origin_icao", "origin_lat", "origin_lon"])["is_delayed"]
            .agg(["mean", "count"])
            .reset_index()
        )
        airport_stats.columns = ["icao", "lat", "lon", "delay_rate", "flights"]
        airport_stats = airport_stats[airport_stats["flights"] >= 50]
        airport_stats["label"]      = airport_stats["icao"].map(airport_labels).fillna(airport_stats["icao"])
        airport_stats["delay_pct"]  = (airport_stats["delay_rate"] * 100).round(1)
        airport_stats["size_norm"]  = (airport_stats["flights"] / airport_stats["flights"].max() * 35 + 5).round(0)

        fig = px.scatter_mapbox(
            airport_stats,
            lat="lat", lon="lon",
            color="delay_rate",
            size="size_norm",
            hover_name="label",
            custom_data=["delay_pct", "flights"],
            color_continuous_scale="RdYlGn_r",
            range_color=[0.10, 0.50],
            size_max=40,
            zoom=3.5,
            center={"lat": -14, "lon": -51},
            mapbox_style="carto-positron",
        )
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>Taxa de atraso: %{customdata[0]:.1f}%<br>Voos: %{customdata[1]:,}<extra></extra>"
        )
        fig.update_layout(
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Taxa de atraso", tickformat=".0%"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tamanho do ponto = volume de voos | Cor = taxa histórica de atraso na partida")
