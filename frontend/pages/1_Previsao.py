import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE  = os.getenv("API_BASE_URL", "https://flight-risk-430527068071.us-central1.run.app")
THRESHOLD = 0.3
MODEL_DIR = Path(__file__).parent.parent.parent / "model"
DATA_DIR  = Path(__file__).parent.parent.parent / ".data"

AIRLINE_NAMES = {
    "ABJ": "Abaeté Aviação",
    "ACN": "Azul Conecta",
    "AEB": "Avion Express Brasil",
    "ASO": "Avianca Brasil",
    "AZU": "Azul Linhas Aéreas",
    "BPC": "Braspress Air Cargo",
    "CQB": "Apuí Táxi Aéreo",
    "GLO": "Gol Linhas Aéreas",
    "LTG": "LATAM Cargo Brasil",
    "MWM": "Modern Logistics",
    "OMI": "OMNI Táxi Aéreo",
    "PAM": "MAP Linhas Aéreas",
    "PLS": "Placar Linhas Aéreas",
    "PTB": "Passaredo",
    "SID": "Sideral Linhas Aéreas",
    "TAM": "LATAM Brasil",
    "TOT": "Total Express",
    "TTL": "Total Linhas Aéreas",
}

ALL_AIRLINES = ["ABJ", "ACN", "AEB", "ASO", "AZU", "BPC", "CQB", "GLO",
                "LTG", "MWM", "OMI", "PAM", "PLS", "PTB", "SID", "TAM", "TOT", "TTL"]

WX_LABELS = {
    range(0, 1):    "☀️ Céu limpo",
    range(1, 4):    "⛅ Parcialmente nublado",
    range(45, 49):  "🌫️ Neblina",
    range(51, 68):  "🌧️ Chuva",
    range(71, 78):  "❄️ Neve",
    range(80, 83):  "🌦️ Pancadas de chuva",
    range(95, 100): "⛈️ Tempestade",
}


def wx_label(code):
    if code is None:
        return "—"
    code = int(code)
    for r, label in WX_LABELS.items():
        if code in r:
            return label
    return "🌤️ Variável"


def risk_info(proba):
    if proba < THRESHOLD:
        return "🟢", "Baixo risco de atraso", "#16a34a"
    elif proba < 0.6:
        return "🟡", "Risco moderado de atraso", "#d97706"
    else:
        return "🔴", "Alto risco de atraso", "#dc2626"


def build_gauge(proba, threshold=THRESHOLD):
    _, _, color = risk_info(proba)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 30],  "color": "#f0fdf4"},
                {"range": [30, 60], "color": "#fefce8"},
                {"range": [60, 100], "color": "#fef2f2"},
            ],
            "threshold": {
                "line": {"color": "#475569", "width": 3},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=230, margin=dict(t=10, b=0, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def log_missing_request(context: dict):
    try:
        log_file = DATA_DIR / "missing_requests.jsonl"
        entry = {"timestamp": datetime.now().isoformat(), **context}
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def show_api_error(exc, context: dict | None = None):
    is_no_data = (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response.status_code in (404, 422, 400)
    )
    is_offline = isinstance(exc, requests.exceptions.ConnectionError)

    if is_no_data:
        if context:
            log_missing_request(context)
        st.markdown(
            """
            <div style="border-radius:12px; border:1.5px solid #e2e8f0;
                        padding:28px 24px; text-align:center; background:#f8fafc">
                <div style="font-size:36px; margin-bottom:10px">🗺️</div>
                <div style="font-size:17px; font-weight:700; color:#1e293b; margin-bottom:6px">
                    Ainda não temos dados suficientes para essa combinação
                </div>
                <div style="font-size:14px; color:#64748b; line-height:1.6">
                    Registramos sua consulta e mapeamos o que falta.<br>
                    Isso nos ajuda a expandir o modelo para novas rotas e companhias. 🙏
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif is_offline:
        st.markdown(
            """
            <div style="border-radius:12px; border:1.5px solid #fca5a5;
                        padding:28px 24px; text-align:center; background:#fff1f2">
                <div style="font-size:36px; margin-bottom:10px">🛰️</div>
                <div style="font-size:17px; font-weight:700; color:#991b1b; margin-bottom:6px">
                    Serviço temporariamente fora do ar
                </div>
                <div style="font-size:14px; color:#7f1d1d">
                    Tente novamente em alguns instantes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="border-radius:12px; border:1.5px solid #fcd34d;
                        padding:28px 24px; text-align:center; background:#fffbeb">
                <div style="font-size:36px; margin-bottom:10px">⚠️</div>
                <div style="font-size:17px; font-weight:700; color:#92400e; margin-bottom:6px">
                    Algo inesperado aconteceu
                </div>
                <div style="font-size:14px; color:#78350f">
                    Tente novamente. Se o problema persistir, entre em contato com o suporte.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_route_hour_stats():
    for fname in ["flights_with_weather.parquet", "flights_features.parquet"]:
        p = DATA_DIR / fname
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["origin_icao", "destination_icao", "dep_scheduled", "arr_scheduled", "arr_actual"])
            arr_delay = (pd.to_datetime(df["arr_actual"]) - pd.to_datetime(df["arr_scheduled"])).dt.total_seconds() / 60
            df["is_delayed"] = (arr_delay >= 15).astype(int)
        except Exception:
            try:
                df = pd.read_parquet(p, columns=["origin_icao", "destination_icao", "dep_scheduled", "delayed_15_min_plus"])
                df["is_delayed"] = df["delayed_15_min_plus"].astype(int)
            except Exception:
                continue
        df["dep_hour"] = pd.to_datetime(df["dep_scheduled"]).dt.hour
        df["route"] = df["origin_icao"] + "_" + df["destination_icao"]
        stats = (
            df.groupby(["route", "dep_hour"])["is_delayed"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats.columns = ["route", "dep_hour", "delay_rate", "count"]
        return stats[stats["count"] >= 20].reset_index(drop=True)
    return None


@st.cache_data
def load_airports():
    df = pd.read_csv(DATA_DIR / "airports_reference.csv")
    df = df.dropna(subset=["ident"])
    df["label"] = df.apply(
        lambda r: f"{r['ident']} — {r['municipality']}" if pd.notna(r.get("municipality")) else r["ident"],
        axis=1,
    )
    return df.sort_values("municipality").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_route_combos():
    for fname in ["flights_with_weather.parquet", "flights_features.parquet"]:
        p = DATA_DIR / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["origin_icao", "destination_icao", "airline_icao"])
        origin_to_dests = (
            df.groupby("origin_icao")["destination_icao"]
            .unique()
            .apply(sorted)
            .to_dict()
        )
        df["route"] = df["origin_icao"] + "_" + df["destination_icao"]
        route_to_airlines = (
            df.groupby("route")["airline_icao"]
            .unique()
            .apply(sorted)
            .to_dict()
        )
        return origin_to_dests, route_to_airlines
    return {}, {}


def call_weather(origin_icao, destination_icao, dep_scheduled):
    resp = requests.get(
        f"{API_BASE}/weather/",
        params={"origin_icao": origin_icao, "destination_icao": destination_icao, "dep_scheduled": dep_scheduled},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def call_predict(payload):
    resp = requests.post(f"{API_BASE}/predict/", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def call_explain_shap(result: dict) -> str:
    shap = result.get("shap_explanation", {})
    all_values = shap.get("values", {})
    top5 = dict(sorted(all_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    payload = {
        "shap_explanation": {"base_value": shap.get("base_value", 0), "values": top5},
        "delay_proba":       result["delay_proba"],
        "airline_icao":      result["airline_icao"],
        "origin_icao":       result["origin_icao"],
        "destination_icao":  result["destination_icao"],
    }
    resp = requests.post(f"{API_BASE}/explain/shap", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("explanation", "")


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("✈️ Flight Risk")
st.caption("Previsão de probabilidade de atraso em voos domésticos brasileiros")
st.divider()

airports = load_airports()
airport_options = airports["label"].tolist()
icao_from_label = dict(zip(airports["label"], airports["ident"]))

airline_labels = [
    f"{k} — {AIRLINE_NAMES[k]}" if k in AIRLINE_NAMES else k
    for k in sorted(ALL_AIRLINES)
]
icao_from_airline = {
    (f"{k} — {AIRLINE_NAMES[k]}" if k in AIRLINE_NAMES else k): k
    for k in ALL_AIRLINES
}

origin_to_dests, route_to_airlines = load_route_combos()

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Dados do voo")

    # 1. Origem — todas as opções
    origin_sel  = st.selectbox("Aeroporto de origem", airport_options)
    origin_code_sel = icao_from_label.get(origin_sel, "")

    # 2. Destino — só destinos com rota a partir da origem escolhida
    valid_dest_codes = set(origin_to_dests.get(origin_code_sel, []))
    dest_opts = (
        [opt for opt in airport_options if icao_from_label.get(opt, "") in valid_dest_codes]
        if valid_dest_codes else airport_options
    )
    dest_sel = st.selectbox("Aeroporto de destino", dest_opts)
    dest_code_sel = icao_from_label.get(dest_sel, "")

    # 3. Companhia — só as que operam essa rota
    route_key_sel = f"{origin_code_sel}_{dest_code_sel}"
    valid_al_codes = set(route_to_airlines.get(route_key_sel, []))
    airline_opts = (
        [l for l in airline_labels if icao_from_airline.get(l, "") in valid_al_codes]
        if valid_al_codes else airline_labels
    )
    airline_sel = st.selectbox("Companhia aérea", airline_opts)

    c1, c2 = st.columns(2)
    with c1:
        dep_date     = st.date_input("Data de partida", value=date.today(), max_value=date.today() + timedelta(days=15))
        dep_time_val = st.time_input("Hora de partida", value=time(8, 0), step=300)
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        arr_time_val = st.time_input("Hora de chegada", value=time(9, 30), step=300)

    predict_btn = st.button("🔍 Prever Atraso", use_container_width=True, type="primary")

with col_result:
    st.subheader("Resultado")

    if predict_btn:
        if origin_sel == dest_sel:
            st.warning("Origem e destino não podem ser iguais.")
        else:
            airline_code = icao_from_airline[airline_sel]
            origin_code  = icao_from_label[origin_sel]
            dest_code    = icao_from_label[dest_sel]

            dep_dt = datetime.combine(dep_date, dep_time_val)
            arr_dt = datetime.combine(dep_date, arr_time_val)
            if arr_dt <= dep_dt:
                arr_dt += timedelta(days=1)

            dep_str = dep_dt.strftime("%Y-%m-%dT%H:%M:%S")
            arr_str = arr_dt.strftime("%Y-%m-%dT%H:%M:%S")

            with st.spinner("Buscando clima e calculando predição..."):
                try:
                    wx = call_weather(origin_code, dest_code, dep_str)

                    payload = {
                        "airline_icao":     airline_code,
                        "origin_icao":      origin_code,
                        "destination_icao": dest_code,
                        "dep_scheduled":    dep_str,
                        "arr_scheduled":    arr_str,
                        "threshold":        THRESHOLD,
                        "origin_wx_weathercode":       wx.get("origin_wx_weathercode"),
                        "origin_wx_temperature_2m":    wx.get("origin_wx_temperature_2m"),
                        "origin_wx_precipitation":     wx.get("origin_wx_precipitation"),
                        "origin_wx_windspeed_10m":     wx.get("origin_wx_windspeed_10m"),
                        "origin_wx_windgusts_10m":     wx.get("origin_wx_windgusts_10m"),
                        "origin_wx_cloudcover":        wx.get("origin_wx_cloudcover"),
                        "origin_wx_surface_pressure":  wx.get("origin_wx_surface_pressure"),
                        "destination_wx_weathercode":      wx.get("destination_wx_weathercode"),
                        "destination_wx_temperature_2m":   wx.get("destination_wx_temperature_2m"),
                        "destination_wx_precipitation":    wx.get("destination_wx_precipitation"),
                        "destination_wx_windspeed_10m":    wx.get("destination_wx_windspeed_10m"),
                        "destination_wx_windgusts_10m":    wx.get("destination_wx_windgusts_10m"),
                        "destination_wx_cloudcover":       wx.get("destination_wx_cloudcover"),
                        "destination_wx_surface_pressure": wx.get("destination_wx_surface_pressure"),
                    }

                    result = call_predict(payload)
                    st.session_state["result"] = result
                    st.session_state["explanation"] = None  # marca como pendente

                except Exception as exc:
                    show_api_error(exc, context={
                        "airline_icao":     airline_code,
                        "origin_icao":      origin_code,
                        "destination_icao": dest_code,
                        "dep_scheduled":    dep_str,
                    })

    if "result" in st.session_state:
        result = st.session_state["result"]
        proba  = result["delay_proba"]
        icon, label, color = risk_info(proba)

        feats    = result.get("features", {})
        temporal = feats.get("temporal", {})
        route    = feats.get("route", {})
        weather  = feats.get("weather", {})
        lookup   = feats.get("lookup", {})

        st.markdown(
            f"""
            <div style="border-radius:10px; border:1.5px solid {color};
                        padding:18px 20px; margin-bottom:8px;">
                <div style="font-size:22px; font-weight:700; color:{color}">{icon} {label}</div>
                <div style="font-size:13px; color:#64748b; margin-top:3px">
                    Probabilidade de atraso na partida
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(build_gauge(proba), use_container_width=True)
        st.caption("— Limiar de risco: 30%")

        # Explicação LLM — carrega em paralelo ao restante da página
        if st.session_state.get("explanation") is None:
            with st.spinner("🤖 Analisando os principais fatores..."):
                try:
                    st.session_state["explanation"] = call_explain_shap(result)
                except Exception:
                    st.session_state["explanation"] = ""

        explanation = st.session_state.get("explanation", "")
        if explanation:
            st.markdown(
                f"""
                <div style="border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0;
                            padding:16px 18px; margin-top:4px; margin-bottom:8px;
                            font-size:14px; color:#334155; line-height:1.65">
                    🤖 {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

        hist_mean = lookup.get("route_hist_delay_mean")
        if result.get("predicted_delayed") == 1 and hist_mean is not None:
            st.info(f"⏱ Quando essa rota atrasa, o atraso médio histórico é **{hist_mean:.0f} min**")

        with st.expander("Ver detalhes da predição"):
            wx_cols = st.columns(2)

            with wx_cols[0]:
                st.markdown("**🌤 Clima — Origem**")
                cond = weather.get("origin_wx_condition", "")
                st.write(wx_label(weather.get("origin_wx_weathercode")) if not cond else f"**{cond.capitalize()}**")
                st.write(f"🌡 {weather.get('origin_wx_temperature_2m', '—')} °C")
                st.write(f"🌧 Precipitação: {weather.get('origin_wx_precipitation', '—')} mm")
                st.write(f"💨 Vento: {weather.get('origin_wx_windspeed_10m', '—')} km/h")
                st.write(f"🌪 Rajadas: {weather.get('origin_wx_windgusts_10m', '—')} km/h")
                st.write(f"☁️ Cobertura: {weather.get('origin_wx_cloudcover', '—')}%")

            with wx_cols[1]:
                st.markdown("**🌤 Clima — Destino**")
                cond_d = weather.get("destination_wx_condition", "")
                st.write(wx_label(weather.get("destination_wx_weathercode")) if not cond_d else f"**{cond_d.capitalize()}**")
                st.write(f"🌡 {weather.get('destination_wx_temperature_2m', '—')} °C")
                st.write(f"🌧 Precipitação: {weather.get('destination_wx_precipitation', '—')} mm")
                st.write(f"💨 Vento: {weather.get('destination_wx_windspeed_10m', '—')} km/h")
                st.write(f"🌪 Rajadas: {weather.get('destination_wx_windgusts_10m', '—')} km/h")
                st.write(f"☁️ Cobertura: {weather.get('destination_wx_cloudcover', '—')}%")

            st.divider()

            flag_cols = st.columns(4)
            flags = [
                ("🎉 Feriado",         temporal.get("dep_is_holiday", 0)),
                ("⏰ Horário de pico",  temporal.get("dep_is_peak_hour", 0)),
                ("🛣 Rota troncal",    route.get("is_trunk_route", 0)),
                ("📅 Fim de semana",   temporal.get("dep_is_weekend", 0)),
            ]
            for col, (flag_name, val) in zip(flag_cols, flags):
                with col:
                    st.markdown(
                        f"<div style='text-align:center; font-size:13px'>{'✅' if val else '❌'}<br>{flag_name}</div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown("**📍 Rota**")
                st.write(f"Distância: **{route.get('distance_km', 0):.0f} km** ({route.get('flight_range', '—')})")
                st.write(f"Duração prevista: **{route.get('scheduled_duration_min', 0):.0f} min**")
                st.write(f"Diferença de altitude: **{route.get('elevation_diff_ft', 0):.0f} ft**")

            with info_cols[1]:
                st.markdown("**📈 Histórico da rota**")
                hist_rate = lookup.get("route_hist_delay_rate")
                hist_std  = lookup.get("route_hist_delay_std")
                if hist_rate is not None:
                    st.write(f"Taxa histórica de atraso: **{hist_rate*100:.1f}%**")
                if hist_mean is not None:
                    std_str = f" ± {hist_std:.0f} min" if hist_std is not None else ""
                    st.write(f"Atraso médio: **{hist_mean:.0f} min{std_str}**")
                al_rate = lookup.get("airline_hour_delay_rate")
                if al_rate is not None:
                    st.write(f"Taxa companhia nesse horário: **{al_rate*100:.1f}%**")

# ── Melhor janela horária ─────────────────────────────────────────────────────

if "result" in st.session_state:
    origin_code_cur = icao_from_label.get(origin_sel, "")
    dest_code_cur   = icao_from_label.get(dest_sel, "")
    route_key = f"{origin_code_cur}_{dest_code_cur}"

    route_hour_stats = load_route_hour_stats()
    if route_hour_stats is not None:
        route_data = route_hour_stats[route_hour_stats["route"] == route_key].copy()
        if len(route_data) >= 4:
            route_data["faixa"] = (route_data["dep_hour"] // 3 * 3).apply(
                lambda h: f"{h:02d}h–{h+3:02d}h"
            )
            hourly = (
                route_data.groupby("faixa")
                .agg(delay_rate=("delay_rate", "mean"), count=("count", "sum"))
                .reset_index()
            )
            best_slot = hourly.loc[hourly["delay_rate"].idxmin(), "faixa"]

            st.divider()
            st.subheader("⏰ Melhor horário para essa rota")
            st.caption(f"Taxa histórica de atraso por faixa de 3h — melhor janela: **{best_slot}**")
            fig = px.bar(
                hourly, x="faixa", y="delay_rate",
                color="delay_rate",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 0.6],
                text=hourly["delay_rate"].map(lambda x: f"{x*100:.0f}%"),
                labels={"faixa": "Horário de partida", "delay_rate": "Taxa de atraso"},
            )
            fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(t=10, b=10))
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
