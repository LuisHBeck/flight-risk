import json
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE   = os.getenv("API_BASE_URL", "https://flight-risk-430527068071.us-central1.run.app")
THRESHOLD  = 0.3
MODEL_DIR  = Path(__file__).parent.parent.parent / "model"
DATA_DIR   = Path(__file__).parent.parent.parent / ".data"

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
ALL_AIRLINES = sorted(AIRLINE_NAMES.keys())

airline_labels   = [f"{k} — {AIRLINE_NAMES[k]}" for k in ALL_AIRLINES]
icao_from_airline = {f"{k} — {AIRLINE_NAMES[k]}": k for k in ALL_AIRLINES}


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


@st.cache_data
def load_airports():
    df = pd.read_csv(DATA_DIR / "airports_reference.csv")
    df = df.dropna(subset=["ident"])
    df["label"] = df.apply(
        lambda r: f"{r['ident']} — {r['municipality']}" if str(r.get("municipality", "")) not in ("", "nan") else r["ident"],
        axis=1,
    )
    return df.sort_values("municipality").reset_index(drop=True)


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


def risk_info(proba):
    if proba < THRESHOLD:
        return "🟢", "Baixo risco", "#16a34a"
    elif proba < 0.6:
        return "🟡", "Risco moderado", "#d97706"
    else:
        return "🔴", "Alto risco", "#dc2626"


def build_gauge(proba):
    _, _, color = risk_info(proba)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 30],  "color": "#f0fdf4"},
                {"range": [30, 60], "color": "#fefce8"},
                {"range": [60, 100],"color": "#fef2f2"},
            ],
            "threshold": {
                "line": {"color": "#475569", "width": 3},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=10, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def log_missing_request(context: dict):
    try:
        log_file = DATA_DIR / "missing_requests.jsonl"
        entry = {"timestamp": datetime.now().isoformat(), **context}
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def friendly_error_msg(exc, context: dict | None = None) -> str:
    is_no_data = (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response.status_code in (404, 422, 400)
    )
    is_offline = isinstance(exc, requests.exceptions.ConnectionError)

    if is_no_data:
        if context:
            log_missing_request(context)
        return "🗺️ Sem dados para essa combinação — consulta registrada!"
    elif is_offline:
        return "🛰️ Serviço fora do ar — tente em instantes."
    else:
        return "⚠️ Erro inesperado — tente novamente."


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("⚖️ Comparar Voos")
st.caption("Compare até 3 opções de voo e escolha o menor risco de atraso")
st.divider()

airports        = load_airports()
airport_options = airports["label"].tolist()
icao_from_label = dict(zip(airports["label"], airports["ident"]))
origin_to_dests, route_to_airlines = load_route_combos()

# ── Rota e data compartilhadas ────────────────────────────────────────────────

st.subheader("Rota")
col_orig, col_dest, col_date = st.columns([2, 2, 1])
with col_orig:
    origin_sel  = st.selectbox("Aeroporto de origem", airport_options, key="cmp_origin")
    origin_code_cmp = icao_from_label.get(origin_sel, "")
with col_dest:
    valid_dest_codes = set(origin_to_dests.get(origin_code_cmp, []))
    dest_opts = (
        [opt for opt in airport_options if icao_from_label.get(opt, "") in valid_dest_codes]
        if valid_dest_codes else airport_options
    )
    dest_sel = st.selectbox("Aeroporto de destino", dest_opts, key="cmp_dest")
with col_date:
    dep_date = st.date_input("Data de partida", value=date.today(), max_value=date.today() + timedelta(days=15), key="cmp_date")

st.divider()

# ── Inputs por voo ────────────────────────────────────────────────────────────

st.subheader("Opções de voo")

NUM_FLIGHTS = 3
flight_inputs = []

dest_code_cmp   = icao_from_label.get(dest_sel, "")
route_key_cmp   = f"{origin_code_cmp}_{dest_code_cmp}"
valid_al_codes  = set(route_to_airlines.get(route_key_cmp, []))
avail_al_labels = (
    [l for l in airline_labels if icao_from_airline.get(l, "") in valid_al_codes]
    if valid_al_codes else airline_labels
)

DEFAULTS = ["TAM", "AZU", "GLO"]

cols = st.columns(NUM_FLIGHTS, gap="medium")
for i, col in enumerate(cols):
    with col:
        st.markdown(f"**Voo {i+1}**")
        default_code  = DEFAULTS[i]
        default_label = f"{default_code} — {AIRLINE_NAMES[default_code]}"
        default_idx   = (
            avail_al_labels.index(default_label)
            if default_label in avail_al_labels
            else 0
        )
        al = st.selectbox("Companhia", avail_al_labels, key=f"al_{i}", index=default_idx)
        dep = st.time_input("Saída", value=time(7 + i * 3, 0), step=300, key=f"dep_{i}")
        arr = st.time_input("Chegada", value=time(8 + i * 3, 30), step=300, key=f"arr_{i}")
        active = st.checkbox("Incluir na comparação", value=True, key=f"active_{i}")
        flight_inputs.append({"airline": al, "dep": dep, "arr": arr, "active": active})

st.divider()
compare_btn = st.button("⚖️ Comparar voos", use_container_width=True, type="primary")

# ── Resultado ─────────────────────────────────────────────────────────────────

if compare_btn:
    if origin_sel == dest_sel:
        st.warning("Origem e destino não podem ser iguais.")
        st.stop()

    origin_code = icao_from_label[origin_sel]
    dest_code   = icao_from_label[dest_sel]
    active_flights = [f for f in flight_inputs if f["active"]]

    if len(active_flights) < 2:
        st.warning("Ative pelo menos 2 voos para comparar.")
        st.stop()

    results = []
    with st.spinner("Calculando risco para cada voo..."):
        for i, f in enumerate(active_flights):
            dep_dt = datetime.combine(dep_date, f["dep"])
            arr_dt = datetime.combine(dep_date, f["arr"])
            if arr_dt <= dep_dt:
                arr_dt += timedelta(days=1)
            dep_str = dep_dt.strftime("%Y-%m-%dT%H:%M:%S")
            arr_str = arr_dt.strftime("%Y-%m-%dT%H:%M:%S")
            airline_code = icao_from_airline[f["airline"]]

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
                res = call_predict(payload)
                results.append({**f, "result": res, "dep_str": dep_str, "error": None})
            except Exception as exc:
                msg = friendly_error_msg(exc, context={
                    "airline_icao":     airline_code,
                    "origin_icao":      origin_code,
                    "destination_icao": dest_code,
                    "dep_scheduled":    dep_str,
                })
                results.append({**f, "result": None, "dep_str": dep_str, "error": msg})

    st.session_state["cmp_results"] = results

if "cmp_results" in st.session_state:
    results = st.session_state["cmp_results"]
    valid   = [r for r in results if r["result"] is not None]

    if not valid:
        st.error("Nenhuma predição foi obtida. Verifique se a API está rodando.")
        st.stop()

    best_idx = min(range(len(valid)), key=lambda i: valid[i]["result"]["delay_proba"])

    st.subheader("Resultado da comparação")
    result_cols = st.columns(len(valid), gap="large")

    for idx, (col, r) in enumerate(zip(result_cols, valid)):
        proba        = r["result"]["delay_proba"]
        icon, label, color = risk_info(proba)
        is_best      = (r is valid[best_idx])
        airline_code = icao_from_airline[r["airline"]]

        with col:
            st.markdown(
                f"<div style='text-align:center; font-size:17px; font-weight:700'>"
                f"{airline_code} — {r['dep'].strftime('%H:%M')}</div>"
                f"<div style='text-align:center; font-size:13px; color:#64748b'>"
                f"{AIRLINE_NAMES.get(airline_code, airline_code)}</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_gauge(proba), use_container_width=True, key=f"gauge_{idx}")
            st.markdown(
                f"<div style='text-align:center; font-size:16px; color:{color}'>"
                f"{icon} {label}</div>",
                unsafe_allow_html=True,
            )

            lookup = r["result"].get("features", {}).get("lookup", {})
            hist_rate = lookup.get("route_hist_delay_rate")
            al_rate   = lookup.get("airline_hour_delay_rate")
            if hist_rate is not None:
                st.caption(f"Rota historicamente: {hist_rate*100:.1f}%")
            if al_rate is not None:
                st.caption(f"Companhia nesse horário: {al_rate*100:.1f}%")

            if is_best:
                st.markdown(
                    f"<div style='background:{color}18; border:2px solid {color}; "
                    f"border-radius:10px; padding:10px; text-align:center; margin-top:10px'>"
                    f"<b style='color:{color}'>✅ Melhor opção</b></div>",
                    unsafe_allow_html=True,
                )
