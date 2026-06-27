# Dicionário de Dados — Dataset de Voos

**Projeto:** Previsão de Atrasos em Voos Domésticos Brasileiros
**Fontes:** ANAC (VRAA) • OurAirports • Open-Meteo (ERA5 Reanalysis)
**Período:** 2022–2025 • **Granularidade climática:** horária

---

## Identificação do Voo

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `airline_icao` | string | — | Código ICAO da companhia aérea operadora do voo (ex: TAM, GLO, AZU) |
| `origin_icao` | string | — | Código ICAO do aeroporto de origem (ex: SBGR, SBRJ) |
| `destination_icao` | string | — | Código ICAO do aeroporto de destino |

---

## Horários

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `dep_scheduled` | datetime | — | Data e hora programada de partida (ex: 2022-01-06 14:20:00) |
| `dep_actual` | datetime | — | Data e hora real de partida. Pode ser anterior à programada caso o voo adiantou |
| `arr_scheduled` | datetime | — | Data e hora programada de chegada ao destino |
| `arr_actual` | datetime | — | Data e hora real de chegada ao destino |

---

## Aeroporto de Origem
> Fonte: OurAirports

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `origin_type` | string | — | Classificação do aeroporto: `large_airport`, `medium_airport` ou `small_airport` |
| `origin_lat` | float | graus | Latitude geográfica do aeroporto de origem |
| `origin_lon` | float | graus | Longitude geográfica do aeroporto de origem |
| `origin_elevation_ft` | float | pés | Altitude do aeroporto em relação ao nível do mar |
| `origin_region` | string | — | Código ISO da região/estado (ex: BR-SP, BR-RJ) |

---

## Aeroporto de Destino
> Fonte: OurAirports

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `destination_type` | string | — | Classificação do aeroporto: `large_airport`, `medium_airport` ou `small_airport` |
| `destination_lat` | float | graus | Latitude geográfica do aeroporto de destino |
| `destination_lon` | float | graus | Longitude geográfica do aeroporto de destino |
| `destination_elevation_ft` | float | pés | Altitude do aeroporto em relação ao nível do mar |
| `destination_region` | string | — | Código ISO da região/estado (ex: BR-MG, BR-BA) |

---

## Clima no Aeroporto de Origem
> Fonte: Open-Meteo (ERA5) • Horário: partida programada (`dep_scheduled`)

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `origin_wx_temperature_2m` | float | °C | Temperatura do ar a 2m do solo no horário de partida |
| `origin_wx_precipitation` | float | mm | Volume de precipitação (chuva/neve) na hora da partida |
| `origin_wx_windspeed_10m` | float | km/h | Velocidade média do vento a 10m do solo na partida |
| `origin_wx_windgusts_10m` | float | km/h | Rajada máxima de vento a 10m do solo. Mais crítica que a velocidade média para operações |
| `origin_wx_cloudcover` | float | % | Percentual do céu coberto por nuvens (0 = céu limpo, 100 = totalmente nublado) |
| `origin_wx_weathercode` | int | WMO | Código de condição climática (ver tabela abaixo) |
| `origin_wx_surface_pressure` | float | hPa | Pressão atmosférica ao nível da superfície. Quedas bruscas indicam frentes de mau tempo |

---

## Clima no Aeroporto de Destino
> Fonte: Open-Meteo (ERA5) • Horário: chegada programada (`arr_scheduled`)

| Coluna | Tipo | Unidade | Descrição |
|---|---|---|---|
| `destination_wx_temperature_2m` | float | °C | Temperatura do ar a 2m do solo no horário previsto de chegada |
| `destination_wx_precipitation` | float | mm | Volume de precipitação na hora prevista de chegada |
| `destination_wx_windspeed_10m` | float | km/h | Velocidade média do vento a 10m do solo na chegada |
| `destination_wx_windgusts_10m` | float | km/h | Rajada máxima de vento a 10m do solo na chegada |
| `destination_wx_cloudcover` | float | % | Percentual do céu coberto por nuvens no destino na chegada |
| `destination_wx_weathercode` | int | WMO | Código de condição climática (ver tabela abaixo) |
| `destination_wx_surface_pressure` | float | hPa | Pressão atmosférica ao nível da superfície no destino na chegada |

---

## Referência — Códigos WMO

| Código | Condição |
|---|---|
| 0 | Céu limpo |
| 1–3 | Parcialmente nublado |
| 45–48 | Neblina |
| 51–67 | Chuva (leve a forte) |
| 71–77 | Neve |
| 80–82 | Pancadas de chuva |
| 95–99 | Tempestade |

---

## Observações

- O clima é obtido via reanálise ERA5 do Open-Meteo com granularidade horária. O join é feito truncando `dep_scheduled` e `arr_scheduled` para a hora cheia mais próxima.
- `origin_elevation_ft` e `destination_elevation_ft` estão em pés. Para converter para metros: `ft × 0,3048`.
- Linhas sem coordenadas de aeroporto (~0,3% do dataset original) foram removidas antes do join climático.
