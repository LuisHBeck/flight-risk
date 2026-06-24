# 📋 Dicionário de Dados — Dataset de Treino

Documentação de todas as colunas presentes no arquivo `flights_features.parquet`, gerado pelo script `feature_engineering.py` a partir dos dados brutos da ANAC combinados com dados climáticos da API Open-Meteo.

---

## 🎯 Target

| Coluna | Tipo | Descrição |
|---|---|---|
| `is_delayed` | int (0/1) | **Variável alvo.** `1` se o voo atrasou mais de 15 minutos na partida, `0` caso contrário. O limiar de 15 minutos segue o critério oficial da ANAC. |

---

## ✈️ Identificação do Voo

| Coluna | Tipo | Descrição |
|---|---|---|
| `airline_icao` | str | Código ICAO da companhia aérea (ex: `TAM`, `GLO`, `AZU`). |
| `origin_icao` | str | Código ICAO do aeroporto de origem (ex: `SBRJ`, `SBGR`). |
| `destination_icao` | str | Código ICAO do aeroporto de destino. |

---

## 🕐 Features Temporais

Extraídas do horário de partida programado (`dep_scheduled`). As versões cíclicas (sin/cos) permitem que o modelo entenda a continuidade do tempo — por exemplo, que 23h e 0h são horários próximos, e que dezembro e janeiro são meses adjacentes.

| Coluna | Tipo | Descrição |
|---|---|---|
| `dep_day_of_year` | int | Dia do ano da partida programada (1–366). Captura sazonalidade anual — verão (jan/fev) e julho concentram mais atrasos no Brasil. |
| `dep_is_weekend` | int (0/1) | `1` se a partida ocorre no sábado ou domingo. Fins de semana têm padrão operacional diferente dos dias úteis. |
| `dep_hour_sin` | float | Componente seno da hora de partida. Encoding cíclico da hora do dia (0–23). |
| `dep_hour_cos` | float | Componente cosseno da hora de partida. Encoding cíclico da hora do dia (0–23). |
| `dep_dow_sin` | float | Componente seno do dia da semana. Encoding cíclico (0 = segunda, 6 = domingo). |
| `dep_dow_cos` | float | Componente cosseno do dia da semana. Encoding cíclico (0 = segunda, 6 = domingo). |
| `dep_month_sin` | float | Componente seno do mês de partida. Encoding cíclico (1–12). |
| `dep_month_cos` | float | Componente cosseno do mês de partida. Encoding cíclico (1–12). |
| `dep_time_block` | str | Bloco do dia da partida: `early_morning` (0–5h), `morning` (6–11h), `afternoon` (12–17h), `evening` (18–23h). |
| `dep_is_peak_hour` | int (0/1) | `1` se a partida ocorre em horário de pico dos aeroportos brasileiros (7–9h ou 17–20h). |
| `dep_is_holiday` | int (0/1) | `1` se a data de partida coincide com um feriado nacional brasileiro. |

---

## 🛫 Features de Rota e Aeroporto

| Coluna | Tipo | Descrição |
|---|---|---|
| `origin_region` | str | Região/estado do aeroporto de origem no formato `BR-XX` (ex: `BR-SP`, `BR-RJ`). |
| `destination_region` | str | Região/estado do aeroporto de destino no formato `BR-XX`. |
| `origin_elevation_ft` | float | Altitude do aeroporto de origem em pés. Aeroportos em altitude elevada têm restrições operacionais específicas. |
| `destination_elevation_ft` | float | Altitude do aeroporto de destino em pés. |
| `route` | str | Rota no formato `ORIGEM_DESTINO` (ex: `SBRJ_SBGR`). Captura o perfil histórico de pontualidade de cada par de aeroportos. |
| `region_pair` | str | Par de regiões no formato `BR-XX_BR-YY` (ex: `BR-RJ_BR-SP`). Captura padrões regionais sem depender de rotas específicas. |
| `distance_km` | float | Distância em quilômetros entre origem e destino, calculada pela fórmula de Haversine usando as coordenadas geográficas. |
| `flight_range` | str | Classificação do voo por distância: `short` (< 500 km), `medium` (500–1500 km), `long` (> 1500 km). |
| `elevation_diff_ft` | float | Diferença de altitude entre destino e origem em pés (`destination_elevation_ft - origin_elevation_ft`). Valores positivos indicam que o destino é mais alto. |
| `origin_airport_size` | int | Tamanho do aeroporto de origem em escala ordinal: `1` = pequeno, `2` = médio, `3` = grande. |
| `destination_airport_size` | int | Tamanho do aeroporto de destino em escala ordinal: `1` = pequeno, `2` = médio, `3` = grande. |
| `is_trunk_route` | int (0/1) | `1` se a rota é considerada troncal de alta frequência no Brasil (ex: ponte aérea RJ-SP, SP-Brasília, SP-Salvador, etc.). |

---

## 🚦 Features de Congestionamento

Calculadas a partir do próprio dataset, contando quantos voos operam no mesmo aeroporto na mesma hora. São proxies de pressão operacional de pátio sem necessidade de dados externos.

| Coluna | Tipo | Descrição |
|---|---|---|
| `origin_hourly_flights` | int | Número de voos que partem do mesmo aeroporto de origem na mesma hora do mesmo dia. Quanto maior, maior a pressão operacional no pátio de partida. |
| `destination_hourly_arrivals` | int | Número de voos que chegam ao aeroporto de destino na mesma hora do mesmo dia. Alta taxa de chegadas pode gerar fila e atrasar a liberação de gates. |
| `total_hourly_congestion` | int | Soma de `origin_hourly_flights` e `destination_hourly_arrivals`. Índice geral de pressão operacional combinada nos dois aeroportos do voo. |

---

## 🌦️ Features Climáticas — Dados Brutos

Dados horários coletados da API Open-Meteo para o momento do voo, tanto na origem quanto no destino.

### Origem

| Coluna | Tipo | Descrição |
|---|---|---|
| `origin_wx_temperature_2m` | float | Temperatura do ar a 2 metros de altura na origem (°C). Temperaturas extremas afetam desempenho das aeronaves. |
| `origin_wx_precipitation` | float | Precipitação acumulada na hora anterior na origem (mm). |
| `origin_wx_windspeed_10m` | float | Velocidade do vento a 10 metros de altura na origem (km/h). |
| `origin_wx_windgusts_10m` | float | Velocidade máxima das rajadas de vento a 10 metros na origem (km/h). Rajadas fortes podem impedir decolagens. |
| `origin_wx_cloudcover` | float | Cobertura de nuvens na origem (%). `100%` = céu completamente encoberto. |
| `origin_wx_surface_pressure` | float | Pressão atmosférica ao nível da superfície na origem (hPa). Variações bruscas indicam passagem de frentes. |

### Destino

| Coluna | Tipo | Descrição |
|---|---|---|
| `destination_wx_temperature_2m` | float | Temperatura do ar a 2 metros de altura no destino (°C). |
| `destination_wx_precipitation` | float | Precipitação acumulada na hora anterior no destino (mm). |
| `destination_wx_windspeed_10m` | float | Velocidade do vento a 10 metros de altura no destino (km/h). |
| `destination_wx_windgusts_10m` | float | Velocidade máxima das rajadas de vento a 10 metros no destino (km/h). |
| `destination_wx_cloudcover` | float | Cobertura de nuvens no destino (%). |
| `destination_wx_surface_pressure` | float | Pressão atmosférica ao nível da superfície no destino (hPa). |

---

## 🌩️ Features Climáticas — Derivadas

Derivadas do campo `weathercode` da Open-Meteo, que classifica a condição meteorológica predominante em cada hora.

| Coluna | Tipo | Descrição |
|---|---|---|
| `origin_wx_condition` | str | Condição climática na origem: `clear`, `cloudy`, `fog`, `rain`, `snow`, `showers`, `storm`, `other`, `unknown`. |
| `destination_wx_condition` | str | Condição climática no destino (mesma escala acima). |
| `origin_wx_is_fog` | int (0/1) | `1` se há neblina ou nevoeiro na origem (weathercodes 45 ou 48). Neblina causa restrição de visibilidade e pode fechar aeroportos. |
| `origin_wx_is_rain` | int (0/1) | `1` se há chuva ou pancadas na origem (weathercodes 51–67 ou 80–82). |
| `origin_wx_is_storm` | int (0/1) | `1` se há tempestade na origem (weathercodes 95–99). Tempestades são a condição climática mais impactante para operações aéreas. |
| `destination_wx_is_fog` | int (0/1) | `1` se há neblina ou nevoeiro no destino. |
| `destination_wx_is_rain` | int (0/1) | `1` se há chuva ou pancadas no destino. |
| `destination_wx_is_storm` | int (0/1) | `1` se há tempestade no destino. |

---

## 📈 Features de Histórico por Rota

Calculadas sobre o dataset completo antes do split treino/teste e salvas em `route_stats.pkl` para uso em produção. Para rotas não vistas no treino, o valor padrão é a média global de cada estatística.

| Coluna | Tipo | Descrição |
|---|---|---|
| `route_hist_delay_mean` | float | Média histórica de atraso na partida (em minutos) para aquela rota no dataset completo. |
| `route_hist_delay_std` | float | Desvio padrão do atraso na partida para aquela rota. Rotas com alto desvio são mais imprevisíveis operacionalmente. |
| `route_hist_delay_rate` | float | Proporção histórica de voos atrasados (> 15 min) naquela rota. Valor entre 0 e 1. |

---

## 📈 Features de Histórico por Companhia × Hora

Calculadas sobre o dataset completo e salvas em `airline_hour_stats.pkl` para uso em produção. Capturam o perfil de pontualidade de cada companhia em cada horário do dia.

| Coluna | Tipo | Descrição |
|---|---|---|
| `airline_hour_delay_rate` | float | Proporção histórica de voos atrasados (> 15 min) para aquela companhia naquele horário. Valor entre 0 e 1. |
| `airline_hour_delay_mean` | float | Média histórica de atraso na partida (em minutos) para aquela companhia naquele horário. |

---


## 🗂️ Arquivos auxiliares

| Arquivo | Descrição |
|---|---|
| `flights_features.parquet` | Dataset de treino completo com todas as colunas abaixo. |
| `route_stats.pkl` | Lookup table indexada por `route` com as 3 features históricas de rota. Usada em produção para enriquecer novos voos sem reprocessar o dataset completo. |
| `airline_hour_stats.pkl` | Lookup table indexada por `(airline_icao, dep_hour)` com as 2 features históricas de companhia×hora. |

---

## 🗓️ Coluna de metadado temporal

| Coluna | Tipo | Descrição |
|---|---|---|
| `dep_scheduled` | datetime64 | Horário de partida programado. **Não é uma feature de treino** — é mantida no parquet exclusivamente como metadado para realizar o split temporal correto (treino/validação/teste por ano). Deve ser excluída do `X` antes de treinar o modelo. |

---

## 🚀 Como consumir os dados para treinamento

### Carregando o dataset

```python
import pandas as pd
import joblib

# Dataset pronto para treino
df = pd.read_parquet(".data/flights_features.parquet")

# Tabelas auxiliares (para uso em produção / inferência)
route_stats        = joblib.load(".data/route_stats.pkl")
airline_hour_stats = joblib.load(".data/airline_hour_stats.pkl")
```

### Separando features e target

```python
TARGET   = "is_delayed"
METADATA = ["dep_scheduled"]  # apenas para o split — não entra no modelo

# Colunas categóricas que precisam de encoding antes do treino
CAT_COLS = [
    "airline_icao", "origin_icao", "destination_icao",
    "origin_region", "destination_region",
    "route", "region_pair", "flight_range",
    "dep_time_block",
    "origin_wx_condition", "destination_wx_condition",
]

X = df.drop(columns=[TARGET] + METADATA)
y = df[TARGET]
```

### Encoding das colunas categóricas

O LightGBM aceita inteiros, então as colunas categóricas precisam ser convertidas via `LabelEncoder`. Os encoders devem ser salvos para garantir que a mesma transformação seja aplicada em produção.

```python
from sklearn.preprocessing import LabelEncoder
import joblib

encoders = {}
for col in CAT_COLS:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

joblib.dump(encoders, ".models/encoders.pkl")
```

---

## ✂️ Como fazer o split correto

### ⚠️ Por que NÃO usar split aleatório

Este dataset possui **dependência temporal** — voos de 2022 influenciam padrões de 2023, e o modelo deve aprender com o passado para prever o futuro. Um split aleatório (`shuffle=True`) causaria **data leakage temporal**: o modelo veria dados de dezembro de 2024 durante o treino e seria avaliado em janeiro de 2024, inflando artificialmente as métricas.

### ✅ Split correto: por ano completo

O dataset cobre 4 anos completos (2022–2025). A divisão recomendada usa anos inteiros para garantir que treino e teste tenham a mesma sazonalidade representada (verão, inverno, feriados).

```
Treino:    2022 + 2023 + 2024  → ~74% dos dados
Validação: 2025 jan–jun        → ~13% dos dados  (usado para early stopping)
Teste:     2025 jul–dez        → ~13% dos dados  (avaliação final, nunca visto durante treino)
```

```python
VAL_CUTOFF = "2025-07-01"

dates = df["dep_scheduled"]

train_mask = dates.dt.year.isin([2022, 2023, 2024])
val_mask   = (~train_mask) & (dates < VAL_CUTOFF)
test_mask  = (~train_mask) & (dates >= VAL_CUTOFF)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"Treino:    {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
print(f"Validação: {len(X_val):,} ({100*len(X_val)/len(X):.1f}%)")
print(f"Teste:     {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")
```

### Distribuição esperada após o split

| Conjunto | Período | Voos | Taxa de atraso |
|---|---|---|---|
| Treino | 2022–2024 | ~2,19M | ~15,0% |
| Validação | jan–jun 2025 | ~380k | ~13,6% |
| Teste | jul–dez 2025 | ~391k | ~16,1% |

A diferença na taxa de atraso entre os conjuntos é esperada e reflete a sazonalidade real: o segundo semestre (jul–dez) concentra os meses de maior movimento aéreo no Brasil (julho de férias e novembro–dezembro de festas de fim de ano).

---
