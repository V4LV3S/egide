from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ---------- CONFIG ----------
RAW_DIR, PROCESSED_DIR, TRAIN_DIR = Path("data/raw"), Path("data/processed"), Path("data/training")
EOLIC_DIR = RAW_DIR / "constrained_off_eolico"
LOAD_DIR = RAW_DIR / "curva_carga"
INTERCHANGE_DIR = RAW_DIR / "intercambio_subsistemas"
STATE_HOURLY_PATH = PROCESSED_DIR / "state_hourly.parquet"
SEQUENCE_DIR = TRAIN_DIR / "lstm_cnn"

CSV_KW = {"sep": ";", "encoding": "utf-8-sig"}
CHUNKSIZE = 500_000
SEQUENCE_LENGTH = 168  # uma semana de histórico por amostra

# Horizonte de previsão em horas. Os lags do alvo são deslocados por FORECAST_HORIZON,
# então a janela que termina em t só vê curtailment observado até t - FORECAST_HORIZON.
# 1 = previsão da próxima hora; 24 = day-ahead (nesse caso reveja também as features
# exógenas contemporâneas, que hoje entram com o valor verificado da própria hora t).
FORECAST_HORIZON = 1
ROLLING_WINDOWS = (24,)

# Intervalo final do state_hourly. Fora dele, o pipeline levanta erro se faltar dado.
START_TIME = pd.Timestamp("2023-10-01 00:00:00")
END_TIME = pd.Timestamp("2026-08-31 23:00:00")

# Só o subsistema Nordeste entra no pipeline; os demais (ex.: MG, no Sudeste) são descartados.
SUBSYSTEM = "NE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------- COLUNAS ----------
EOLIC_COLUMNS = [
    "id_subsistema", "nom_subsistema", "id_estado", "nom_estado", "id_ons", "din_instante",
    "val_geracaoreferencia", "val_geracaonaorealizadaapurada", "cod_razaorestricao",
    "cod_origemrestricao", "num_minutos_rel", "num_minutos_cnf", "num_minutos_ene",
    "num_minutos_restricao",
]
MINUTE_COLS = ["num_minutos_rel", "num_minutos_cnf", "num_minutos_ene", "num_minutos_restricao"]
EOLIC_NUMERIC = ["val_geracaoreferencia", "val_geracaonaorealizadaapurada", *MINUTE_COLS]
LOAD_COLUMNS = ["id_subsistema", "din_instante", "val_cargaenergiahomwmed"]
INTERCHANGE_COLUMNS = ["din_instante", "id_subsistema_origem", "id_subsistema_destino", "val_intercambiomwmed"]

# Códigos como aparecem no dado bruto (maiúsculos); as colunas derivadas ficam em snake_case minúsculo.
# PAR foi removido do escopo do modelo (target/loss/métricas usam só REL, CNF e ENE).
REASON_CODES, ORIGIN_CODES = ["REL", "CNF", "ENE"], ["LOC", "SIS"]

# Unidades espaciais esperadas no Nordeste: BA e SE ficam separados (não unir em BA_SE).
EXPECTED_STATES = {"BA",  "CE", "PB", "PE", "PI", "RN"}
REASON_COLS = [f"reason_{c.lower()}" for c in REASON_CODES]
ORIGIN_COLS = [f"origin_{c.lower()}" for c in ORIGIN_CODES]
FLAG_COLS = REASON_COLS + ORIGIN_COLS
FLAG_MAX = {c: (c, "max") for c in FLAG_COLS}

STATE_ID_KEYS = ["id_estado", "nom_estado", "id_subsistema", "nom_subsistema"]
STATE_KEYS = ["din_instante", *STATE_ID_KEYS]
ENTITY_KEYS = [*STATE_KEYS, "id_ons"]

TIME_FEATURES = [
    "hour_sin", "hour_cos", "week_day_sin", "week_day_cos",
    "month_day_sin", "month_day_cos", "month_sin", "month_cos",
]

# ---------- LAGS DO ALVO (autorregressivos) ----------
# Sem eles o modelo nunca vê o histórico de curtailment, só as exógenas.
LAG_BASE_COLUMNS = [
    "curtailment_mwmed", "curtailment_ratio", "curtailment_flag", "curtailment_minutes",
    *REASON_COLS, *ORIGIN_COLS,
]
LAG_PREFIX = f"lag{FORECAST_HORIZON}_"
LAG_COLUMNS = [f"{LAG_PREFIX}{c}" for c in LAG_BASE_COLUMNS]
ROLLING_COLUMNS = [f"roll{w}_{c}" for w in ROLLING_WINDOWS for c in ("curtailment_flag", "curtailment_ratio")]
NE_LAG_COLUMN = f"{LAG_PREFIX}ne_curtailment_ratio"
LAG_FEATURE_COLUMNS = [*LAG_COLUMNS, *ROLLING_COLUMNS, NE_LAG_COLUMN]

FEATURE_PREFIXES = ("gerreneol", "load_", "interchange_verified_", "lag", "roll")
FLOAT_PREFIXES = ("gerreneol", "curtailment", "load_", "interchange_", "minutes_", "lag", "roll")

# Pares origem_destino do subsistema Nordeste presentes no dataset final.
INTERCHANGE_PAIRS = ["ne_n", "ne_se", "n_ne", "n_se", "se_n", "se_ne", "se_s", "s_se"]
INTERCHANGE_COLUMNS_FINAL = [f"interchange_verified_{p}_mwmed" for p in INTERCHANGE_PAIRS]

# ---------- AGREGAÇÃO SEMI-HORÁRIA (saída=(coluna, função)) ----------
ENTITY_SPEC = {
    "val_geracaoreferencia": ("val_geracaoreferencia", "mean"),
    "val_geracaonaorealizadaapurada": ("val_geracaonaorealizadaapurada", "mean"),
    **{c: (c, "max") for c in MINUTE_COLS}, **FLAG_MAX,
}
STATE_SPEC = {
    "geracao_referencia_halfhour": ("val_geracaoreferencia", "sum"),
    "gnr_halfhour": ("val_geracaonaorealizadaapurada", "sum"),
    "minutes_rel_halfhour": ("num_minutos_rel", "sum"),
    "minutes_cnf_halfhour": ("num_minutos_cnf", "sum"),
    "minutes_ene_halfhour": ("num_minutos_ene", "sum"),
    "minutes_restricao_max_halfhour": ("num_minutos_restricao", "max"),
    "minutes_restricao_sum_halfhour": ("num_minutos_restricao", "sum"),
    **FLAG_MAX,
}
# Reagrupa chunks que possam ter dividido o mesmo grupo.
MERGE_SPEC = {k: (k, "sum" if fn == "sum" else "max") for k, (_, fn) in STATE_SPEC.items()}

# Coluna final de hora -> (coluna semi-horária de origem, forma de combinar as duas meias-horas).
HOURLY_COMBINE = {
    "gerreneol_mw": ("geracao_referencia_halfhour", "mean"),
    "curtailment_mwmed": ("gnr_halfhour", "mean"),
    "curtailment_minutes": ("minutes_restricao_max_halfhour", "sum"),
    "curtailment_minutes_set_sum": ("minutes_restricao_sum_halfhour", "sum"),
    "minutes_rel": ("minutes_rel_halfhour", "sum"),
    "minutes_cnf": ("minutes_cnf_halfhour", "sum"),
    "minutes_ene": ("minutes_ene_halfhour", "sum"),
    **{c: (c, "max") for c in FLAG_COLS},
}

# ---------- HELPERS ----------
def agg(df: pd.DataFrame, keys: list[str], spec: dict) -> pd.DataFrame:
    return df.groupby(keys, as_index=False, observed=True).agg(**spec)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0, **CSV_KW).columns.tolist()


def read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Lê só as colunas disponíveis, avisando as ausentes."""
    available = set(read_header(path))
    if missing := [c for c in usecols if c not in available]:
        logger.warning("%s | colunas ausentes: %s", path.name, missing)
    return pd.read_csv(path, usecols=[c for c in usecols if c in available], low_memory=False, **CSV_KW)


def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def to_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def list_csvs(directory: Path) -> list[Path]:
    if not (files := sorted(directory.rglob("*.csv"))):
        raise FileNotFoundError(f"Nenhum CSV encontrado em {directory}")
    return files


def read_many(directory: Path, columns: list[str], numeric: list[str]) -> pd.DataFrame:
    """Lê e empilha todos os CSVs de um diretório, com `datetime` normalizado."""
    frames = []
    for path in list_csvs(directory):
        logger.info("Lendo: %s", path)
        df = read_csv(path, columns)
        df["datetime"] = to_dt(df.pop("din_instante"))
        to_num(df, numeric)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ---------- EÓLICO: 30 MIN × ENTIDADE -> 30 MIN × ESTADO ----------
def prepare_eolic_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filtra o subsistema, normaliza tipos e cria as flags de razão/origem da restrição."""
    chunk = chunk[chunk["id_subsistema"].eq(SUBSYSTEM)].copy()
    chunk["din_instante"] = to_dt(chunk["din_instante"])
    to_num(chunk, EOLIC_NUMERIC)
    # O ONS deixa os minutos em branco quando não houve restrição daquele tipo (não é dado
    # ausente, é "0 minutos"); sem isso o NaN se propaga pelas agregações e vira NaN em
    # curtailment_minutes mesmo em horas com geração completa e sem restrição alguma.
    chunk[MINUTE_COLS] = chunk[MINUTE_COLS].fillna(0.0)
    for prefix, col, codes in (("reason", "cod_razaorestricao", REASON_CODES),
                               ("origin", "cod_origemrestricao", ORIGIN_CODES)):
        for code in codes:
            chunk[f"{prefix}_{code.lower()}"] = chunk[col].eq(code).astype("int8")
    return chunk


def aggregate_eolic_file(path: Path) -> pd.DataFrame:
    """Arquivo mensal: 30min×entidade → 30min×estado."""
    logger.info("Processando eólico: %s", path)
    if missing := set(EOLIC_COLUMNS) - set(read_header(path)):
        raise ValueError(f"{path} sem colunas obrigatórias: {sorted(missing)}")
    parts = [
        agg(agg(prepare_eolic_chunk(chunk), ENTITY_KEYS, ENTITY_SPEC), STATE_KEYS, STATE_SPEC)
        for chunk in pd.read_csv(path, usecols=EOLIC_COLUMNS, chunksize=CHUNKSIZE, low_memory=False, **CSV_KW)
    ]
    return agg(pd.concat(parts, ignore_index=True), STATE_KEYS, MERGE_SPEC)


def combine_halfhour_files(files: list[Path]) -> pd.DataFrame:
    """Concatena os meses e reagrupa instantes que caíram em arquivos diferentes."""
    halfhour = pd.concat([aggregate_eolic_file(p) for p in files], ignore_index=True)
    return (halfhour.sort_values(["id_estado", "din_instante"])
                     .drop_duplicates(["din_instante", "id_estado"], keep="last")
                     .reset_index(drop=True))

# ---------- EÓLICO: 30 MIN × ESTADO -> HORA × ESTADO ----------
def combine_halfhours(previous: pd.Series, current: pd.Series, how: str) -> pd.Series:
    """Combina as duas meias-horas de uma hora cheia; NaN se qualquer uma faltar."""
    result = {
        "mean": lambda p, c: (p + c) / 2.0,
        "sum": lambda p, c: p + c,
        "max": lambda p, c: pd.concat([p, c], axis=1).max(axis=1),
    }[how](previous, current)
    return result.where(previous.notna() & current.notna(), np.nan)


def build_eolic_hourly(halfhour: pd.DataFrame) -> pd.DataFrame:
    """Converte o eólico semi-horário em horário, exigindo as duas meias-horas de cada hora."""
    logger.info("Convertendo eólico de 30 min para hora...")

    states = halfhour[STATE_ID_KEYS].drop_duplicates("id_estado").sort_values("id_estado").reset_index(drop=True)
    hours = pd.date_range(START_TIME, END_TIME, freq="h")
    grid = states.merge(pd.DataFrame({"datetime": hours}), how="cross")

    value_cols = sorted({src for src, _ in HOURLY_COMBINE.values()})
    halfhour = halfhour.rename(columns={"din_instante": "datetime"})

    current = halfhour[[*STATE_ID_KEYS, "datetime", *value_cols]].rename(
        columns={c: f"{c}_current" for c in value_cols})
    previous = halfhour[[*STATE_ID_KEYS, "datetime", *value_cols]].assign(
        datetime=lambda d: d["datetime"] + pd.Timedelta(minutes=30)).rename(
        columns={c: f"{c}_previous" for c in value_cols})

    hourly = grid.merge(current, on=[*STATE_ID_KEYS, "datetime"], how="left") \
                 .merge(previous, on=[*STATE_ID_KEYS, "datetime"], how="left")

    # A geração de referência define se a hora está completa; ela nunca falta sozinha.
    complete = hourly["geracao_referencia_halfhour_current"].notna() & \
               hourly["geracao_referencia_halfhour_previous"].notna()
    if incomplete := int((~complete).sum()):
        sample = hourly.loc[~complete, ["id_estado", "datetime"]].head(20).to_dict("records")
        raise RuntimeError(f"Horas eólicas incompletas no período final: {incomplete}. Exemplos: {sample}")

    for out_col, (src_col, how) in HOURLY_COMBINE.items():
        hourly[out_col] = combine_halfhours(hourly[f"{src_col}_previous"], hourly[f"{src_col}_current"], how)
    hourly[FLAG_COLS] = hourly[FLAG_COLS].astype("int8")

    hourly["curtailment_mwh"] = hourly["curtailment_mwmed"]
    hourly["curtailment_flag"] = (
        (hourly["curtailment_mwmed"] > 0)
        | (hourly["curtailment_minutes"] > 0)
        | (hourly[REASON_COLS].max(axis=1) > 0)
    ).astype("int8")

    keep = [*STATE_ID_KEYS, "datetime", "gerreneol_mw", "curtailment_mwmed", "curtailment_minutes",
            "curtailment_minutes_set_sum", "minutes_rel", "minutes_cnf", "minutes_ene",
            *REASON_COLS, *ORIGIN_COLS, "curtailment_mwh", "curtailment_flag"]
    return hourly[keep].sort_values(["id_estado", "datetime"]).reset_index(drop=True)

# ---------- CARGA / INTERCÂMBIO ----------
def build_load() -> pd.DataFrame:
    df = read_many(LOAD_DIR, LOAD_COLUMNS, ["val_cargaenergiahomwmed"])
    df = df[df["id_subsistema"].eq(SUBSYSTEM)]
    return agg(df, ["datetime", "id_subsistema"],
               {"load_subsystem_mwmed": ("val_cargaenergiahomwmed", "last")})


def build_interchange() -> pd.DataFrame:
    """
    Intercâmbio verificado: uma coluna por par origem_destino.
    Par ausente numa hora que existe no dataset bruto = 0. Hora inteira ausente = NaN.
    """
    df = read_many(INTERCHANGE_DIR, INTERCHANGE_COLUMNS, ["val_intercambiomwmed"])
    df["pair"] = df["id_subsistema_origem"].astype(str) + "_" + df["id_subsistema_destino"].astype(str)

    source_hours = pd.DatetimeIndex(df["datetime"].dropna().drop_duplicates().sort_values())
    all_hours = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")

    pivot = (df.pivot_table(index="datetime", columns="pair", values="val_intercambiomwmed", aggfunc="last")
               .reindex(all_hours))
    pivot.index.name = "datetime"
    existing = pivot.index.isin(source_hours)
    pivot.loc[existing] = pivot.loc[existing].fillna(0.0)
    pivot.columns = [f"interchange_verified_{p.lower()}_mwmed" for p in pivot.columns]
    return pivot.reset_index().sort_values("datetime").reset_index(drop=True)

# ---------- FEATURES TEMPORAIS ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variáveis cíclicas de hora do dia, dia da semana, dia do mês e mês do ano."""
    dt = df["time"].dt
    cycles = (
        ("hour", dt.hour, 24.0),
        ("week_day", dt.dayofweek, 7.0),
        ("month_day", dt.day - 1, dt.days_in_month),
        ("month", dt.month - 1, 12.0),
    )
    for name, value, period in cycles:
        angle = 2 * np.pi * value / period
        df[f"{name}_sin"], df[f"{name}_cos"] = np.sin(angle), np.cos(angle)
    return df

# ---------- FEATURES DE LAG ----------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lags autorregressivos do alvo, deslocados por FORECAST_HORIZON dentro de cada estado.

    Na janela que termina em t, a coluna lag{h}_x no passo t-k vale x(t-k-h): o modelo vê
    todo o histórico de curtailment até t-h, nunca o valor de t (sem vazamento do alvo).

    - curtailment_ratio: curtailment / geração de referência, comparável entre estados
      de tamanhos muito diferentes (BA vs PB), ao contrário do MWmed bruto.
    - roll{w}_*: médias móveis dos lags (persistência recente de restrição).
    - lag{h}_ne_curtailment_ratio: restrição agregada do Nordeste, capturando eventos
      sistêmicos (origin_sis) que atingem vários estados ao mesmo tempo.
    """
    df = df.sort_values(["id_estado", "time"]).reset_index(drop=True)
    gen, cut = df["gerreneol_mw"], df["curtailment_mwmed"]
    df["curtailment_ratio"] = np.where(gen > 0, cut / gen.where(gen > 0), np.where(cut.notna(), 0.0, np.nan))
    df["curtailment_ratio"] = df["curtailment_ratio"].clip(0.0, 1.0)

    h = FORECAST_HORIZON
    by_state = df.groupby("id_estado", observed=True, sort=False)
    for col in LAG_BASE_COLUMNS:
        df[f"{LAG_PREFIX}{col}"] = by_state[col].shift(h).astype("float32")

    for w in ROLLING_WINDOWS:
        for col in ("curtailment_flag", "curtailment_ratio"):
            df[f"roll{w}_{col}"] = (df.groupby("id_estado", observed=True, sort=False)[f"{LAG_PREFIX}{col}"]
                                      .transform(lambda s: s.rolling(w, min_periods=w).mean()))

    # O grid é completo (build_eolic_hourly garante), então shift(h) no índice horário é exato.
    ne = df.groupby("time")[["curtailment_mwmed", "gerreneol_mw"]].sum(min_count=1)
    ne_ratio = (ne["curtailment_mwmed"] / ne["gerreneol_mw"].where(ne["gerreneol_mw"] > 0)).clip(0.0, 1.0)
    df[NE_LAG_COLUMN] = df["time"].map(ne_ratio.shift(h))
    return df

# ---------- DATASET HORÁRIO ----------
def create_state_grid(eolic: pd.DataFrame) -> pd.DataFrame:
    states = eolic[STATE_ID_KEYS].drop_duplicates("id_estado").sort_values("id_estado").reset_index(drop=True)
    hours = pd.date_range(START_TIME, END_TIME, freq="h")
    return states.merge(pd.DataFrame({"datetime": hours}), how="cross")


def build_state_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (dataset horário por estado, mapeamento estado→índice)."""
    logger.info("Construindo state_hourly...")
    eolic = build_eolic_hourly(combine_halfhour_files(list_csvs(EOLIC_DIR)))
    load, interchange = build_load(), build_interchange()

    state = (create_state_grid(eolic)
             .merge(eolic, on=["datetime", *STATE_ID_KEYS], how="left")
             .merge(load, on=["datetime", "id_subsistema"], how="left")
             .merge(interchange, on="datetime", how="left")
             .sort_values(["id_estado", "datetime"]).reset_index(drop=True))
    state["time"] = state.pop("datetime")
    state = add_time_features(state)
    state = add_lag_features(state)

    if state["time"].min() != START_TIME:
        raise RuntimeError(f"Início incorreto: {state['time'].min()}")
    if state["time"].max() != END_TIME:
        raise RuntimeError(f"Fim incorreto: {state['time'].max()}")
    if duplicated := state.duplicated(["id_estado", "time"]).sum():
        raise RuntimeError(f"Chaves id_estado × time duplicadas: {duplicated}")
    if missing_gerren := int(state["gerreneol_mw"].isna().sum()):
        raise RuntimeError(f"Existem horas sem gerreneol_mw no período final: {missing_gerren}")
    if (found_states := set(state["id_estado"].unique())) != EXPECTED_STATES:
        raise RuntimeError(
            f"Unidades espaciais inesperadas (BA e SE devem ficar separados, sem BA_SE): "
            f"esperado={sorted(EXPECTED_STATES)} obtido={sorted(found_states)}"
        )

    # Identificadores e `time` ficam no parquet (split, embedding, relatórios), mas NÃO
    # entram em X: get_feature_columns só seleciona por FEATURE_PREFIXES + TIME_FEATURES.
    final_columns = [
        "id_estado", "nom_estado", "id_subsistema", "nom_subsistema", "time",
        "gerreneol_mw", "curtailment_mwmed", "curtailment_ratio", "curtailment_minutes",
        "curtailment_minutes_set_sum", "minutes_rel", "minutes_cnf", "minutes_ene",
        *REASON_COLS, *ORIGIN_COLS, "curtailment_mwh", "curtailment_flag", "load_subsystem_mwmed",
        *INTERCHANGE_COLUMNS_FINAL, *TIME_FEATURES, *LAG_FEATURE_COLUMNS,
    ]
    if missing_columns := [c for c in final_columns if c not in state.columns]:
        raise RuntimeError(f"Colunas finais ausentes: {missing_columns}")

    state = state[final_columns].sort_values(["id_estado", "time"]).reset_index(drop=True)
    floats = [c for c in state.columns if c.startswith(FLOAT_PREFIXES)]
    state[floats] = state[floats].apply(pd.to_numeric, errors="coerce").astype("float32")
    state[[*FLAG_COLS, "curtailment_flag"]] = state[[*FLAG_COLS, "curtailment_flag"]].astype("int8")

    mapping = state[["id_estado", "nom_estado"]].drop_duplicates().sort_values("id_estado").reset_index(drop=True)
    mapping["state_idx"] = mapping.index
    return state, mapping

# ---------- COBERTURA ----------
def build_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    interchange_cols = [c for c in df.columns if c.startswith("interchange_verified_")]
    expected_hours = len(pd.date_range(START_TIME, END_TIME, freq="h"))

    tmp = df.assign(_gerren_missing=df["gerreneol_mw"].isna(),
                    _load_missing=df["load_subsystem_mwmed"].isna(),
                    _interchange_missing=df[interchange_cols].isna().any(axis=1),
                    _lag_missing=df[LAG_FEATURE_COLUMNS].isna().any(axis=1))
    rep = tmp.groupby("id_estado", observed=True).agg(
        nom_estado=("nom_estado", "first"), n_hours=("time", "size"),
        n_gerren_missing=("_gerren_missing", "sum"), n_load_missing=("_load_missing", "sum"),
        n_interchange_missing_rows=("_interchange_missing", "sum"),
        n_lag_missing_rows=("_lag_missing", "sum"),
    ).reset_index()
    rep["expected_hours"] = expected_hours
    rep["pct_time_coverage"] = rep["n_hours"] / expected_hours * 100
    return rep

# ---------- FEATURES DO MODELO ----------
# Nunca entram em X (identificadores constantes ou redundantes com o embedding de estado).
EXCLUDED_FROM_FEATURES = {"id_estado", "nom_estado", "id_subsistema", "nom_subsistema", "time"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES) and c not in EXCLUDED_FROM_FEATURES]
    cols += [c for c in TIME_FEATURES if c in df.columns]
    # Proteção contra vazamento: nenhum alvo contemporâneo pode ser feature.
    if leaked := set(cols) & {*TARGET_COLUMNS, "curtailment_ratio"}:
        raise RuntimeError(f"Alvo contemporâneo entre as features: {sorted(leaked)}")
    return cols

# ---------- SEQUÊNCIAS ----------
TARGET_MAP = {
    "y_curtailment_flag": "curtailment_flag", "y_curtailment_mwmed": "curtailment_mwmed",
    "y_curtailment_mwh": "curtailment_mwh", "y_curtailment_minutes": "curtailment_minutes",
}
TARGET_COLUMNS = [*TARGET_MAP.values(), *REASON_COLS]
SEQ_DTYPES = {
    "X": "float32", "y_curtailment_flag": "int8", "y_curtailment_mwmed": "float32",
    "y_curtailment_mwh": "float32", "y_curtailment_minutes": "float32",
    "y_reason": "int8", "state_idx": "int16", "end_datetime": "int64",
}


def iter_segments(df: pd.DataFrame, feature_columns: list[str]) -> Iterator[pd.DataFrame]:
    """Trechos por estado com horas consecutivas e features completas, com ao menos uma semana."""
    for _, group in df.groupby("id_estado", observed=True):
        g = group.sort_values("time").reset_index(drop=True)
        valid = g[feature_columns].notna().all(axis=1)
        continuous = g["time"].diff().eq(pd.Timedelta(hours=1))
        if len(continuous):
            continuous.iloc[0] = True  # primeiro registro do estado inicia um segmento
        seg_id = (~(valid & continuous)).cumsum()
        for _, seg in g[valid & continuous].groupby(seg_id[valid & continuous], sort=False):
            if len(seg) >= SEQUENCE_LENGTH:
                yield seg.reset_index(drop=True)


def window_targets(segment: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Instantes finais das janelas do segmento e máscara de alvo válido."""
    tail = segment.iloc[SEQUENCE_LENGTH - 1:]
    return tail, tail[TARGET_COLUMNS].notna().all(axis=1).to_numpy()


def build_sequences(df: pd.DataFrame, feature_columns: list[str], mapping: pd.DataFrame) -> None:
    """
    X: [n_amostras, SEQUENCE_LENGTH, n_features]; alvos: ocorrência, mwmed, mwh, minutos, motivos.
    """
    state_to_idx = dict(zip(mapping["id_estado"], mapping["state_idx"]))
    L, F = SEQUENCE_LENGTH, len(feature_columns)

    segments = list(iter_segments(df, feature_columns))
    n = sum(int(window_targets(s)[1].sum()) for s in segments)
    if n == 0:
        raise RuntimeError("Nenhuma sequência válida foi encontrada.")
    logger.info("Amostras: %d | sequência: %d | features: %d", n, L, F)

    shapes = {"X": (n, L, F), "y_reason": (n, len(REASON_COLS))}
    out = {name: np.lib.format.open_memmap(SEQUENCE_DIR / f"{name}.npy", mode="w+", dtype=dt,
                                           shape=shapes.get(name, (n,)))
           for name, dt in SEQ_DTYPES.items()}

    pos = 0
    for segment in segments:
        tail, mask = window_targets(segment)
        if not mask.any():
            continue
        tail, sl = tail[mask], slice(pos, pos + int(mask.sum()))
        windows = sliding_window_view(segment[feature_columns].to_numpy("float32"), L, axis=0)  # (k, F, L)
        out["X"][sl] = windows[mask].transpose(0, 2, 1)
        for name, col in TARGET_MAP.items():
            out[name][sl] = tail[col].to_numpy()
        out["y_reason"][sl] = tail[REASON_COLS].to_numpy("int8")
        out["state_idx"][sl] = tail["id_estado"].map(state_to_idx).to_numpy("int16")
        out["end_datetime"][sl] = tail["time"].to_numpy("datetime64[ns]").astype("int64")
        pos = sl.stop
    for arr in out.values():
        arr.flush()
    logger.info("Sequências gravadas: %d", pos)

# ---------- PIPELINE ----------
def main() -> None:
    for d in (PROCESSED_DIR, TRAIN_DIR, SEQUENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Período: %s -> %s | horizonte: %dh", START_TIME, END_TIME, FORECAST_HORIZON)

    state, mapping = build_state_hourly()
    state.to_parquet(STATE_HOURLY_PATH, index=False)
    logger.info("Dataset horário salvo: %s", STATE_HOURLY_PATH)

    features = get_feature_columns(state)
    logger.info("Features selecionadas (%d): %s", len(features), features)

    (TRAIN_DIR / "feature_columns.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping.to_json(TRAIN_DIR / "state_mapping.json", orient="records", force_ascii=False, indent=2)
    build_coverage_report(state).to_csv(TRAIN_DIR / "coverage_report.csv", index=False, encoding="utf-8-sig")

    build_sequences(state, features, mapping)
    logger.info("Pipeline finalizado.")


if __name__ == "__main__":
    main()