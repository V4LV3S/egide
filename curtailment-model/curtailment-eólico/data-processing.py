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
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
WINDOWS = (3, 6, 12, 24, 48, 168)

# Escopo: subsistema mantido e estados unidos (ajuste aos valores reais dos CSVs).
SUBSYSTEM = "NE"
STATE_MERGE = {"BA": "BA_SE", "SE": "BA_SE"}  # id_estado original -> id do grupo
STATE_NAMES = {"BA_SE": "Bahia+Sergipe"}      # nome do grupo

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
REASON_CODES, ORIGIN_CODES = ["REL", "CNF", "ENE", "PAR"], ["LOC", "SIS"]
REASON_COLS = [f"reason_{c}" for c in REASON_CODES]
FLAG_COLS = REASON_COLS + [f"origin_{c}" for c in ORIGIN_CODES]
FLAG_MAX = {c: (c, "max") for c in FLAG_COLS}

STATE_KEYS = ["din_instante", "id_estado", "nom_estado", "id_subsistema", "nom_subsistema"]
ENTITY_KEYS = [*STATE_KEYS, "id_ons"]
HOUR_KEYS = ["datetime", *STATE_KEYS[1:]]

# ---------- AGREGAÇÕES (saída=(coluna, função)) ----------
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
    **FLAG_MAX, "n_entities": ("id_ons", "nunique"),
}
# Reagrupa chunks que possam ter dividido o mesmo grupo.
MERGE_SPEC = {k: (k, "sum" if fn == "sum" else "max") for k, (_, fn) in STATE_SPEC.items()}
HOURLY_SPEC = {
    "GerRenEOL_MW": ("geracao_referencia_halfhour", "mean"),
    "curtailment_MWmed": ("gnr_halfhour", "mean"),
    "curtailment_minutes": ("minutes_restricao_max_halfhour", "sum"),
    "curtailment_minutes_set_sum": ("minutes_restricao_sum_halfhour", "sum"),
    "minutes_REL": ("minutes_rel_halfhour", "sum"),
    "minutes_CNF": ("minutes_cnf_halfhour", "sum"),
    "minutes_ENE": ("minutes_ene_halfhour", "sum"),
    **FLAG_MAX, "n_halfhours": ("din_instante", "nunique"), "n_entities": ("n_entities", "max"),
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

# ---------- EÓLICO ----------
def prepare_eolic_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filtra o subsistema, une estados, normaliza tipos e cria flags de razão/origem."""
    chunk = chunk[chunk["id_subsistema"].eq(SUBSYSTEM)].copy()
    chunk["din_instante"] = to_dt(chunk["din_instante"])
    chunk["id_estado"] = chunk["id_estado"].replace(STATE_MERGE)
    chunk["nom_estado"] = chunk["id_estado"].map(STATE_NAMES).fillna(chunk["nom_estado"])
    to_num(chunk, EOLIC_NUMERIC)
    for prefix, col, codes in (("reason", "cod_razaorestricao", REASON_CODES),
                               ("origin", "cod_origemrestricao", ORIGIN_CODES)):
        for code in codes:
            chunk[f"{prefix}_{code}"] = chunk[col].eq(code).astype("int8")
    return chunk


def aggregate_eolic_file(path: Path) -> pd.DataFrame:
    """Arquivo mensal: 30min×entidade → 30min×estado → hora×estado."""
    logger.info("Processando eólico: %s", path)
    if missing := set(EOLIC_COLUMNS) - set(read_header(path)):
        raise ValueError(f"{path} sem colunas obrigatórias: {sorted(missing)}")
    parts = [
        agg(agg(prepare_eolic_chunk(chunk), ENTITY_KEYS, ENTITY_SPEC), STATE_KEYS, STATE_SPEC)
        for chunk in pd.read_csv(path, usecols=EOLIC_COLUMNS, chunksize=CHUNKSIZE, low_memory=False, **CSV_KW)
    ]
    half = agg(pd.concat(parts, ignore_index=True), STATE_KEYS, MERGE_SPEC)
    half["datetime"] = half["din_instante"].dt.floor("h")
    hourly = agg(half, HOUR_KEYS, HOURLY_SPEC)
    hourly["curtailment_MWh"] = hourly["curtailment_MWmed"]  # MWmed em 1h ≡ MWh
    hourly["curtailment_flag"] = (
        (hourly["curtailment_MWmed"].fillna(0) > 0)
        | (hourly["curtailment_minutes"].fillna(0) > 0)
        | (hourly[REASON_COLS].max(axis=1).fillna(0) > 0)
    ).astype("int8")
    hourly["complete_eolic_hour"] = (hourly["n_halfhours"] == 2).astype("int8")
    return hourly


def build_eolic_hourly() -> pd.DataFrame:
    df = pd.concat([aggregate_eolic_file(p) for p in list_csvs(EOLIC_DIR)], ignore_index=True)
    return (df.sort_values(["id_estado", "datetime"])
              .drop_duplicates(["datetime", "id_estado"], keep="last")
              .reset_index(drop=True))

# ---------- CARGA / INTERCÂMBIO ----------
def build_load() -> pd.DataFrame:
    df = read_many(LOAD_DIR, LOAD_COLUMNS, ["val_cargaenergiahomwmed"])
    return agg(df, ["datetime", "id_subsistema"],
               {"load_subsystem_MWmed": ("val_cargaenergiahomwmed", "last")})


def build_interchange() -> pd.DataFrame:
    """
    Intercâmbio verificado: uma coluna por par origem_destino, em grade horária contínua.
    Ausência do par em uma hora é representada por 0.
    """
    df = read_many(INTERCHANGE_DIR, INTERCHANGE_COLUMNS, ["val_intercambiomwmed"])
    required = {"datetime", "id_subsistema_origem", "id_subsistema_destino", "val_intercambiomwmed"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colunas obrigatórias ausentes no intercâmbio: {sorted(missing)}")
    df["pair"] = df["id_subsistema_origem"].astype(str) + "_" + df["id_subsistema_destino"].astype(str)
    hours = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h", name="datetime")
    pivot = (df.pivot_table(index="datetime", columns="pair", values="val_intercambiomwmed", aggfunc="last")
               .reindex(hours).fillna(0.0))
    pivot.columns = [f"interchange_verified_{p}_MWmed" for p in pivot.columns]
    return pivot.reset_index()

# ---------- GRADE ----------
def create_state_grid(eolic: pd.DataFrame) -> pd.DataFrame:
    """Grade horária contínua por estado. GerRenEOL ausente NÃO é preenchido."""
    mapping = eolic[STATE_KEYS[1:]].drop_duplicates("id_estado").sort_values("id_estado")
    hours = pd.date_range(eolic["datetime"].min(), eolic["datetime"].max(), freq="h")
    return mapping.merge(pd.DataFrame({"datetime": hours}), how="cross")

# ---------- FEATURES ----------
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
FLOAT_PREFIXES = ("GerRenEOL", "curtailment", "load_", "interchange_", "minutes_")
FEATURE_PREFIXES = ("GerRenEOL", "load_", "interchange_verified_")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"].dt
    for name, values, period in (("hour", dt.hour, 24), ("dow", dt.dayofweek, 7), ("doy", dt.dayofyear, 365.25)):
        angle = 2 * np.pi * values / period
        df[f"{name}_sin"], df[f"{name}_cos"] = np.sin(angle), np.cos(angle)
    df["is_weekend"] = (dt.dayofweek >= 5).astype("int8")
    return df


def _roll(s: pd.Series, window: int, stat: str) -> pd.Series:
    return getattr(s.rolling(window), stat)()


def add_gerren_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, janelas móveis e variações de GerRenEOL (janelas usam shift(1): sem vazamento)."""
    col = df["GerRenEOL_MW"]
    by = df.groupby("id_estado", sort=False)["GerRenEOL_MW"]
    new = {f"GerRenEOL_lag_{l}h": by.shift(l) for l in LAGS}
    shifted = by.shift(1).groupby(df["id_estado"])
    for w in WINDOWS:
        for stat in ("mean", "std", "max"):
            new[f"GerRenEOL_roll_{stat}_{w}h"] = shifted.transform(_roll, w, stat)
    for h in (1, 24, 168):
        new[f"GerRenEOL_change_{h}h"] = col - new[f"GerRenEOL_lag_{h}h"]
    for h in (1, 24):
        new[f"GerRenEOL_pct_change_{h}h"] = new[f"GerRenEOL_change_{h}h"] / new[f"GerRenEOL_lag_{h}h"].replace(0, np.nan)
    return df.assign(**new)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Features da LSTM/CNN. Exclui alvos, razões, origens e intercâmbio programado."""
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    return cols + [c for c in TIME_FEATURES if c in df.columns]

# ---------- DATASET HORÁRIO ----------
def build_state_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (dataset horário por estado, mapeamento estado→índice)."""
    logger.info("Construindo eólico, carga e intercâmbio...")
    eolic, load, interchange = build_eolic_hourly(), build_load(), build_interchange()
    state = (create_state_grid(eolic)
             .merge(eolic, on=HOUR_KEYS, how="left")
             .merge(load, on=["datetime", "id_subsistema"], how="left")
             .merge(interchange, on="datetime", how="left")
             .sort_values(["id_estado", "datetime"]).reset_index(drop=True))

    # Intercâmbio é sistêmico: horas fora da grade do intercâmbio viram 0.
    inter_cols = [c for c in state.columns if c.startswith("interchange_verified_")]
    state[inter_cols] = state[inter_cols].fillna(0.0)
    state = add_gerren_features(add_time_features(state))

    mapping = state[["id_estado", "nom_estado"]].drop_duplicates().sort_values("id_estado").reset_index(drop=True)
    mapping["state_idx"] = mapping.index
    state["state_idx"] = state["id_estado"].map(dict(zip(mapping["id_estado"], mapping["state_idx"]))).astype("int16")

    floats = [c for c in state.columns if c.startswith(FLOAT_PREFIXES)]
    state[floats] = state[floats].apply(pd.to_numeric, errors="coerce").astype("float32")
    return state, mapping


def build_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.assign(complete=df["complete_eolic_hour"].fillna(0),
                    gerren_missing=df["GerRenEOL_MW"].isna(),
                    load_missing=df["load_subsystem_MWmed"].isna())
    rep = tmp.groupby("id_estado", observed=True).agg(
        nom_estado=("nom_estado", "first"), n_hours=("datetime", "size"),
        n_complete_eolic_hours=("complete", "sum"), pct_complete_eolic_hours=("complete", "mean"),
        n_gerren_missing=("gerren_missing", "sum"), n_load_missing=("load_missing", "sum"),
    ).reset_index()
    rep["pct_complete_eolic_hours"] *= 100
    return rep

# ---------- SEQUÊNCIAS ----------
TARGET_MAP = {
    "y_curtailment_flag": "curtailment_flag", "y_curtailment_mwmed": "curtailment_MWmed",
    "y_curtailment_mwh": "curtailment_MWh", "y_curtailment_minutes": "curtailment_minutes",
}
TARGET_COLUMNS = [*TARGET_MAP.values(), *REASON_COLS]
SEQ_DTYPES = {
    "X": "float32", "y_curtailment_flag": "int8", "y_curtailment_mwmed": "float32",
    "y_curtailment_mwh": "float32", "y_curtailment_minutes": "float32",
    "y_reason": "int8", "state_idx": "int16", "end_datetime": "int64",
}


def iter_segments(df: pd.DataFrame, feature_columns: list[str]) -> Iterator[pd.DataFrame]:
    """
    Trechos por estado com horas consecutivas, features completas e hora eólica completa.
    Alvos não quebram o histórico: só o instante final da janela precisa de alvo válido.
    """
    step = pd.Timedelta(hours=1)
    for _, group in df.groupby("id_estado", observed=True):
        g = group.sort_values("datetime").reset_index(drop=True)
        valid = g["complete_eolic_hour"].eq(1) & g[feature_columns].notna().all(axis=1)
        continuous = g["datetime"].diff().eq(step)
        continuous.iloc[0] = True  # primeiro registro inicia um segmento
        seg_id = (~(valid & continuous)).cumsum()
        for _, seg in g[valid].groupby(seg_id[valid], sort=False):
            if len(seg) >= SEQUENCE_LENGTH:
                yield seg


def window_targets(seg: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Instantes finais das janelas do segmento e máscara de alvo válido."""
    tail = seg.iloc[SEQUENCE_LENGTH - 1:]
    return tail, tail[TARGET_COLUMNS].notna().all(axis=1).to_numpy()


def build_sequences(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """
    X: [n_amostras, SEQUENCE_LENGTH, n_features]; alvos: ocorrência, MWmed, MWh, minutos, motivos.
    """
    L, F = SEQUENCE_LENGTH, len(feature_columns)
    n = sum(int(window_targets(s)[1].sum()) for s in iter_segments(df, feature_columns))
    if n == 0:
        raise RuntimeError("Nenhuma sequência válida foi encontrada.")
    logger.info("Amostras: %d | sequência: %d | features: %d", n, L, F)

    shapes = {"X": (n, L, F), "y_reason": (n, len(REASON_COLS))}
    out = {name: np.lib.format.open_memmap(SEQUENCE_DIR / f"{name}.npy", mode="w+", dtype=dt,
                                           shape=shapes.get(name, (n,)))
           for name, dt in SEQ_DTYPES.items()}
    pos = 0
    for seg in iter_segments(df, feature_columns):
        tail, mask = window_targets(seg)
        if not mask.any():
            continue
        tail, sl = tail[mask], slice(pos, pos + int(mask.sum()))
        windows = sliding_window_view(seg[feature_columns].to_numpy("float32"), L, axis=0)  # (k, F, L)
        out["X"][sl] = windows[mask].transpose(0, 2, 1)
        for name, col in TARGET_MAP.items():
            out[name][sl] = tail[col].to_numpy()
        out["y_reason"][sl] = tail[REASON_COLS].to_numpy("int8")
        out["state_idx"][sl] = tail["state_idx"].to_numpy()
        out["end_datetime"][sl] = tail["datetime"].to_numpy("datetime64[ns]").astype("int64")
        pos = sl.stop
    for arr in out.values():
        arr.flush()
    logger.info("Sequências gravadas: %d", pos)

# ---------- PIPELINE ----------
def main() -> None:
    for d in (PROCESSED_DIR, TRAIN_DIR, SEQUENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    state, mapping = build_state_hourly()
    state.to_parquet(STATE_HOURLY_PATH, index=False)
    logger.info("Dataset horário salvo: %s", STATE_HOURLY_PATH)

    features = get_feature_columns(state)
    logger.info("Features selecionadas (%d): %s", len(features), features)

    build_coverage_report(state).to_csv(TRAIN_DIR / "coverage_report.csv", index=False, encoding="utf-8-sig")
    mapping.to_json(TRAIN_DIR / "state_mapping.json", orient="records", force_ascii=False, indent=2)
    (TRAIN_DIR / "feature_columns.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")

    build_sequences(state, features)
    logger.info("Pipeline finalizado.")


if __name__ == "__main__":
    main()