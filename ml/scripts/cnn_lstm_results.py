"""Exportação dos resultados de teste dos modelos CNN-LSTM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


EntityType = Literal["regiao_geoletrica", "usina"]
JSON_VALUE_DECIMALS = 4


def _json_values(values: np.ndarray) -> list[float]:
    """Arredonda valores e elimina apenas resíduos numéricos próximos de zero."""
    rounded = np.round(np.asarray(values, dtype=float), JSON_VALUE_DECIMALS)
    zero_tolerance = 10 ** (-JSON_VALUE_DECIMALS)
    rounded[np.abs(rounded) <= zero_tolerance] = 0.0
    return rounded.tolist()


@dataclass(frozen=True)
class TargetAffineTransform:
    """Transformação que devolve o alvo normalizado à escala original."""

    multiplier: float
    offset: float

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Aplica a transformação inversa sem alterar o array recebido."""
        return np.asarray(values, dtype=float) * self.multiplier + self.offset


def _single_value_column(data: pd.DataFrame, source_path: Path) -> str:
    """Encontra a única coluna de valor disponível em um Parquet."""
    value_columns = [
        column for column in data.columns if column.startswith("val_")
    ]
    if len(value_columns) != 1:
        raise ValueError(
            f"{source_path} deve possuir exatamente uma coluna iniciada por "
            f"'val_'; encontradas: {value_columns}."
        )
    return value_columns[0]


def fit_original_target_transform(
    normalized_parquet_path: Path,
    original_parquet_path: Path,
) -> TargetAffineTransform:
    """Recupera a transformação inversa alinhando teste e série original.

    Séries originais sub-horárias são agregadas pela média da hora encerrada no
    timestamp. Esse é o alinhamento usado nos datasets individuais existentes.
    Tanto o StandardScaler das usinas quanto o P99 congelado do teste MMGD
    resultam em uma relação afim, validada antes de qualquer exportação.
    """
    if not original_parquet_path.is_file():
        raise FileNotFoundError(
            "Parquet com a série em escala original não encontrado: "
            f"{original_parquet_path}"
        )

    normalized = pd.read_parquet(normalized_parquet_path)
    original = pd.read_parquet(original_parquet_path)
    normalized_column = _single_value_column(
        normalized, normalized_parquet_path
    )
    original_column = _single_value_column(original, original_parquet_path)
    required_normalized_columns = {"time", "split", normalized_column}
    if not required_normalized_columns.issubset(normalized.columns):
        raise ValueError(
            f"Colunas time/split ausentes em {normalized_parquet_path}."
        )

    original = original.loc[:, ["time", original_column]].copy()
    original["time"] = pd.to_datetime(original["time"], errors="raise")
    original_hourly = (
        original.set_index("time")[original_column]
        .resample("h", label="right", closed="right")
        .mean()
        .rename("original_value")
        .reset_index()
    )
    test_values = normalized.loc[
        normalized["split"] == "test", ["time", normalized_column]
    ].copy()
    test_values["time"] = pd.to_datetime(test_values["time"], errors="raise")
    aligned = test_values.merge(
        original_hourly,
        on="time",
        how="inner",
        validate="one_to_one",
    ).dropna()
    if len(aligned) < 2 or aligned[normalized_column].nunique() < 2:
        raise ValueError(
            "Não há valores de teste suficientes para recuperar a escala "
            f"original de {normalized_parquet_path}."
        )

    normalized_values = aligned[normalized_column].to_numpy(dtype=float)
    original_values = aligned["original_value"].to_numpy(dtype=float)
    design = np.column_stack(
        (normalized_values, np.ones(len(normalized_values)))
    )
    multiplier, offset = np.linalg.lstsq(
        design, original_values, rcond=None
    )[0]
    restored = normalized_values * multiplier + offset
    tolerance = max(1e-6, float(np.ptp(original_values)) * 1e-9)
    if multiplier <= 0 or not np.allclose(
        restored, original_values, rtol=1e-9, atol=tolerance
    ):
        raise ValueError(
            "A série original não corresponde à normalização do dataset: "
            f"{original_parquet_path}."
        )
    return TargetAffineTransform(float(multiplier), float(offset))


def load_split_target_times(
    parquet_path: Path,
    *,
    split: str,
    lookback: int,
    horizon: int,
    expected_sample_count: int,
) -> np.ndarray:
    """Reconstrói os timestamps dos alvos conforme a criação das janelas."""
    if not parquet_path.is_file():
        raise FileNotFoundError(
            "Parquet necessário para recuperar os timestamps não encontrado: "
            f"{parquet_path}"
        )

    data = pd.read_parquet(parquet_path, columns=["time", "split"])
    timestamps = (
        pd.to_datetime(
            data.loc[data["split"] == split, "time"], errors="raise"
        )
        .sort_values()
        .to_numpy(dtype="datetime64[ns]")
    )
    window_size = lookback + horizon
    hourly_delta = np.timedelta64(1, "h")
    target_times = [
        timestamps[start + lookback : start + window_size]
        for start in range(len(timestamps) - window_size + 1)
        if np.all(
            np.diff(timestamps[start : start + window_size]) == hourly_delta
        )
    ]
    result = (
        np.stack(target_times)
        if target_times
        else np.empty((0, horizon), dtype="datetime64[ns]")
    )
    if result.shape != (expected_sample_count, horizon):
        raise ValueError(
            f"Os timestamps reconstruídos não correspondem às janelas de {split}: "
            f"esperado {(expected_sample_count, horizon)}, obtido {result.shape}. "
            "Confirme que o NPZ e o Parquet pertencem ao mesmo dataset e à "
            "mesma execução."
        )
    return result


def write_test_results(
    output_path: Path,
    *,
    dataset_name: str,
    entity_type: EntityType,
    target_times: np.ndarray,
    observed_values: np.ndarray,
    predicted_values: np.ndarray,
    value_scale: Literal["normalizada", "original"] = "normalizada",
) -> None:
    """Salva previsões e observações de cada janela de teste em JSON.

    Em cada registro, o índice zero das listas corresponde a ``time0`` e os
    índices seguintes avançam em intervalos de uma hora.
    """
    target_times = np.asarray(target_times)
    observed_values = np.asarray(observed_values)
    predicted_values = np.asarray(predicted_values)
    if observed_values.ndim != 2:
        raise ValueError("Os valores de teste devem ter duas dimensões.")
    if predicted_values.shape != observed_values.shape:
        raise ValueError(
            "Previsões e observações devem possuir o mesmo formato: "
            f"{predicted_values.shape} != {observed_values.shape}."
        )
    if target_times.shape != observed_values.shape:
        raise ValueError(
            "Timestamps e observações devem possuir o mesmo formato: "
            f"{target_times.shape} != {observed_values.shape}."
        )
    if not np.isfinite(observed_values).all() or not np.isfinite(
        predicted_values
    ).all():
        raise ValueError("Previsões e observações devem conter apenas finitos.")

    results = [
        {
            "time0": np.datetime_as_string(target_times[index, 0], unit="s"),
            "valores_previstos": _json_values(predicted_values[index]),
            "valores_observados": _json_values(observed_values[index]),
        }
        for index in range(len(observed_values))
    ]
    payload = {
        "tipo_serie": entity_type,
        entity_type: dataset_name,
        "escala_valores": value_scale,
        "intervalo_horas": 1,
        "horizonte_horas": int(observed_values.shape[1]),
        "resultados": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
