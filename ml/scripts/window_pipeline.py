"""Criação de janelas temporais sem vazamento para previsão multi-horizonte."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMN = "time"
SPLIT_COLUMN = "split"
DEFAULT_TARGET_COLUMN = "val_cargammgd"
DEFAULT_CALENDAR_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "week_day_sin",
    "week_day_cos",
    "month_day_sin",
    "month_day_cos",
    "month_sin",
    "month_cos",
)

SPLIT_NAMES = ("train", "validation", "test")


def create_split_windows(
    data: pd.DataFrame,
    *,
    past_columns: Sequence[str],
    calendar_columns: Sequence[str],
    target_column: str,
    lookback: int = 24,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria janelas de um único split, descartando trechos sem horas contínuas.

    Para um horizonte iniciado em ``i``, a entrada observada é ``[i-lookback, i)``
    e calendário e alvo futuro são ``[i, i+horizon)``. Assim, carga e PCAs do
    futuro nunca entram em ``X_past``.
    """
    _validate_window_configuration(
        data, past_columns, calendar_columns, target_column, lookback, horizon
    )
    ordered = data.sort_values(TIME_COLUMN, ignore_index=True)
    timestamps = ordered[TIME_COLUMN].to_numpy(dtype="datetime64[ns]")
    past_values = ordered.loc[:, past_columns].to_numpy(dtype=float)
    calendar_values = ordered.loc[:, calendar_columns].to_numpy(dtype=float)
    target_values = ordered[target_column].to_numpy(dtype=float)

    total_window_size = lookback + horizon
    past_windows: list[np.ndarray] = []
    calendar_windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    hourly_delta = np.timedelta64(1, "h")

    for start in range(len(ordered) - total_window_size + 1):
        end = start + total_window_size
        window_timestamps = timestamps[start:end]
        if not np.all(np.diff(window_timestamps) == hourly_delta):
            continue

        future_start = start + lookback
        past_time = window_timestamps[:lookback]
        future_time = window_timestamps[lookback:]
        assert past_time[-1] + hourly_delta == future_time[0]
        assert len(future_time) == horizon

        past_windows.append(past_values[start:future_start])
        calendar_windows.append(calendar_values[future_start:end])
        targets.append(target_values[future_start:end])

    return (
        _stack_or_empty(past_windows, lookback, len(past_columns)),
        _stack_or_empty(calendar_windows, horizon, len(calendar_columns)),
        _stack_or_empty(targets, horizon),
    )


def create_train_validation_test_windows(
    data: pd.DataFrame,
    *,
    past_columns: Sequence[str],
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
    target_column: str = DEFAULT_TARGET_COLUMN,
    lookback: int = 24,
    horizon: int = 24,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Cria janelas independentes para treino, validação e teste.

    O Parquet atual nomeia a validação como ``validation``. Também é aceito
    ``val`` para permitir o uso de tabelas que adotem essa convenção.
    """
    _validate_dataset_splits(data)
    validation_name = "validation" if "validation" in set(data[SPLIT_COLUMN]) else "val"
    split_frames = {
        "train": data.loc[data[SPLIT_COLUMN] == "train"],
        "validation": data.loc[data[SPLIT_COLUMN] == validation_name],
        "test": data.loc[data[SPLIT_COLUMN] == "test"],
    }
    windows = [
        create_split_windows(
            split_frames[name],
            past_columns=past_columns,
            calendar_columns=calendar_columns,
            target_column=target_column,
            lookback=lookback,
            horizon=horizon,
        )
        for name in SPLIT_NAMES
    ]
    return tuple(value for split_windows in windows for value in split_windows)  # type: ignore[return-value]


def build_windows_from_parquet(
    source_path: Path,
    *,
    past_columns: Sequence[str] | None = None,
    calendar_columns: Sequence[str] = DEFAULT_CALENDAR_COLUMNS,
    target_column: str = DEFAULT_TARGET_COLUMN,
    lookback: int = 24,
    horizon: int = 24,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Lê um Parquet consolidado e retorna suas janelas por split."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo consolidado não encontrado: {source_path}")
    data = pd.read_parquet(source_path)
    resolved_past_columns = (
        default_past_columns(data, target_column=target_column)
        if past_columns is None
        else past_columns
    )
    return create_train_validation_test_windows(
        data,
        past_columns=resolved_past_columns,
        calendar_columns=calendar_columns,
        target_column=target_column,
        lookback=lookback,
        horizon=horizon,
    )


def default_past_columns(data: pd.DataFrame, *, target_column: str = DEFAULT_TARGET_COLUMN) -> list[str]:
    """Retorna carga e todas as componentes PCA para o Parquet consolidado."""
    if target_column not in data.columns:
        raise ValueError(f"Coluna alvo ausente: {target_column}.")
    pca_columns = [column for column in data.columns if "_pca" in column]
    if not pca_columns:
        raise ValueError("Nenhuma componente PCA foi encontrada no DataFrame.")
    return [target_column, *pca_columns]


def _validate_window_configuration(
    data: pd.DataFrame,
    past_columns: Sequence[str],
    calendar_columns: Sequence[str],
    target_column: str,
    lookback: int,
    horizon: int,
) -> None:
    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback e horizon devem ser inteiros positivos.")
    columns = [TIME_COLUMN, target_column, *past_columns, *calendar_columns]
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Colunas ausentes para criação das janelas: {missing}.")
    if target_column not in past_columns:
        raise ValueError("A coluna alvo deve fazer parte das features observadas no passado.")
    if set(past_columns) & set(calendar_columns):
        raise ValueError("Features passadas e de calendário não podem se sobrepor.")
    if data[TIME_COLUMN].isna().any() or data[TIME_COLUMN].duplicated().any():
        raise ValueError("Cada split deve possuir timestamps válidos e sem duplicatas.")
    numeric_columns = [*past_columns, *calendar_columns]
    values = data.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("As features usadas nas janelas devem ser finitas.")


def _validate_dataset_splits(data: pd.DataFrame) -> None:
    if SPLIT_COLUMN not in data.columns:
        raise ValueError(f"Coluna obrigatória ausente: {SPLIT_COLUMN}.")
    labels = set(data[SPLIT_COLUMN].dropna())
    if "validation" in labels and "val" in labels:
        raise ValueError("Use somente um rótulo de validação: 'validation' ou 'val'.")
    expected = {"train", "test"}
    expected.add("validation" if "validation" in labels else "val")
    unknown = labels - expected
    missing = expected - labels
    if unknown or missing:
        raise ValueError(f"Splits inválidos; ausentes: {sorted(missing)}, desconhecidos: {sorted(unknown)}.")


def _stack_or_empty(values: list[np.ndarray], *shape: int) -> np.ndarray:
    """Empilha janelas ou preserva o formato esperado quando não há amostras."""
    return np.stack(values) if values else np.empty((0, *shape), dtype=float)
