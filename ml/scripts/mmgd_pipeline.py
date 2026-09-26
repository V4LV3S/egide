"""Pré-processamento temporal e normalização de séries de geração MMGD."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:  # Permite importar o módulo nos testes e usá-lo pela execução direta.
    from .pca_pipeline import create_temporal_partitions, split_labels
except ImportError:  # pragma: no cover - caminho usado somente na execução direta.
    from pca_pipeline import create_temporal_partitions, split_labels


TIME_COLUMN = "time"
VALUE_COLUMN = "val_cargammgd"
REFERENCE_QUANTILE = 0.99
ROLLING_WINDOW_DAYS = 60
EXPECTED_START = pd.Timestamp("2023-10-01 00:00:00")
EXPECTED_END = pd.Timestamp("2026-08-31 23:00:00")


def prepare_mmgd_dataframe(
    data: pd.DataFrame, source_path: Path, *, value_column: str
) -> pd.DataFrame:
    """Valida, recorta e ordena uma série MMGD no intervalo esperado."""
    required_columns = {TIME_COLUMN, value_column}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{source_path} não possui as colunas obrigatórias: {sorted(missing_columns)}."
        )

    prepared = data[[TIME_COLUMN, value_column]].copy()
    prepared[TIME_COLUMN] = pd.to_datetime(prepared[TIME_COLUMN], errors="raise")
    if prepared[TIME_COLUMN].isna().any():
        raise ValueError(f"{source_path} possui instantes ausentes.")
    prepared = prepared.loc[
        prepared[TIME_COLUMN].between(EXPECTED_START, EXPECTED_END, inclusive="both")
    ].copy()
    prepared[value_column] = pd.to_numeric(prepared[value_column], errors="raise")
    if prepared[TIME_COLUMN].duplicated().any():
        raise ValueError(f"{source_path} possui instantes repetidos.")
    if not np.isfinite(prepared[value_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{source_path} possui valores ausentes ou infinitos em {value_column}.")
    return prepared.sort_values(TIME_COLUMN, ignore_index=True)


def normalize_mmgd_dataframe(
    data: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    value_column: str,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Divide cronologicamente e normaliza por P99 móvel do treino.

    Cada valor de treino é dividido pelo P99 dos ``rolling_window_days`` dias
    anteriores, incluindo o próprio instante. Validação e teste usam o último
    P99 estimado no treino. Portanto, nenhuma janela de referência consulta
    dados desses dois blocos, evitando vazamento temporal.
    """
    if (
        isinstance(rolling_window_days, bool)
        or not isinstance(rolling_window_days, int)
        or rolling_window_days <= 0
    ):
        raise ValueError("rolling_window_days deve ser um inteiro positivo.")

    if data.empty:
        raise ValueError(
            f"A série não possui dados entre {EXPECTED_START} e {EXPECTED_END}."
        )

    actual_start = data[TIME_COLUMN].iloc[0]
    actual_end = data[TIME_COLUMN].iloc[-1]
    if actual_start != EXPECTED_START or actual_end != EXPECTED_END:
        raise ValueError(
            f"A série deve cobrir exatamente de {EXPECTED_START} até {EXPECTED_END}; "
            f"intervalo encontrado: {actual_start} até {actual_end}."
        )

    partitions = create_temporal_partitions(
        len(data), train_fraction, validation_fraction
    )
    train = data.iloc[partitions.train].set_index(TIME_COLUMN)[value_column]
    train_reference = train.rolling(
        f"{rolling_window_days}D", min_periods=1
    ).quantile(REFERENCE_QUANTILE)
    if (train_reference <= 0).any():
        invalid_reference = train_reference <= 0
        if (train[invalid_reference] != 0).any():
            raise ValueError(
                "Não é possível normalizar valores não nulos com P99 de treino menor ou igual a zero."
            )

    # A referência final do treino é congelada para validação e teste. Isso
    # impede que uma janela móvel use observações futuras durante a transformação.
    reference = np.concatenate(
        (
            train_reference.to_numpy(),
            np.full(len(data.iloc[partitions.validation]), train_reference.iloc[-1]),
            np.full(len(data.iloc[partitions.test]), train_reference.iloc[-1]),
        )
    )
    values = data[value_column].to_numpy(dtype=float)
    normalized_values = np.divide(
        values,
        reference,
        out=np.zeros_like(values),
        where=(values != 0) & (reference > 0),
    )

    normalized = data.copy()
    normalized[value_column] = normalized_values
    normalized.insert(1, "split", split_labels(len(normalized), partitions))
    return normalized


def build_mmgd_dataset(
    source_path: Path,
    *,
    train_fraction: float,
    validation_fraction: float,
    value_column: str = VALUE_COLUMN,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Carrega uma série MMGD e retorna tempo, partição e valor normalizado."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {source_path}")
    data = prepare_mmgd_dataframe(
        pd.read_parquet(source_path), source_path, value_column=value_column
    )
    return normalize_mmgd_dataframe(
        data,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        value_column=value_column,
        rolling_window_days=rolling_window_days,
    )


def create_mmgd_parquet_files(
    input_directory: Path,
    output_directory: Path,
    *,
    train_fraction: float,
    validation_fraction: float,
    value_column: str = VALUE_COLUMN,
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
    files_to_process: tuple[str, ...] | None = None,
    overwrite_existing_outputs: bool = False,
) -> list[Path]:
    """Gera um Parquet normalizado por área a partir dos arquivos MMGD."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_directory}")

    input_paths = (
        [input_directory / file_name for file_name in files_to_process]
        if files_to_process is not None
        else sorted(input_directory.glob("*.parquet"))
    )
    if not input_paths:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {input_directory}")
    missing_paths = [str(path) for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Arquivos não encontrados: {', '.join(missing_paths)}")

    output_paths: list[Path] = []
    for input_path in input_paths:
        output_path = output_directory / input_path.name
        if output_path.exists() and not overwrite_existing_outputs:
            print(f"Ignorando saída existente: {output_path}")
            continue

        dataset = build_mmgd_dataset(
            input_path,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            value_column=value_column,
            rolling_window_days=rolling_window_days,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)
        print(f"{input_path.name} -> {output_path}")
        output_paths.append(output_path)
    return output_paths
