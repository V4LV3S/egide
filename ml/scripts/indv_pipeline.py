"""Pre-processamento temporal de geração de referência por usina."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:  # Permite importar o módulo nos testes e usá-lo pela execução direta.
    from .pca_pipeline import create_temporal_partitions, split_labels
except ImportError:  # pragma: no cover - caminho usado somente na execução direta.
    from pca_pipeline import create_temporal_partitions, split_labels


TIME_COLUMN = "time"
PLANT_COLUMN = "nom_usina"
VALUE_COLUMN = "val_disponibilidade"


def prepare_indv_dataframe(data: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Valida e ordena a série de uma única usina."""
    required_columns = {TIME_COLUMN, PLANT_COLUMN, VALUE_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{source_path} não possui as colunas obrigatórias: "
            f"{sorted(missing_columns)}."
        )

    plant_names = data[PLANT_COLUMN].dropna().unique()
    if len(plant_names) != 1 or data[PLANT_COLUMN].isna().any():
        raise ValueError(
            f"{source_path} deve conter dados de exatamente uma usina."
        )

    prepared = data[[TIME_COLUMN, VALUE_COLUMN]].copy()
    prepared[TIME_COLUMN] = pd.to_datetime(prepared[TIME_COLUMN], errors="raise")
    if prepared.empty:
        raise ValueError(f"{source_path} não possui observações.")
    if prepared[TIME_COLUMN].isna().any():
        raise ValueError(f"{source_path} possui instantes ausentes.")
    if prepared[TIME_COLUMN].duplicated().any():
        raise ValueError(f"{source_path} possui instantes repetidos.")

    prepared[VALUE_COLUMN] = pd.to_numeric(
        prepared[VALUE_COLUMN], errors="raise"
    )
    if not np.isfinite(prepared[VALUE_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{source_path} possui valores ausentes ou infinitos em "
            f"{VALUE_COLUMN}."
        )
    return prepared.sort_values(TIME_COLUMN, ignore_index=True)


def normalize_indv_dataframe(
    data: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Divide cronologicamente e padroniza usando somente dados de treino.

    O ``StandardScaler`` estima média e desvio-padrão no treino. A mesma
    transformação é aplicada a validação e teste, evitando vazamento temporal.
    """
    partitions = create_temporal_partitions(
        len(data), train_fraction, validation_fraction
    )
    values = data[[VALUE_COLUMN]].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaler.fit(values[partitions.train])
    normalized_values = scaler.transform(values).ravel()

    normalized = data.copy()
    normalized[VALUE_COLUMN] = normalized_values
    normalized.insert(1, "split", split_labels(len(normalized), partitions))
    return normalized


def build_indv_dataset(
    source_path: Path,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Carrega uma usina e retorna tempo, partição e geração normalizada."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {source_path}")
    data = prepare_indv_dataframe(pd.read_parquet(source_path), source_path)
    return normalize_indv_dataframe(
        data,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )


def create_indv_parquet_files(
    input_directory: Path,
    output_directory: Path,
    *,
    train_fraction: float,
    validation_fraction: float,
    files_to_process: tuple[str, ...] | None = None,
    overwrite_existing_outputs: bool = False,
) -> list[Path]:
    """Gera um Parquet normalizado para cada arquivo individual de usina."""
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Diretório de entrada não encontrado: {input_directory}"
        )

    input_paths = (
        [input_directory / file_name for file_name in files_to_process]
        if files_to_process is not None
        else sorted(input_directory.glob("*.parquet"))
    )
    if not input_paths:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {input_directory}")
    missing_paths = [str(path) for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Arquivos não encontrados: {', '.join(missing_paths)}"
        )

    output_paths: list[Path] = []
    for input_path in input_paths:
        output_path = output_directory / input_path.name
        if output_path.exists() and not overwrite_existing_outputs:
            print(f"Ignorando saída existente: {output_path}")
            continue

        dataset = build_indv_dataset(
            input_path,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)
        print(f"{input_path.name} -> {output_path}")
        output_paths.append(output_path)
    return output_paths
