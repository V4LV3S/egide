"""Cria features meteorológicas 2x2 por usina, sem aplicar PCA."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:  # Permite importar nos testes e executar diretamente pela IDE.
    from .pca_pipeline import (
        TemporalPartitions,
        align_on_common_time,
        apply_log1p_transform,
        create_temporal_partitions,
        load_data_variable,
        prepare_feature_matrix,
        split_labels,
    )
except ImportError:  # pragma: no cover - usado somente na execução direta.
    from pca_pipeline import (
        TemporalPartitions,
        align_on_common_time,
        apply_log1p_transform,
        create_temporal_partitions,
        load_data_variable,
        prepare_feature_matrix,
        split_labels,
    )


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado-usinas"
OUTPUT_DIRECTORY = ROOT / "ml" / "data" / "meteoro_indiv"
OUTPUT_FILE_NAME = "grid_features.parquet"

VARIABLES = ("t2m", "ssr", "tcc", "tp")
VARIABLES_WITH_LOG1P = ("tp",)
PLANTS_TO_PROCESS: tuple[str, ...] | None = None
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.20
OVERWRITE_EXISTING_OUTPUTS = False
EXPECTED_GRID_FEATURES = 4


def transform_grid_variable(
    data_array: xr.DataArray,
    source_path: Path,
    partitions: TemporalPartitions,
    *,
    apply_log1p: bool,
) -> xr.DataArray:
    """Imputa e padroniza quatro células usando somente dados de treino."""
    matrix = prepare_feature_matrix(data_array, source_path)
    if matrix.sizes["spatial_point"] != EXPECTED_GRID_FEATURES:
        raise ValueError(
            f"{source_path} deve possuir grade 2x2; encontrados "
            f"{matrix.sizes['spatial_point']} pontos."
        )
    if apply_log1p:
        matrix = apply_log1p_transform(matrix, source_path)

    train = matrix.values[partitions.train]
    if np.isnan(train).all(axis=0).any():
        raise ValueError(
            f"{source_path} possui célula sem observação no treino."
        )

    validation = matrix.values[partitions.validation]
    test = matrix.values[partitions.test]
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train)
    validation = imputer.transform(validation)
    test = imputer.transform(test)

    scaler = StandardScaler()
    normalized = np.concatenate(
        (
            scaler.fit_transform(train),
            scaler.transform(validation),
            scaler.transform(test),
        )
    )
    feature_dimension = f"{data_array.name}_grid_feature"
    return xr.DataArray(
        normalized,
        dims=("time", feature_dimension),
        coords={
            "time": matrix.time.values,
            feature_dimension: np.arange(EXPECTED_GRID_FEATURES),
        },
        name=data_array.name,
        attrs={
            "source_variable": data_array.name,
            "spatial_features": EXPECTED_GRID_FEATURES,
            "spatial_reduction": "none",
            "imputation_fit_scope": "train_only",
            "normalization_fit_scope": "train_only",
            "value_transformation": "log1p" if apply_log1p else "none",
        },
    )


def build_grid_dataset(
    plant_directory: Path,
    variables: Sequence[str],
    *,
    train_fraction: float,
    validation_fraction: float,
    variables_with_log1p: tuple[str, ...] = (),
) -> xr.Dataset:
    """Cria dataset normalizado com quatro features por variável."""
    if not variables:
        raise ValueError("Informe ao menos uma variável meteorológica.")

    sources_by_name: dict[str, tuple[Path, xr.DataArray]] = {}
    for path in sorted(plant_directory.glob("*.nc")):
        array = load_data_variable(path)
        if array.name in sources_by_name:
            raise ValueError(
                f"Variável duplicada em {plant_directory.name}: {array.name}."
            )
        sources_by_name[array.name] = (path, array)

    missing = sorted(set(variables) - set(sources_by_name))
    if missing:
        raise ValueError(
            f"Variáveis ausentes em {plant_directory.name}: {', '.join(missing)}."
        )

    sources = align_on_common_time(
        [sources_by_name[name] for name in variables]
    )
    partitions = create_temporal_partitions(
        sources[0][1].sizes["time"], train_fraction, validation_fraction
    )
    features = [
        transform_grid_variable(
            array,
            source_path,
            partitions,
            apply_log1p=array.name in variables_with_log1p,
        )
        for source_path, array in sources
    ]
    dataset = xr.merge(features)
    dataset = dataset.assign_coords(
        split=("time", split_labels(dataset.sizes["time"], partitions))
    )
    dataset.attrs.update(
        {
            "plant": plant_directory.name,
            "variables": ", ".join(variables),
            "features_per_variable": EXPECTED_GRID_FEATURES,
            "spatial_reduction": "none",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
            "time_alignment": "inner",
            "preprocessing_fit_scope": "train_only",
        }
    )
    return dataset


def grid_dataset_to_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    """Converte dataset 2x2 em tabela com quatro colunas por variável."""
    if "split" not in dataset.coords:
        raise ValueError("Dataset meteorológico sem coordenada 'split'.")

    frames: list[pd.DataFrame] = []
    for data_array in dataset.data_vars.values():
        other_dimensions = [
            dimension for dimension in data_array.dims if dimension != "time"
        ]
        values = data_array.transpose("time", *other_dimensions)
        matrix = np.asarray(values.values).reshape(values.sizes["time"], -1)
        names = [
            f"{data_array.name}_{index}" for index in range(matrix.shape[1])
        ]
        frames.append(
            pd.DataFrame(matrix, index=values.time.values, columns=names)
        )

    dataframe = pd.concat(frames, axis=1)
    dataframe.index.name = "time"
    dataframe = dataframe.reset_index()
    dataframe.insert(1, "split", dataset.split.values)
    return dataframe


def create_plant_parquet_files(
    input_directory: Path,
    output_directory: Path,
    variables: Sequence[str],
    *,
    train_fraction: float,
    validation_fraction: float,
    variables_with_log1p: tuple[str, ...] = (),
    plants_to_process: tuple[str, ...] | None = None,
    overwrite_existing_outputs: bool = False,
) -> list[Path]:
    """Gera um Parquet de features 2x2 para cada conjunto solar."""
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Diretório de entrada não encontrado: {input_directory}"
        )

    plant_directories = (
        [input_directory / name for name in plants_to_process]
        if plants_to_process is not None
        else sorted(path for path in input_directory.iterdir() if path.is_dir())
    )
    missing = [str(path) for path in plant_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Diretórios de usina ausentes: {', '.join(missing)}"
        )
    if not plant_directories:
        raise FileNotFoundError(
            f"Nenhum diretório de usina em {input_directory}"
        )

    output_paths: list[Path] = []
    for plant_directory in plant_directories:
        output_path = output_directory / plant_directory.name / OUTPUT_FILE_NAME
        if output_path.exists() and not overwrite_existing_outputs:
            print(f"Ignorando saída existente: {output_path}")
            continue

        dataset = build_grid_dataset(
            plant_directory,
            variables,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            variables_with_log1p=variables_with_log1p,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_dataset_to_dataframe(dataset).to_parquet(output_path, index=False)
        print(f"{plant_directory.name} -> {output_path}")
        output_paths.append(output_path)
    return output_paths


def main() -> list[Path]:
    """Executa criação das features conforme configuração do módulo."""
    return create_plant_parquet_files(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
        VARIABLES,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        variables_with_log1p=VARIABLES_WITH_LOG1P,
        plants_to_process=PLANTS_TO_PROCESS,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
    )


if __name__ == "__main__":
    main()
