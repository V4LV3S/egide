"""Etapas reutilizáveis do pré-processamento temporal e PCA para dados meteorológicos."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PcaComponents = int | float
PcaConfiguration = Mapping[str, PcaComponents]


@dataclass(frozen=True)
class TemporalPartitions:
    """Fatias cronológicas e disjuntas da série temporal."""

    train: slice
    validation: slice
    test: slice


def validate_pca_components(value: PcaComponents) -> PcaComponents:
    """Valida o número de componentes ou a fração de variância explicada."""
    if isinstance(value, bool):
        raise ValueError("pca_components deve ser inteiro positivo ou float em (0, 1].")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and 0.0 < value <= 1.0:
        return value
    raise ValueError("pca_components deve ser inteiro positivo ou float em (0, 1].")


def create_temporal_partitions(
    size: int, train_fraction: float, validation_fraction: float
) -> TemporalPartitions:
    """Cria as partições cronológicas treino/validação/teste sem sobreposição."""
    if size < 3:
        raise ValueError("A série precisa ter ao menos três instantes.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction deve estar entre 0 e 1.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve estar entre 0 e 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Treino e validação devem somar menos que 1.")

    train_end = int(size * train_fraction)
    validation_end = int(size * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end == train_end or validation_end >= size:
        raise ValueError("A série não possui instantes para todas as partições.")
    return TemporalPartitions(
        train=slice(0, train_end),
        validation=slice(train_end, validation_end),
        test=slice(validation_end, size),
    )


def split_labels(size: int, partitions: TemporalPartitions) -> np.ndarray:
    """Cria a coluna de identificação da partição para a matriz final."""
    labels = np.empty(size, dtype="U10")
    labels[partitions.train] = "train"
    labels[partitions.validation] = "validation"
    labels[partitions.test] = "test"
    return labels


def load_data_variable(path: Path) -> xr.DataArray:
    """Carrega a única variável de um NetCDF regional para a memória."""
    with xr.open_dataset(path) as dataset:
        names = list(dataset.data_vars)
        if len(names) != 1:
            raise ValueError(f"{path} deve conter uma variável; encontradas: {names}.")
        return dataset[names[0]].load()


def align_on_common_time(
    sources: list[tuple[Path, xr.DataArray]],
) -> list[tuple[Path, xr.DataArray]]:
    """Alinha variáveis pelos instantes comuns, em ordem cronológica."""
    common_time = sources[0][1].time.values
    for _, data_array in sources[1:]:
        common_time = np.intersect1d(common_time, data_array.time.values)
    if not len(common_time):
        raise ValueError("As variáveis não possuem instantes em comum.")
    return [(path, array.sel(time=common_time)) for path, array in sources]


def prepare_feature_matrix(data_array: xr.DataArray, source_path: Path) -> xr.DataArray:
    """Achata as dimensões espaciais, mantendo ``time`` como primeira dimensão."""
    if "time" not in data_array.dims:
        raise ValueError(f"{source_path} não possui a dimensão 'time'.")
    spatial_dimensions = tuple(dimension for dimension in data_array.dims if dimension != "time")
    if not spatial_dimensions:
        raise ValueError(f"{source_path} não possui dimensões espaciais.")
    matrix = data_array.stack(spatial_point=spatial_dimensions).transpose("time", "spatial_point")
    if not np.issubdtype(matrix.dtype, np.number) or np.isinf(matrix.values).any():
        raise ValueError(f"{source_path} possui valores não numéricos ou infinitos.")
    return matrix


def apply_log1p_transform(matrix: xr.DataArray, source_path: Path) -> xr.DataArray:
    """Aplica log1p a uma matriz não negativa, preservando valores ausentes."""
    finite_values = matrix.values[np.isfinite(matrix.values)]
    if (finite_values < 0).any():
        raise ValueError(f"{source_path} possui valores negativos; log1p requer valores não negativos.")
    return xr.apply_ufunc(np.log1p, matrix)


def transform_variable(
    data_array: xr.DataArray,
    source_path: Path,
    partitions: TemporalPartitions,
    pca_components: PcaComponents,
    *,
    apply_log1p: bool,
) -> xr.DataArray:
    """Ajusta imputação, normalização e PCA no treino; projeta todas as partes.

    Os pontos de grade sem observação no treino são removidos. Assim, mediana,
    escala e PCA nunca usam validação ou teste no ajuste de seus parâmetros.
    """
    matrix = prepare_feature_matrix(data_array, source_path)
    if apply_log1p:
        matrix = apply_log1p_transform(matrix, source_path)

    train_values = matrix.values[partitions.train]
    observed_in_train = ~np.isnan(train_values).all(axis=0)
    matrix = matrix.isel(spatial_point=np.flatnonzero(observed_in_train))
    if matrix.sizes["spatial_point"] == 0:
        raise ValueError(f"{source_path} não possui pontos observados no treino.")

    train = matrix.values[partitions.train]
    validation = matrix.values[partitions.validation]
    test = matrix.values[partitions.test]
    imputer = SimpleImputer(strategy="median")
    train = imputer.fit_transform(train)
    validation = imputer.transform(validation)
    test = imputer.transform(test)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    validation = scaler.transform(validation)
    test = scaler.transform(test)

    model = PCA(n_components=validate_pca_components(pca_components))
    projected_train = model.fit_transform(train)
    projected_validation = model.transform(validation)
    projected_test = model.transform(test)
    features = np.concatenate((projected_train, projected_validation, projected_test))
    component_dimension = f"{data_array.name}_pca_component"
    return xr.DataArray(
        features,
        dims=("time", component_dimension),
        coords={
            "time": matrix.time.values,
            component_dimension: np.arange(features.shape[1]),
        },
        name=f"{data_array.name}_pca",
        attrs={
            "source_variable": data_array.name,
            "pca_components_requested": str(validate_pca_components(pca_components)),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "normalization_before_pca": "train_only",
            "value_transformation": "log1p" if apply_log1p else "none",
            "pca_fit_scope": "train_only",
        },
    )


def normalize_pca_features(
    features: list[xr.DataArray], partitions: TemporalPartitions
) -> list[xr.DataArray]:
    """Normaliza a matriz concatenada de PCs com parâmetros do treino.

    Cada variável já passou por sua própria normalização e PCA. Esta etapa
    ajusta um segundo ``StandardScaler`` na matriz final ``X_train`` e o usa
    para transformar, sem reajuste, ``X_train``, ``X_validation`` e ``X_test``.
    """
    if not features:
        raise ValueError("É necessário ao menos um conjunto de componentes PCA.")

    component_counts = [feature.shape[1] for feature in features]
    combined_features = np.concatenate([feature.values for feature in features], axis=1)
    train = combined_features[partitions.train]
    validation = combined_features[partitions.validation]
    test = combined_features[partitions.test]

    final_scaler = StandardScaler()
    normalized_features = np.concatenate(
        (
            final_scaler.fit_transform(train),
            final_scaler.transform(validation),
            final_scaler.transform(test),
        )
    )

    normalized_arrays: list[xr.DataArray] = []
    start = 0
    for feature, component_count in zip(features, component_counts, strict=True):
        end = start + component_count
        normalized_feature = feature.copy(data=normalized_features[:, start:end])
        normalized_feature.attrs = {
            **feature.attrs,
            "final_normalization": "train_only",
        }
        normalized_arrays.append(normalized_feature)
        start = end
    return normalized_arrays


def build_pca_dataset(
    region_directory: Path,
    pca_configuration: PcaConfiguration,
    *,
    train_fraction: float,
    validation_fraction: float,
    variables_with_log1p: tuple[str, ...] = (),
) -> xr.Dataset:
    """Cria a matriz PCA regional e a normaliza finalmente sem vazamento."""
    if not pca_configuration:
        raise ValueError("A configuração PCA não pode estar vazia.")
    files = sorted(region_directory.glob("*.nc"))
    sources_by_name: dict[str, tuple[Path, xr.DataArray]] = {}
    for path in files:
        array = load_data_variable(path)
        if array.name in sources_by_name:
            raise ValueError(f"Variável duplicada na região: {array.name}.")
        sources_by_name[array.name] = (path, array)
    missing = sorted(set(pca_configuration) - set(sources_by_name))
    if missing:
        raise ValueError(f"Variáveis ausentes em {region_directory.name}: {', '.join(missing)}.")

    sources = align_on_common_time([sources_by_name[name] for name in pca_configuration])
    partitions = create_temporal_partitions(
        sources[0][1].sizes["time"], train_fraction, validation_fraction
    )
    features = [
        transform_variable(
            array,
            source_path,
            partitions,
            pca_configuration[array.name],
            apply_log1p=array.name in variables_with_log1p,
        )
        for source_path, array in sources
    ]
    normalized_features = normalize_pca_features(features, partitions)
    dataset = xr.merge(normalized_features)
    dataset = dataset.assign_coords(split=("time", split_labels(dataset.sizes["time"], partitions)))
    dataset.attrs.update(
        {
            "region": region_directory.name,
            "variables": ", ".join(pca_configuration),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
            "time_alignment": "inner",
            "preprocessing_fit_scope": "train_only",
            "final_normalization_fit_scope": "train_only",
            "pca_fit_scope": "train_only",
        }
    )
    return dataset


def pca_dataset_to_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    """Converte o Dataset PCA em matriz tabular para consumo direto do modelo."""
    if "split" not in dataset.coords:
        raise ValueError("Dataset PCA sem a coordenada 'split'.")
    columns: list[pd.DataFrame] = []
    for data_array in dataset.data_vars.values():
        values = data_array.transpose("time", *[dim for dim in data_array.dims if dim != "time"])
        matrix = np.asarray(values.values).reshape(values.sizes["time"], -1)
        names = [
            data_array.name if matrix.shape[1] == 1 else f"{data_array.name}_{index}"
            for index in range(matrix.shape[1])
        ]
        columns.append(pd.DataFrame(matrix, index=values.time.values, columns=names))
    dataframe = pd.concat(columns, axis=1)
    dataframe.index.name = "time"
    dataframe = dataframe.reset_index()
    dataframe.insert(1, "split", dataset.split.values)
    return dataframe


def create_region_parquet_files(
    input_directory: Path,
    output_directory: Path,
    pca_configuration: PcaConfiguration,
    *,
    train_fraction: float,
    validation_fraction: float,
    variables_with_log1p: tuple[str, ...] = (),
    regions_to_process: tuple[str, ...] | None = None,
    overwrite_existing_outputs: bool = False,
) -> list[Path]:
    """Gera um Parquet PCA final para cada região selecionada."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_directory}")

    region_directories = (
        [input_directory / region for region in regions_to_process]
        if regions_to_process is not None
        else sorted(path for path in input_directory.iterdir() if path.is_dir())
    )
    missing = [str(path) for path in region_directories if not path.is_dir()]
    if missing:
        missing_regions = ", ".join(missing)
        raise FileNotFoundError(f"Regiões não encontradas: {missing_regions}")

    output_paths: list[Path] = []
    for region_directory in region_directories:
        output_path = output_directory / region_directory.name / "pca_features.parquet"
        if output_path.exists() and not overwrite_existing_outputs:
            print(f"Ignorando saída existente: {output_path}")
            continue

        dataset = build_pca_dataset(
            region_directory,
            pca_configuration,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            variables_with_log1p=variables_with_log1p,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pca_dataset_to_dataframe(dataset).to_parquet(output_path, index=False)
        print(f"{region_directory.name} -> {output_path}")
        output_paths.append(output_path)
    return output_paths
