"""Cria features PCA de vento e solar a partir dos NetCDFs recortados.

Para cada regiao, o fluxo e executado nesta ordem:

1. carregar e alinhar as variaveis no tempo;
2. imputar lacunas e, quando aplicável, normalizar a série inteira;
3. ajustar e aplicar a PCA na série inteira;
4. salvar as features e os metadados em NetCDF e Parquet.

Cada pasta de ``data/processed/meteoro-recortado`` gera um arquivo por grupo,
como ``ml/data/meteoro/{REGIAO}/meteoro_pca.nc`` e um Parquet por variável.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


PcaComponents = int | float
PcaComponentsByVariable = Mapping[str, PcaComponents]
PcaConfiguration = Mapping[str, PcaComponentsByVariable]

ROOT = Path(__file__).resolve().parents[2]

# CONFIGURACAO EDITAVEL
# Use um valor entre 0 e 1 para a variancia explicada (ex.: 0.95) ou um numero
# inteiro para definir a quantidade fixa de componentes (ex.: 20).
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"
OUTPUT_DIRECTORY = ROOT / "ml" / "data"
PCA_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "meteoro"
HOURLY_FEATURES_PATH = OUTPUT_DIRECTORY / "date_features.nc"
HOURLY_FEATURES_PARQUET_PATH = OUTPUT_DIRECTORY / "date_features.parquet"
DATA_FEATURES_PATH = OUTPUT_DIRECTORY / "data_features.nc"
DATA_FEATURES_PARQUET_PATH = OUTPUT_DIRECTORY / "data_features.parquet"

# A saída PCA é organizada diretamente por região. A separação temporal será
# feita posteriormente pelo fluxo de LSTM.
CREATE_COMBINED_DATASET = False

# Configure as variáveis e o número de componentes (inteiro) ou a variância
# explicada (float entre 0 e 1) usada na PCA.
PCA_CONFIGURATION: dict[str, dict[str, PcaComponents]] = {
    "meteoro": {
        "t2m": 0.95,
        "tp": 0.90,
        "ws100": 0.90,
        "sp": 0.90,
        "ssr": 0.90,
        "tcc": 0.90,
    },
}

# Essas variaveis ja estao na escala apropriada para a PCA. Elas ainda passam
# pela imputacao de valores ausentes, mas nao recebem StandardScaler.
VARIABLES_WITHOUT_STANDARDIZATION = ("tcc",)

# ``log1p`` preserva zeros e reduz a assimetria provocada por eventos de
# precipitacao muito altos. A transformacao acontece antes da imputacao,
# normalizacao e PCA, sem ajustar parametros com dados futuros.
VARIABLES_WITH_LOG1P = ("tp",)




def validate_pca_components(value: PcaComponents) -> PcaComponents:
    """Valida a quantidade de componentes ou a variancia explicada desejada."""
    if isinstance(value, bool):
        raise ValueError("pca_components deve ser um inteiro ou float positivo.")
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and 0.0 < value <= 1.0:
        return value
    raise ValueError(
        "pca_components deve ser um inteiro positivo ou um float entre 0 e 1."
    )


def validate_pca_configuration(
    configuration: PcaConfiguration,
) -> dict[str, dict[str, PcaComponents]]:
    """Valida e normaliza a configuracao PCA por dataset e variavel."""
    if not configuration:
        raise ValueError("PCA_CONFIGURATION precisa conter ao menos um dataset.")

    validated: dict[str, dict[str, PcaComponents]] = {}
    for dataset_name, components_by_variable in configuration.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("O nome de cada dataset PCA deve ser uma string nao vazia.")
        if not components_by_variable:
            raise ValueError(
                f"O dataset {dataset_name!r} precisa conter ao menos uma variavel."
            )

        validated[dataset_name] = {}
        for variable_name, pca_components in components_by_variable.items():
            if not isinstance(variable_name, str) or not variable_name:
                raise ValueError(
                    f"O dataset {dataset_name!r} possui um nome de variavel invalido."
                )
            validated[dataset_name][variable_name] = validate_pca_components(
                pca_components
            )
    return validated


def load_data_variable(input_path: Path) -> xr.DataArray:
    """Carrega a unica variavel de um NetCDF de entrada."""
    with xr.open_dataset(input_path) as dataset:
        variable_names = list(dataset.data_vars)
        if len(variable_names) != 1:
            raise ValueError(
                f"{input_path} deve conter exatamente uma variavel; "
                f"encontradas: {variable_names}."
            )
        return dataset[variable_names[0]].load()


def prepare_feature_matrix(data_array: xr.DataArray, source_path: Path) -> xr.DataArray:
    """Organiza uma variavel espacial como matriz de tempo por ponto de grade."""
    if "time" not in data_array.dims:
        raise ValueError(f"{source_path} nao possui a dimensao 'time'.")

    spatial_dimensions = tuple(
        dimension for dimension in data_array.dims if dimension != "time"
    )
    if not spatial_dimensions:
        raise ValueError(f"{source_path} nao possui dimensoes espaciais.")

    matrix = data_array.stack(spatial_point=spatial_dimensions).transpose(
        "time", "spatial_point"
    )
    matrix = matrix.dropna(dim="spatial_point", how="all")
    if matrix.sizes["spatial_point"] == 0:
        raise ValueError(f"{source_path} nao possui pontos validos apos o recorte.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{source_path} possui valores nao numericos.")
    if np.isinf(matrix.values).any():
        raise ValueError(f"{source_path} possui valores infinitos.")
    return matrix


def apply_log1p_transform(
    matrix: xr.DataArray, source_path: Path
) -> xr.DataArray:
    """Aplica ``log1p`` a uma matriz nao negativa, preservando valores ausentes."""
    values = matrix.values
    finite_values = values[np.isfinite(values)]
    if (finite_values < 0).any():
        raise ValueError(
            f"{source_path} possui valores negativos; log1p requer valores nao negativos."
        )

    transformed = xr.apply_ufunc(np.log1p, matrix)
    transformed.attrs = dict(matrix.attrs)
    transformed.attrs["value_transformation"] = "log1p"
    return transformed



def prepare_pca_input(
    matrix: xr.DataArray, source_path: Path, standardize: bool
) -> tuple[xr.DataArray, np.ndarray]:
    """Remove pontos totalmente ausentes e ajusta o preparo na série inteira."""
    has_observation = ~np.isnan(matrix.values).all(axis=0)
    filtered_matrix = matrix.isel(spatial_point=np.flatnonzero(has_observation))
    if filtered_matrix.sizes["spatial_point"] == 0:
        raise ValueError(f"{source_path} não possui pontos observados.")

    values = SimpleImputer(strategy="median").fit_transform(filtered_matrix.values)
    if standardize:
        values = StandardScaler().fit_transform(values)
    return filtered_matrix, values


def apply_pca(
    data_array: xr.DataArray,
    source_path: Path,
    pca_components: PcaComponents = 0.95,
) -> xr.DataArray:
    """Ajusta o preparo e a PCA usando toda a série temporal disponível."""
    matrix = prepare_feature_matrix(data_array, source_path)
    applies_log1p = data_array.name in VARIABLES_WITH_LOG1P
    if applies_log1p:
        matrix = apply_log1p_transform(matrix, source_path)
    standardize = data_array.name not in VARIABLES_WITHOUT_STANDARDIZATION
    matrix, values = prepare_pca_input(matrix, source_path, standardize)
    pca = PCA(n_components=validate_pca_components(pca_components))
    features = pca.fit_transform(values)

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
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "standardization": "standard" if standardize else "not applied",
            "value_transformation": "log1p" if applies_log1p else "none",
            "fit_scope": "full_time_series",
        },
    )


def align_on_common_time(
    sources: list[tuple[Path, xr.DataArray]],
) -> list[tuple[Path, xr.DataArray]]:
    """Restringe as variaveis aos instantes compartilhados por toda a regiao."""
    common_time = sources[0][1].time.values
    for _, data_array in sources[1:]:
        common_time = np.intersect1d(common_time, data_array.time.values)
    if len(common_time) == 0:
        raise ValueError("As variaveis da regiao nao possuem instantes em comum.")
    return [
        (source_path, data_array.sel(time=common_time))
        for source_path, data_array in sources
    ]


def output_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    """Define compressao para todas as features PCA."""
    return {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in dataset.data_vars
    }


def load_and_select_sources(
    region_directory: Path, variable_names: tuple[str, ...]
) -> list[tuple[Path, xr.DataArray]]:
    """Carrega as variaveis solicitadas de uma regiao e alinha seus tempos."""
    input_files = sorted(region_directory.glob("*.nc"))
    if not input_files:
        raise FileNotFoundError(f"Nenhum NetCDF encontrado em {region_directory}.")

    sources = [
        (input_path, load_data_variable(input_path)) for input_path in input_files
    ]
    sources_by_variable = {
        data_array.name: (input_path, data_array)
        for input_path, data_array in sources
    }
    unavailable_variables = sorted(set(variable_names) - sources_by_variable.keys())
    if unavailable_variables:
        available_variables = ", ".join(sorted(sources_by_variable))
        raise ValueError(
            f"Variaveis ausentes em {region_directory.name}: "
            f"{', '.join(unavailable_variables)}. Disponiveis: {available_variables}."
        )
    return align_on_common_time([sources_by_variable[name] for name in variable_names])


def create_pca_dataset(
    region_directory: Path,
    dataset_name: str,
    pca_components_by_variable: PcaComponentsByVariable,
) -> xr.Dataset:
    """Executa preparo e PCA na série inteira para um grupo regional."""
    validated_configuration = validate_pca_configuration(
        {dataset_name: pca_components_by_variable}
    )[dataset_name]
    variable_names = tuple(validated_configuration)

    features = [
        apply_pca(
            data_array,
            input_path,
            validated_configuration[data_array.name],
        )
        for input_path, data_array in load_and_select_sources(
            region_directory, variable_names
        )
    ]
    dataset = xr.merge(features)
    dataset.attrs.update(
        {
            "region": region_directory.name,
            "dataset_name": dataset_name,
            "variables": ", ".join(variable_names),
            "variables_without_standardization": ", ".join(
                name
                for name in variable_names
                if name in VARIABLES_WITHOUT_STANDARDIZATION
            ),
            "variables_with_log1p": ", ".join(
                name for name in variable_names if name in VARIABLES_WITH_LOG1P
            ),
            "pca_components_requested": json.dumps(validated_configuration),
            "pca_fit_scope": "full_time_series",
            "time_alignment": "inner",
        }
    )
    return dataset


def save_pca_dataset(dataset: xr.Dataset, output_path: Path) -> None:
    """Salva as features PCA em NetCDF comprimido."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path, format="NETCDF4", encoding=output_encoding(dataset))


def load_dataset(path: Path) -> xr.Dataset:
    """Carrega um NetCDF em memoria para que seu arquivo possa ser fechado."""
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def prefix_pca_features(dataset: xr.Dataset, prefix: str) -> xr.Dataset:
    """Prefixa variaveis e dimensoes PCA para evitar colisao entre regioes."""
    without_split = dataset.drop_vars("split", errors="ignore")
    rename_map = {
        name: f"{prefix}_{name}" for name in without_split.data_vars
    }
    rename_map.update(
        {
            dimension: f"{prefix}_{dimension}"
            for dimension in without_split.dims
            if dimension != "time"
        }
    )
    return without_split.rename(rename_map)


def feature_columns(data_array: xr.DataArray) -> xr.DataArray:
    """Converte uma variavel temporal em colunas nomeadas de uma matriz ML."""
    if "time" not in data_array.dims:
        raise ValueError(f"{data_array.name} nao possui a dimensao 'time'.")

    non_time_dimensions = tuple(
        dimension for dimension in data_array.dims if dimension != "time"
    )
    values = np.asarray(data_array.transpose("time", *non_time_dimensions).values)
    values = values.reshape(data_array.sizes["time"], -1)
    feature_names = [
        data_array.name if values.shape[1] == 1 else f"{data_array.name}_{index}"
        for index in range(values.shape[1])
    ]
    return xr.DataArray(
        values,
        dims=("time", "feature"),
        coords={"time": data_array.time.values, "feature": feature_names},
        name="X",
    )


def build_ml_feature_matrix(dataset: xr.Dataset) -> xr.DataArray:
    """Achata todas as features em uma matriz ``X(time, feature)`` para ML."""
    if not dataset.data_vars:
        raise ValueError("Dataset sem variaveis para construir a matriz de features.")
    columns = [feature_columns(data_array) for data_array in dataset.data_vars.values()]
    matrix = xr.concat(columns, dim="feature").transpose("time", "feature")
    matrix.name = "X"
    matrix.attrs = {
        "description": "Matriz de entrada para modelos de machine learning.",
        "feature_count": matrix.sizes["feature"],
    }
    return matrix




def feature_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    """Converte ``X(time, feature)`` em tabela para Parquet sem partições."""
    if "X" not in dataset.data_vars:
        raise ValueError("Dataset precisa conter a matriz 'X'.")

    dataframe = dataset["X"].to_pandas()
    dataframe.index.name = "time"
    return dataframe.reset_index()


def save_parquet_features(dataset: xr.Dataset, output_path: Path) -> list[Path]:
    """Grava uma única tabela Parquet sem partições temporais."""
    dataframe = feature_dataframe(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return [output_path]


def save_date_features_parquet(input_path: Path, output_path: Path) -> Path:
    """Converte ``date_features.nc`` em uma tabela Parquet compartilhada.

    A tabela mantem ``time`` e uma coluna para cada feature horaria. Ela nao
    não recebe classificação temporal; a separação será definida no treinamento.
    """
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de features horarias inexistente: {input_path}"
        )

    dataset = load_dataset(input_path)
    if "time" not in dataset.dims:
        raise ValueError(f"{input_path} nao possui a dimensao 'time'.")

    dataframe = build_ml_feature_matrix(dataset).to_pandas()
    dataframe.index.name = "time"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.reset_index().to_parquet(output_path, index=False)
    return output_path
def save_variable_parquets(
    dataset: xr.Dataset, output_directory: Path, region: str, dataset_name: str
) -> list[Path]:
    """Grava um Parquet por variavel PCA em ``REGIAO/DATASET``.

    Cada tabela possui ``time`` e uma coluna por componente PCA. A separação
    temporal é responsabilidade do fluxo de LSTM posterior.
    """

    variable_directory = output_directory / region
    variable_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for variable_name, data_array in dataset.data_vars.items():
        columns = feature_columns(data_array)
        dataframe = columns.to_pandas()
        dataframe.index.name = "time"
        dataframe = dataframe.reset_index()
        output_path = variable_directory / f"{variable_name}.parquet"
        dataframe.to_parquet(output_path, index=False)
        paths.append(output_path)
    return paths


def create_data_features(
    hourly_features_path: Path,
    pca_paths: list[Path],
    output_path: Path,
) -> Path:
    """Une features horárias e PCA no eixo ``time`` e grava ``data_features``.

    A intersecao temporal e usada para garantir que toda linha final tenha as
    features horarias e meteorologicas. Cada feature PCA recebe os prefixos de
    regiao e dataset, por exemplo ``BA_SE_solar_ssr_pca``.
    """
    if not hourly_features_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de features horarias inexistente: {hourly_features_path}"
        )
    if not pca_paths:
        raise ValueError("Nenhum arquivo PCA foi fornecido para concatenacao.")

    hourly_features = load_dataset(hourly_features_path)
    if "time" not in hourly_features.dims:
        raise ValueError(f"{hourly_features_path} nao possui a dimensao 'time'.")

    datasets = [hourly_features]
    source_names = [hourly_features_path.name]
    for pca_path in pca_paths:
        pca_dataset = load_dataset(pca_path)
        if "time" not in pca_dataset.dims:
            raise ValueError(f"{pca_path} nao possui a dimensao 'time'.")
        region = pca_dataset.attrs.get("region")
        dataset_name = pca_dataset.attrs.get("dataset_name")
        if not region or not dataset_name:
            raise ValueError(
                f"{pca_path} precisa conter os atributos 'region' e 'dataset_name'."
            )
        datasets.append(prefix_pca_features(pca_dataset, f"{region}_{dataset_name}"))
        source_names.append(pca_path.name)

    combined = xr.merge(datasets, join="inner", compat="no_conflicts")
    if combined.sizes.get("time", 0) == 0:
        raise ValueError("As features horarias e PCA nao possuem instantes em comum.")
    combined["X"] = build_ml_feature_matrix(combined)
    combined.attrs.update(
        {
            "hourly_features_source": hourly_features_path.name,
            "pca_feature_sources": ", ".join(source_names[1:]),
            "time_alignment": "inner",
            "pca_feature_prefix": "{REGIAO}_{DATASET}",
        }
    )
    save_pca_dataset(combined, output_path)
    save_parquet_features(combined, output_path.with_suffix(".parquet"))
    return output_path


def process_region(
    region_directory: Path,
    output_directory: Path,
    dataset_name: str,
    pca_components_by_variable: PcaComponentsByVariable,
) -> Path:
    """Cria e persiste um grupo PCA, com Parquets diretamente na pasta regional."""
    dataset = create_pca_dataset(
        region_directory, dataset_name, pca_components_by_variable
    )
    group_directory = output_directory / region_directory.name
    output_path = group_directory / f"{dataset_name}_pca.nc"
    save_pca_dataset(dataset, output_path)
    save_variable_parquets(
        dataset, output_directory, region_directory.name, dataset_name
    )
    return output_path


def process_all_regions(
    input_directory: Path = INPUT_DIRECTORY,
    output_directory: Path = PCA_OUTPUT_DIRECTORY,
    pca_configuration: PcaConfiguration = PCA_CONFIGURATION,
) -> list[Path]:
    """Gera um dataset PCA para cada configuracao e pasta regional."""
    region_directories = sorted(path for path in input_directory.iterdir() if path.is_dir())
    if not region_directories:
        raise FileNotFoundError(f"Nenhuma pasta regional encontrada em {input_directory}.")
    validated_configuration = validate_pca_configuration(pca_configuration)

    output_paths: list[Path] = []
    for region_directory in region_directories:
        for dataset_name, components_by_variable in validated_configuration.items():
            output_paths.append(
                process_region(
                    region_directory,
                    output_directory,
                    dataset_name,
                    components_by_variable,
                )
            )
    return output_paths


def main() -> None:
    """Executa a PCA e grava os Parquets em ``ml/data/meteoro``."""
    outputs = process_all_regions(
        INPUT_DIRECTORY, PCA_OUTPUT_DIRECTORY, PCA_CONFIGURATION
    )
    for output_path in outputs:
        print(f"Concluido: {output_path}")
    if CREATE_COMBINED_DATASET:
        data_features_path = create_data_features(
            HOURLY_FEATURES_PATH, outputs, DATA_FEATURES_PATH
        )
        print(f"Concluido: {data_features_path}")


if __name__ == "__main__":
    main()
