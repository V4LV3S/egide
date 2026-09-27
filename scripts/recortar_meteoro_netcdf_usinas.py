"""Trata e recorta NetCDFs nos quatro pontos mais próximos de conjuntos solares.

As conversões físicas de precipitação e radiação reutilizam exatamente o
pipeline de ``recortar_meteoro_netcdf.py``. Cada conjunto de usinas recebe um
diretório com os quatro NetCDFs recortados.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:  # Permite importar nos testes e executar diretamente pela IDE.
    from .recortar_meteoro_netcdf import (
        append_chunk_to_netcdf,
        output_encoding,
        prepare_data_before_crop,
    )
except ImportError:  # pragma: no cover - usado somente na execução direta.
    from recortar_meteoro_netcdf import (
        append_chunk_to_netcdf,
        output_encoding,
        prepare_data_before_crop,
    )


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-coarsened"
PLANTS_PATH = ROOT / "data" / "cadastro" / "conj_usinas_sol.parquet"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado-usinas"

VARIABLE_FILES = ("2t.nc", "ssr.nc", "tcc.nc", "tp.nc")

# ``None`` processa todos. Caso contrário, use nomes da coluna ``conjunto``,
# por exemplo: ("Assu Sol", "Futura").
PLANTS_TO_PROCESS: tuple[str, ...] | None = None
TIME_CHUNK_SIZE = 24 * 31 * 48
OVERWRITE_EXISTING_OUTPUTS = True

REQUIRED_PLANT_COLUMNS = ("conjunto", "latitude", "longitude")


def validate_configuration() -> None:
    """Valida configurações ajustáveis e caminhos de entrada."""
    if not isinstance(TIME_CHUNK_SIZE, int) or TIME_CHUNK_SIZE < 1:
        raise ValueError("TIME_CHUNK_SIZE deve ser um inteiro maior ou igual a 1.")
    if not INPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Diretório meteorológico não encontrado: {INPUT_DIRECTORY}"
        )
    if not PLANTS_PATH.is_file():
        raise FileNotFoundError(f"Cadastro de usinas não encontrado: {PLANTS_PATH}")


def plant_directory_name(name: str) -> str:
    """Converte nome do cadastro para o padrão de diretórios individuais."""
    normalized = re.sub(r"[^\w]+", "_", name.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("Nome de conjunto vazio ou inválido.")
    return f"conj_{normalized}"


def load_plant_centres(
    path: Path,
    selected_names: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Carrega centros, valida unicidade e aplica seleção opcional."""
    plants = pd.read_parquet(path)
    missing = set(REQUIRED_PLANT_COLUMNS) - set(plants.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    plants = plants.loc[:, REQUIRED_PLANT_COLUMNS].copy()
    if plants.empty:
        raise ValueError("Cadastro de conjuntos solares está vazio.")
    if plants["conjunto"].isna().any() or plants["conjunto"].duplicated().any():
        raise ValueError("Nomes dos conjuntos devem ser preenchidos e únicos.")
    for coordinate in ("latitude", "longitude"):
        plants[coordinate] = pd.to_numeric(plants[coordinate], errors="raise")
        if plants[coordinate].isna().any():
            raise ValueError(f"Coordenada ausente na coluna {coordinate}.")

    if selected_names is None:
        return plants.sort_values("conjunto", ignore_index=True)

    selected = tuple(dict.fromkeys(selected_names))
    available = set(plants["conjunto"])
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(
            f"Conjunto(s) não encontrado(s): {', '.join(unknown)}. "
            f"Disponíveis: {', '.join(sorted(available))}."
        )
    indexed = plants.set_index("conjunto")
    return indexed.loc[list(selected)].reset_index()


def crop_to_square(
    dataset: xr.Dataset,
    *,
    centre_latitude: float,
    centre_longitude: float,
) -> xr.Dataset:
    """Seleciona duas latitudes por duas longitudes mais próximas do centro."""
    required_coordinates = {"latitude", "longitude"}
    missing = required_coordinates - set(dataset.coords)
    if missing:
        raise ValueError(f"Coordenadas espaciais ausentes: {sorted(missing)}")

    selected_indices: dict[str, np.ndarray] = {}
    for coordinate, centre in (
        ("latitude", centre_latitude),
        ("longitude", centre_longitude),
    ):
        values = dataset[coordinate].to_numpy()
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(
                f"A coordenada {coordinate} deve conter ao menos dois pontos."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"A coordenada {coordinate} possui valores inválidos.")
        if not values.min() <= centre <= values.max():
            raise ValueError(
                f"Centro {centre} fora dos limites de {coordinate}: "
                f"{values.min()} a {values.max()}."
            )

        # Ordenar novamente os índices preserva a orientação original dos
        # eixos, inclusive latitude decrescente nos arquivos ERA5.
        nearest = np.argsort(np.abs(values - centre), kind="stable")[:2]
        selected_indices[coordinate] = np.sort(nearest)

    cropped = dataset.isel(**selected_indices)
    latitude_step = abs(float(cropped.latitude.diff("latitude").item()))
    longitude_step = abs(float(cropped.longitude.diff("longitude").item()))
    if not np.isclose(latitude_step, longitude_step):
        raise ValueError(
            "Os quatro pontos mais próximos não formam quadrado na grade: "
            f"passo latitude={latitude_step}°, "
            f"passo longitude={longitude_step}°."
        )
    return cropped


def crop_file_for_plant(
    input_path: Path,
    output_path: Path,
    *,
    plant_name: str,
    centre_latitude: float,
    centre_longitude: float,
) -> None:
    """Aplica tratamento físico, recorta quadrado e grava em blocos temporais."""
    with xr.open_dataset(input_path) as source:
        if "time" not in source.dims:
            raise ValueError(f"{input_path.name} não possui a dimensão 'time'.")

        prepared = prepare_data_before_crop(source)
        cropped = crop_to_square(
            prepared,
            centre_latitude=centre_latitude,
            centre_longitude=centre_longitude,
        )
        total_times = cropped.sizes["time"]
        if total_times == 0:
            raise ValueError(f"{input_path.name} não possui instantes para processar.")

        grid_shape = (
            cropped.sizes["latitude"],
            cropped.sizes["longitude"],
        )
        print(
            f"Processando {plant_name}/{input_path.name}: grade "
            f"{grid_shape[0]}x{grid_shape[1]}, {total_times} instantes..."
        )
        first_chunk = cropped.isel(time=slice(0, TIME_CHUNK_SIZE))
        first_chunk.to_netcdf(
            output_path,
            mode="w",
            format="NETCDF4",
            unlimited_dims=["time"],
            encoding=output_encoding(first_chunk),
        )
        for start in range(TIME_CHUNK_SIZE, total_times, TIME_CHUNK_SIZE):
            end = min(start + TIME_CHUNK_SIZE, total_times)
            append_chunk_to_netcdf(cropped.isel(time=slice(start, end)), output_path)
            print(f"  {end}/{total_times} instantes gravados")


def main() -> None:
    """Recorta quatro variáveis para cada conjunto solar selecionado."""
    validate_configuration()
    input_paths = [INPUT_DIRECTORY / file_name for file_name in VARIABLE_FILES]
    missing_files = [str(path) for path in input_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"NetCDFs de entrada não encontrados: {', '.join(missing_files)}"
        )

    plants = load_plant_centres(PLANTS_PATH, PLANTS_TO_PROCESS)
    for plant in plants.itertuples(index=False):
        directory_name = plant_directory_name(plant.conjunto)
        plant_output_directory = OUTPUT_DIRECTORY / directory_name
        plant_output_directory.mkdir(parents=True, exist_ok=True)
        for input_path in input_paths:
            output_path = plant_output_directory / input_path.name
            if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
                print(f"Ignorando {directory_name}/{input_path.name}: saída existente.")
                continue
            crop_file_for_plant(
                input_path,
                output_path,
                plant_name=directory_name,
                centre_latitude=float(plant.latitude),
                centre_longitude=float(plant.longitude),
            )
            print(f"Concluído: {output_path}")


if __name__ == "__main__":
    main()
