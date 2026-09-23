"""Recorta os NetCDFs meteorológicos coarsened pelos shapefiles dos estados.

O recorte usa Salem para reduzir a grade ao retângulo de cada shapefile e
mascarar as células que ficam fora do polígono. Execute este arquivo pela IDE,
sem argumentos de terminal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import salem  # Registra o acessor ``.salem`` no xarray.
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import date2num


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-coarsened"
SHAPES_DIRECTORY = ROOT / "data" / "shapes-estados"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"

# Use ``None`` para todos os shapefiles disponíveis ou informe nomes sem a
# extensão, por exemplo: ("BA_SE", "PE").
SHAPES_TO_PROCESS: tuple[str, ...] | None = None

# Processa poucos instantes por vez para limitar o uso de memória.
TIME_CHUNK_SIZE = 24 * 31 * 12

# Inclui também células da grade tocadas pela borda do estado. Isto evita que
# estados pequenos desapareçam após um coarsening mais agressivo.
ALL_TOUCHED = True
OVERWRITE_EXISTING_OUTPUTS = False


def validate_configuration() -> None:
    """Valida os valores ajustáveis no início do arquivo."""
    if not isinstance(TIME_CHUNK_SIZE, int) or TIME_CHUNK_SIZE < 1:
        raise ValueError("TIME_CHUNK_SIZE deve ser um inteiro maior ou igual a 1.")


def select_shapes(directory: Path, names: tuple[str, ...] | None) -> list[Path]:
    """Lista os shapefiles solicitados e falha cedo para nomes inexistentes."""
    available = {path.stem: path for path in sorted(directory.glob("*.shp"))}
    if not available:
        raise FileNotFoundError(f"Nenhum shapefile encontrado em {directory}.")
    if names is None:
        return list(available.values())

    selected = tuple(dict.fromkeys(names))
    unknown = sorted(set(selected) - available.keys())
    if unknown:
        raise ValueError(
            "Shapefile(s) não encontrado(s): "
            f"{', '.join(unknown)}. Disponíveis: {', '.join(available)}."
        )
    return [available[name] for name in selected]


def output_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    """Comprime as variáveis meteorológicas da saída."""
    return {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in dataset.data_vars
    }


def append_chunk_to_netcdf(chunk: xr.Dataset, output_path: Path) -> None:
    """Anexa um bloco temporal ao NetCDF de saída."""
    with NetCDFDataset(output_path, mode="a") as target:
        start = len(target.dimensions["time"])
        time_size = chunk.sizes["time"]
        for name, variable in chunk.variables.items():
            if "time" not in variable.dims:
                continue

            time_axis = variable.dims.index("time")
            target_slice = [slice(None)] * variable.ndim
            target_slice[time_axis] = slice(start, start + time_size)
            values = variable.values
            target_variable = target.variables[name]
            if np.issubdtype(values.dtype, np.datetime64):
                values = date2num(
                    values.astype("datetime64[us]").astype(object),
                    units=target_variable.units,
                    calendar=getattr(target_variable, "calendar", "standard"),
                )
            target_variable[tuple(target_slice)] = values


def crop_file(input_path: Path, shape_path: Path, output_path: Path) -> None:
    """Recorta um arquivo usando Salem e o grava em blocos temporais."""
    with xr.open_dataset(input_path) as source:
        if "time" not in source.dims:
            raise ValueError(f"{input_path.name} não possui a dimensão 'time'.")

        # ``subset`` reduz a área processada; ``roi`` aplica a máscara precisa
        # do polígono e preserva a caixa delimitadora resultante.
        subset = source.salem.subset(shape=str(shape_path))
        cropped = subset.salem.roi(shape=str(shape_path), all_touched=ALL_TOUCHED)
        total_times = cropped.sizes["time"]
        if total_times == 0:
            raise ValueError(f"{input_path.name} não possui instantes para processar.")

        print(f"Processando {shape_path.stem}/{input_path.name} ({total_times} instantes)...")
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
    """Recorta todos os NetCDFs para cada shapefile selecionado."""
    validate_configuration()
    input_files = sorted(INPUT_DIRECTORY.glob("*.nc"))
    if not input_files:
        raise FileNotFoundError(f"Nenhum arquivo .nc encontrado em {INPUT_DIRECTORY}.")

    for shape_path in select_shapes(SHAPES_DIRECTORY, SHAPES_TO_PROCESS):
        shape_output_directory = OUTPUT_DIRECTORY / shape_path.stem
        shape_output_directory.mkdir(parents=True, exist_ok=True)
        for input_path in input_files:
            output_path = shape_output_directory / input_path.name
            if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
                print(f"Ignorando {shape_path.stem}/{input_path.name}: saída já existe.")
                continue

            crop_file(input_path, shape_path, output_path)
            print(f"Concluído: {output_path}")


if __name__ == "__main__":
    main()
