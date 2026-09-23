"""Reduz a resolução espacial dos NetCDFs meteorológicos consolidados.

Edite ``COARSENING_FACTOR`` antes de executar este arquivo pela IDE. O script
processa cada ``.nc`` de ``data/processed/meteoro-total`` separadamente e
mantém a dimensão temporal intacta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import date2num


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-total"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-coarsened"

# Ajuste este número antes de rodar. Ex.: 5 transforma a grade 401 x 401 em
# 80 x 80, descartando a última linha e coluna incompletas.
COARSENING_FACTOR = 3

# Quantos instantes são carregados e gravados de cada vez. Reduza caso a RAM
# disponível seja limitada.
TIME_CHUNK_SIZE = 24 * 31 * 12

# ``trim`` descarta bordas que não completam uma janela de COARSENING_FACTOR.
BOUNDARY = "trim"
OVERWRITE_EXISTING_OUTPUTS = False


def validate_configuration() -> None:
    """Verifica valores que poderiam produzir uma saída incorreta."""
    if not isinstance(COARSENING_FACTOR, int) or COARSENING_FACTOR < 1:
        raise ValueError("COARSENING_FACTOR deve ser um inteiro maior ou igual a 1.")
    if not isinstance(TIME_CHUNK_SIZE, int) or TIME_CHUNK_SIZE < 1:
        raise ValueError("TIME_CHUNK_SIZE deve ser um inteiro maior ou igual a 1.")
    if BOUNDARY not in {"trim", "pad", "exact"}:
        raise ValueError("BOUNDARY deve ser 'trim', 'pad' ou 'exact'.")


def coarsen_chunk(dataset: xr.Dataset) -> xr.Dataset:
    """Calcula a média espacial para cada instante do bloco recebido."""
    required_dimensions = {"latitude", "longitude"}
    missing_dimensions = required_dimensions - set(dataset.dims)
    if missing_dimensions:
        missing = ", ".join(sorted(missing_dimensions))
        raise ValueError(f"Dataset sem dimensões espaciais esperadas: {missing}.")

    return dataset.coarsen(
        latitude=COARSENING_FACTOR,
        longitude=COARSENING_FACTOR,
        boundary=BOUNDARY,
    ).mean(keep_attrs=True)


def output_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    """Comprime variáveis de dados sem alterar a codificação das coordenadas."""
    return {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in dataset.data_vars
    }


def append_chunk_to_netcdf(chunk: xr.Dataset, output_path: Path) -> None:
    """Anexa um bloco a um NetCDF com ``time`` ilimitado.

    ``xarray.to_netcdf`` não oferece uma opção de append ao longo de uma
    dimensão. A API netCDF4 permite ampliar a dimensão ilimitada sem manter os
    blocos anteriores na memória.
    """
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


def coarsen_file(input_path: Path, output_path: Path) -> None:
    """Faz o coarsening de um arquivo, lendo apenas um bloco temporal por vez."""
    with xr.open_dataset(input_path) as source:
        if "time" not in source.dims:
            raise ValueError(f"{input_path.name} não possui a dimensão 'time'.")

        total_times = source.sizes["time"]
        if total_times == 0:
            raise ValueError(f"{input_path.name} não possui instantes para processar.")

        print(f"Processando {input_path.name} ({total_times} instantes)...")
        first_chunk = coarsen_chunk(source.isel(time=slice(0, TIME_CHUNK_SIZE)))
        first_chunk.to_netcdf(
            output_path,
            mode="w",
            format="NETCDF4",
            unlimited_dims=["time"],
            encoding=output_encoding(first_chunk),
        )

        for start in range(TIME_CHUNK_SIZE, total_times, TIME_CHUNK_SIZE):
            end = min(start + TIME_CHUNK_SIZE, total_times)
            chunk = coarsen_chunk(source.isel(time=slice(start, end)))
            append_chunk_to_netcdf(chunk, output_path)
            print(f"  {end}/{total_times} instantes gravados")


def main() -> None:
    """Coarsen todos os NetCDFs de entrada um a um."""
    validate_configuration()
    input_files = sorted(INPUT_DIRECTORY.glob("*.nc"))
    if not input_files:
        raise FileNotFoundError(f"Nenhum arquivo .nc encontrado em {INPUT_DIRECTORY}.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for input_path in input_files:
        output_path = OUTPUT_DIRECTORY / input_path.name
        if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
            print(f"Ignorando {input_path.name}: saída já existe.")
            continue

        coarsen_file(input_path, output_path)
        print(f"Concluído: {output_path}")


if __name__ == "__main__":
    main()
