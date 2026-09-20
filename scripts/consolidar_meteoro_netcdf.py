"""Converte e consolida arquivos GRIB mensais de meteorologia por variável.

Exemplo de entrada: ``tp_2024_01.grib``. O resultado será ``tp.nc`` no mesmo
diretório, com a dimensão ``time`` ordenada e sem valores repetidos.
"""

from __future__ import annotations

import gc
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import date2num


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "data" / "processed" / "meteoro"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-total"
DEFAULT_REFERENCE_COORDINATES = DEFAULT_DIRECTORY / "coords_era5land.nc"
MONTHLY_FILE = re.compile(r"^(?P<variable>.+)_\d{4}_\d{2}\.grib$")

# Estes campos vêm da grade do ERA5 (0,25°) e precisam ser trazidos para a
# grade do ERA5-Land (0,1°) antes da concatenação dos arquivos mensais.
ERA5_VARIABLES_TO_INTERPOLATE = ("100u", "100v", "tcc")
ERA5_LAND_VARIABLES_TO_FLATTEN_TIME = ("2t", "sp", "tp", "ssr", "str")

# Configuração para execução direta pela IDE. Use ``None`` para consolidar
# todas as variáveis ou informe uma tupla, por exemplo: ("100u", "100v", "tcc").
VARIABLES_TO_CONSOLIDATE: tuple[str, ...] | None = ('tp', )
OVERWRITE_EXISTING_OUTPUTS = False

# Cada bloco é interpolado e escrito antes de ler o próximo.  Reduza este
# valor caso a máquina ainda tenha pouca memória; 24 corresponde a um dia de
# dados horários e evita materializar um mês (ou toda a série) na RAM.
TIME_CHUNK_SIZE = 24 * 60  # Aproximadamente um mês de dados horários
DISK_SPACE_SAFETY_FACTOR = 1.10


@dataclass(frozen=True)
class WriteRun:
    """Índices de um arquivo que devem ser anexados, já na ordem final."""

    path: Path
    positions: tuple[int, ...]


def monthly_files_by_variable(directory: Path) -> dict[str, list[Path]]:
    """Agrupa todos os arquivos mensais por variável, sem recorte temporal."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in directory.glob("*.grib"):
        match = MONTHLY_FILE.match(path.name)
        if match:
            groups[match["variable"]].append(path)

    return {variable: sorted(paths) for variable, paths in sorted(groups.items())}


def select_variables(
    groups: dict[str, list[Path]], variables: Iterable[str] | None
) -> dict[str, list[Path]]:
    """Retorna os grupos solicitados e falha cedo para nomes inexistentes."""
    if variables is None:
        return groups

    selected = tuple(dict.fromkeys(variables))
    unknown = sorted(set(selected) - groups.keys())
    if unknown:
        available = ", ".join(groups)
        requested = ", ".join(unknown)
        raise ValueError(
            f"Variável(is) não encontrada(s): {requested}. Disponíveis: {available}."
        )

    return {variable: groups[variable] for variable in selected}


def interpolate_to_reference_grid(
    dataset: xr.Dataset, reference_coordinates: xr.DataArray
) -> xr.Dataset:
    """Interpola um dataset nas latitudes e longitudes da grade de referência."""
    required_coordinates = {"latitude", "longitude"}
    missing_source = required_coordinates - set(dataset.coords)
    missing_reference = required_coordinates - set(reference_coordinates.coords)
    if missing_source:
        raise ValueError(
            "Dataset sem coordenada(s) para interpolação: "
            f"{', '.join(sorted(missing_source))}."
        )
    if missing_reference:
        raise ValueError(
            "Arquivo de coordenadas de referência sem: "
            f"{', '.join(sorted(missing_reference))}."
        )

    return dataset.interp(
        latitude=reference_coordinates.latitude,
        longitude=reference_coordinates.longitude,
        method="linear",
    )


def requires_interpolation(variable: str) -> bool:
    """Retorna se a variável ERA5 precisa ser levada à grade ERA5-Land."""
    return variable in ERA5_VARIABLES_TO_INTERPOLATE


def flatten_era5land_time(dataset: xr.Dataset) -> xr.Dataset:
    """Substitui as dimensões ``time`` e ``step`` pelo datetime ``valid_time``.

    Os GRIBs ERA5-Land chegam com o instante de referência (``time``) e a
    antecedência em horas (``step``). A coordenada bidimensional ``valid_time``
    já representa a soma dos dois e torna-se o único eixo temporal.
    """
    if not {"time", "step"}.issubset(dataset.dims):
        return dataset
    if "valid_time" not in dataset.coords:
        raise ValueError(
            "Dataset ERA5-Land com dimensões 'time' e 'step' sem coordenada "
            "'valid_time'."
        )

    flattened = dataset.stack(_instant=("time", "step"))
    flattened = flattened.swap_dims({"_instant": "valid_time"})
    flattened = flattened.drop_vars(["time", "step", "_instant"]).rename(
        {"valid_time": "time"}
    )
    return flattened.transpose("time", ...)


def open_monthly_dataset(path: Path) -> xr.Dataset:
    """Abre um GRIB sem deixar o índice auxiliar ``.idx`` no disco."""
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def prepare_time_axis(dataset: xr.Dataset, flatten_time_and_step: bool) -> xr.Dataset:
    """Normaliza e ordena somente o eixo temporal, sem carregar os valores."""
    if "time" not in dataset.coords and "time" not in dataset.dims:
        raise ValueError("Dataset sem coordenada ou dimensão 'time'.")
    if flatten_time_and_step:
        dataset = flatten_era5land_time(dataset)
    return dataset.sortby("time")


def build_write_plan(
    files: Iterable[Path], flatten_time_and_step: bool = False
) -> list[WriteRun]:
    """Monta um plano pequeno de escrita ordenada e sem tempos repetidos.

    Nesta etapa são lidas apenas as coordenadas temporais. Os campos
    meteorológicos permanecem no GRIB até o respectivo bloco ser escrito.
    """
    file_list = list(files)
    seen_times: set[object] = set()
    entries: list[tuple[object, int, int]] = []

    for file_index, path in enumerate(file_list):
        dataset = open_monthly_dataset(path)
        try:
            prepared = prepare_time_axis(dataset, flatten_time_and_step)
            time_index = prepared.indexes.get("time")
            if time_index is None:
                raise ValueError(f"Não foi possível construir o índice temporal de {path.name}.")

            # A ordem dos arquivos é o critério de desempate, como era no
            # concatenação anterior com ``keep='first'``.
            for position, timestamp in enumerate(time_index):
                if timestamp in seen_times:
                    continue
                seen_times.add(timestamp)
                entries.append((timestamp, file_index, position))
        finally:
            dataset.close()

    entries.sort(key=lambda entry: entry[0])
    run_paths: list[Path] = []
    run_positions: list[list[int]] = []
    for _, file_index, position in entries:
        path = file_list[file_index]
        if run_paths and run_paths[-1] == path:
            run_positions[-1].append(position)
        else:
            run_paths.append(path)
            run_positions.append([position])

    return [
        WriteRun(path=path, positions=tuple(positions))
        for path, positions in zip(run_paths, run_positions, strict=True)
    ]


def append_chunk_to_netcdf(chunk: xr.Dataset, output: Path, start: int) -> None:
    """Anexa um bloco a um NetCDF já criado, materializando só este bloco."""
    time_size = chunk.sizes["time"]
    with NetCDFDataset(output, mode="a") as target:
        for name, variable in chunk.variables.items():
            if "time" not in variable.dims:
                continue

            time_axis = variable.dims.index("time")
            target_slice = [slice(None)] * variable.ndim
            target_slice[time_axis] = slice(start, start + time_size)
            # ``values`` é intencional aqui: no máximo TIME_CHUNK_SIZE
            # instantes são lidos do GRIB antes de serem gravados. O backend
            # netCDF4 recebe datas como números CF, não como datetime64.
            values = variable.values
            target_variable = target.variables[name]
            if np.issubdtype(values.dtype, np.datetime64):
                values = date2num(
                    values.astype("datetime64[us]").astype(object),
                    units=target_variable.units,
                    calendar=getattr(target_variable, "calendar", "standard"),
                )
            target_variable[tuple(target_slice)] = values


def format_bytes(size: int) -> str:
    """Formata tamanhos para mensagens de erro acionáveis."""
    return f"{size / 1024**3:.1f} GiB"


def ensure_enough_disk_space(
    write_plan: Iterable[WriteRun], output_directory: Path, flatten_time_and_step: bool
) -> None:
    """Falha cedo se não houver espaço para o NetCDF temporário completo."""
    runs = list(write_plan)
    total_instants = sum(len(run.positions) for run in runs)
    if not runs or total_instants == 0:
        return

    dataset = open_monthly_dataset(runs[0].path)
    try:
        sample = prepare_time_axis(dataset, flatten_time_and_step)
        sample_instants = sample.sizes["time"]
        estimated_bytes = 0
        for variable in sample.variables.values():
            if "time" in variable.dims:
                estimated_bytes += math.ceil(
                    variable.nbytes * total_instants / sample_instants
                )
            else:
                estimated_bytes += variable.nbytes
    finally:
        dataset.close()

    required_bytes = math.ceil(estimated_bytes * DISK_SPACE_SAFETY_FACTOR)
    available_bytes = shutil.disk_usage(output_directory).free
    if available_bytes < required_bytes:
        raise OSError(
            "Espaço em disco insuficiente para consolidar o NetCDF: "
            f"são necessários ao menos {format_bytes(required_bytes)} livres, "
            f"mas há {format_bytes(available_bytes)}. Libere espaço antes de "
            "executar, pois o arquivo é gravado primeiro como temporário."
        )


def consolidate_variable(
    variable: str,
    files: list[Path],
    output: Path,
    reference_coordinates: xr.DataArray | None = None,
    flatten_time_and_step: bool = False,
) -> None:
    """Grava uma variável sem materializar a série completa na memória."""
    temporary: Path | None = None
    written_instants = 0
    has_written_data = False

    try:
        write_plan = build_write_plan(files, flatten_time_and_step)
        ensure_enough_disk_space(
            write_plan, output.parent, flatten_time_and_step
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{variable}.",
            suffix=".tmp.nc",
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)

        for run in write_plan:
            dataset = open_monthly_dataset(run.path)
            try:
                prepared = prepare_time_axis(dataset, flatten_time_and_step)
                for start in range(0, len(run.positions), TIME_CHUNK_SIZE):
                    positions = run.positions[start : start + TIME_CHUNK_SIZE]
                    chunk = prepared.isel(time=list(positions))
                    try:
                        # Selecionar antes de interpolar é essencial: assim a
                        # interpolação nunca materializa um mês inteiro.
                        if reference_coordinates is not None:
                            chunk = interpolate_to_reference_grid(
                                chunk, reference_coordinates
                            )

                        if has_written_data:
                            append_chunk_to_netcdf(chunk, temporary, written_instants)
                        else:
                            chunk.to_netcdf(
                                temporary,
                                mode="w",
                                engine="netcdf4",
                                unlimited_dims=["time"],
                            )
                            has_written_data = True
                        written_instants += len(positions)
                    finally:
                        # ``chunk`` é uma visão/expressão do arquivo ainda
                        # aberto; fechá-lo poderia fechar a fonte antes do
                        # próximo bloco. Remover a referência libera os
                        # valores materializados deste bloco.
                        del chunk
            finally:
                dataset.close()

        if not has_written_data:
            raise ValueError(f"Nenhum instante temporal encontrado para {variable}.")
        temporary.replace(output)
        print(f"  {written_instants} instante(s) único(s).")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def consolidate(
    directory: Path,
    output_directory: Path,
    variables: Iterable[str] | None = None,
    reference_coordinates_path: Path = DEFAULT_REFERENCE_COORDINATES,
    overwrite: bool = False,
) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Diretório inexistente: {directory}")

    groups = select_variables(monthly_files_by_variable(directory), variables)
    if not groups:
        print(f"Nenhum arquivo mensal .grib encontrado em {directory}.")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    variables_to_interpolate = {
        variable for variable in groups if requires_interpolation(variable)
    }
    variables_to_flatten_time = set(groups) & set(ERA5_LAND_VARIABLES_TO_FLATTEN_TIME)
    reference_coordinates: xr.DataArray | None = None

    try:
        if variables_to_interpolate:
            if not reference_coordinates_path.is_file():
                raise FileNotFoundError(
                    "Arquivo de coordenadas de referência inexistente: "
                    f"{reference_coordinates_path}"
                )
            reference_coordinates = xr.open_dataarray(reference_coordinates_path)

        for variable, files in groups.items():
            output = output_directory / f"{variable}.nc"
            if output.exists() and not overwrite:
                print(
                    f"Ignorado {output.name}: já existe "
                    "(defina OVERWRITE_EXISTING_OUTPUTS como True para recriar)."
                )
                continue

            is_interpolated = variable in variables_to_interpolate
            flattens_time = variable in variables_to_flatten_time
            actions = []
            if is_interpolated:
                actions.append("interpolando para ERA5-Land")
            if flattens_time:
                actions.append("unificando time + step")
            action = f" ({'; '.join(actions)})" if actions else ""
            print(
                f"Consolidando {variable}: {len(files)} arquivo(s) -> {output.name}{action}"
            )
            try:
                consolidate_variable(
                    variable,
                    files,
                    output,
                    reference_coordinates if is_interpolated else None,
                    flatten_time_and_step=flattens_time,
                )
            finally:
                # Ao sair de consolidate_variable não há referências aos arrays da
                # variável anterior; isto evita acumular memória entre os grupos.
                gc.collect()
    finally:
        if reference_coordinates is not None:
            reference_coordinates.close()


if __name__ == "__main__":
    consolidate(
        DEFAULT_DIRECTORY,
        output_directory=DEFAULT_OUTPUT_DIRECTORY,
        variables=VARIABLES_TO_CONSOLIDATE,
        reference_coordinates_path=DEFAULT_REFERENCE_COORDINATES,
        overwrite=OVERWRITE_EXISTING_OUTPUTS,
    )
