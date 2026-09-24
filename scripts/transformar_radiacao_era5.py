"""Converte os acumulados ``ssr`` e ``str`` recortados do ERA5-Land.

Os arquivos de origem sao preservados. Para cada arquivo de radiacao em
``meteoro-recortado``, o script grava uma copia transformada na mesma estrutura
de regioes em outro diretorio.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CONFIGURACAO EDITAVEL
# ---------------------------------------------------------------------------
# Diretorio com uma pasta por regiao, por exemplo BA_SE/ssr.nc.
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"

# Nova arvore de dados. Nenhum arquivo do diretorio de entrada e alterado.
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado-intervalo"

# ``None`` processa todas as regioes disponiveis. Para limitar, use por
# exemplo: ("BA_SE", "CE"). Os nomes sao relativos a INPUT_DIRECTORY.
REGIONS_TO_TRANSFORM: tuple[str, ...] | None = None

# ``ssr`` e ``str`` sao acumulados desde 00:00 UTC no ERA5-Land.
VARIABLES_TO_TRANSFORM = ("ssr", "str")

# True produz fluxo medio em W m-2. False mantem energia por intervalo em J m-2.
CONVERT_TO_WATTS_PER_SQUARE_METRE = True

# O eixo auxiliar de ``time - 1 minuto`` deixa 00:00 no ciclo anterior e faz
# 01:00 ser o primeiro ponto do novo ciclo diario.
ACCUMULATION_CYCLE_OFFSET = timedelta(minutes=1)

# Blocos de 24 instantes evitam carregar uma serie espacial inteira na memoria.
TIME_CHUNK_SIZE = 24 * 31 * 24
OVERWRITE_EXISTING_OUTPUTS = False


def seconds_since_midnight(timestamp: object) -> float:
    """Retorna os segundos entre 00:00 e um datetime ou cftime datetime."""
    return float(
        timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    )


def seconds_between(current: object, previous: object) -> float:
    """Retorna a duracao em segundos entre dois tempos do NetCDF."""
    return float((current - previous).total_seconds())


def interval_energy(
    accumulated: np.ndarray,
    timestamps: Sequence[object],
    previous_accumulated: np.ndarray | None = None,
    previous_timestamp: object | None = None,
) -> np.ndarray:
    """Calcula a energia de cada intervalo a partir do acumulado diario.

    A acumulacao reinicia as 01:00 UTC. O ciclo e identificado pelo eixo
    auxiliar ``time - 1 minuto``: 00:00 e diferenciado contra 23:00, enquanto
    01:00 usa o proprio acumulado (intervalo 00:00--01:00). Nos demais,
    subtrai-se o valor anterior dentro do mesmo ciclo.
    Se a serie iniciar no meio de um ciclo, o primeiro valor vira ``NaN``, pois
    o acumulado anterior necessario para a diferenca nao esta disponivel.
    """
    if accumulated.shape[0] != len(timestamps):
        raise ValueError("O primeiro eixo de accumulated deve corresponder a timestamps.")

    result = np.full(accumulated.shape, np.nan, dtype=np.float64)
    last_accumulated = previous_accumulated
    last_timestamp = previous_timestamp

    for index, timestamp in enumerate(timestamps):
        current = accumulated[index]
        cycle = timestamp - ACCUMULATION_CYCLE_OFFSET
        previous_cycle = (
            last_timestamp - ACCUMULATION_CYCLE_OFFSET
            if last_timestamp is not None
            else None
        )
        same_cycle = (
            previous_cycle is not None
            and (cycle.year, cycle.month, cycle.day)
            == (previous_cycle.year, previous_cycle.month, previous_cycle.day)
        )
        if timestamp.hour == 1:
            result[index] = current
        elif same_cycle and last_accumulated is not None and last_timestamp is not None:
            elapsed = seconds_between(timestamp, last_timestamp)
            if elapsed > 0:
                result[index] = current - last_accumulated

        last_accumulated = current
        last_timestamp = timestamp

    return result


def interval_duration_seconds(
    timestamps: Sequence[object], previous_timestamp: object | None = None
) -> np.ndarray:
    """Calcula a duracao de cada intervalo, inclusive se houver lacunas."""
    durations = np.full(len(timestamps), np.nan, dtype=np.float64)
    last_timestamp = previous_timestamp

    for index, timestamp in enumerate(timestamps):
        cycle = timestamp - ACCUMULATION_CYCLE_OFFSET
        previous_cycle = (
            last_timestamp - ACCUMULATION_CYCLE_OFFSET
            if last_timestamp is not None
            else None
        )
        same_cycle = (
            previous_cycle is not None
            and (cycle.year, cycle.month, cycle.day)
            == (previous_cycle.year, previous_cycle.month, previous_cycle.day)
        )
        if timestamp.hour == 1:
            durations[index] = seconds_since_midnight(timestamp)
        elif same_cycle and last_timestamp is not None:
            elapsed = seconds_between(timestamp, last_timestamp)
            if elapsed > 0:
                durations[index] = elapsed
        last_timestamp = timestamp

    return durations


def create_variable_like(target: NetCDFDataset, source: object) -> object:
    """Cria uma variavel com o tipo, dimensoes e atributos da origem."""
    fill_value = source.getncattr("_FillValue") if "_FillValue" in source.ncattrs() else None
    copied = target.createVariable(
        source.name, source.datatype, source.dimensions, fill_value=fill_value
    )
    for name in source.ncattrs():
        if name != "_FillValue":
            copied.setncattr(name, source.getncattr(name))
    return copied


def transformed_attributes(source: object) -> dict[str, object]:
    """Atualiza unidade e metadados para a grandeza por intervalo."""
    attributes = {
        name: source.getncattr(name)
        for name in source.ncattrs()
        if name != "_FillValue"
    }
    source_name = attributes.get("long_name", source.name)
    if CONVERT_TO_WATTS_PER_SQUARE_METRE:
        attributes["units"] = "W m-2"
        attributes["long_name"] = f"{source_name}: mean over interval"
    else:
        attributes["units"] = "J m-2"
        attributes["long_name"] = f"{source_name}: energy per interval"
    attributes["comment"] = (
        "Converted from ERA5-Land daily accumulated values; each value "
        "represents the interval ending at its time coordinate."
    )
    return attributes


def read_datetimes(source: NetCDFDataset) -> list[object]:
    """Le o eixo temporal (pequeno) como datetime/cftime datetime."""
    if "time" not in source.variables:
        raise ValueError("NetCDF sem a coordenada temporal 'time'.")
    time = source.variables["time"]
    return list(
        num2date(
            time[:],
            units=time.units,
            calendar=getattr(time, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
    )


def transform_file(input_path: Path, output_path: Path, variable_name: str) -> None:
    """Grava a copia transformada de um NetCDF sem materializar toda a serie."""
    temporary_path = output_path.with_suffix(".tmp.nc")
    if temporary_path.exists():
        temporary_path.unlink()

    with NetCDFDataset(input_path) as source, NetCDFDataset(
        temporary_path, mode="w", format="NETCDF4"
    ) as target:
        if variable_name not in source.variables:
            raise ValueError(
                f"{input_path.name} nao contem a variavel {variable_name!r}."
            )
        radiation = source.variables[variable_name]
        if not radiation.dimensions or radiation.dimensions[0] != "time":
            raise ValueError(
                f"{variable_name} precisa ter 'time' como primeira dimensao."
            )

        for name, dimension in source.dimensions.items():
            target.createDimension(name, None if dimension.isunlimited() else len(dimension))
        for name in source.ncattrs():
            target.setncattr(name, source.getncattr(name))

        target_radiation = None
        for name, source_variable in source.variables.items():
            target_variable = create_variable_like(target, source_variable)
            if name == variable_name:
                target_radiation = target_variable
            else:
                target_variable[...] = source_variable[...]
        if target_radiation is None:
            raise AssertionError("A variavel de radiacao deveria ter sido criada.")
        for name, value in transformed_attributes(radiation).items():
            target_radiation.setncattr(name, value)

        timestamps = read_datetimes(source)
        previous_accumulated: np.ndarray | None = None
        previous_timestamp: object | None = None
        for start in range(0, len(timestamps), TIME_CHUNK_SIZE):
            stop = min(start + TIME_CHUNK_SIZE, len(timestamps))
            block_timestamps = timestamps[start:stop]
            # netCDF4 devolve valores de preenchimento como MaskedArray. Eles
            # precisam virar NaN antes da subtracao: caso contrario, o valor
            # subjacente (frequentemente zero) faria um acumulado incompleto
            # parecer uma observacao valida.
            accumulated = np.ma.filled(radiation[start:stop, ...], np.nan)
            transformed = interval_energy(
                accumulated,
                block_timestamps,
                previous_accumulated,
                previous_timestamp,
            )
            if CONVERT_TO_WATTS_PER_SQUARE_METRE:
                durations = interval_duration_seconds(
                    block_timestamps, previous_timestamp
                )
                transformed /= durations.reshape(
                    (-1,) + (1,) * (transformed.ndim - 1)
                )

            # SSR e radiacao solar liquida descendente e nao deve ser negativa.
            # STR e radiacao termica liquida: seu sinal fisico e preservado.
            if variable_name == "ssr":
                transformed = np.maximum(transformed, 0.0)

            target_radiation[start:stop, ...] = transformed.astype(radiation.dtype)
            previous_accumulated = accumulated[-1].copy()
            previous_timestamp = block_timestamps[-1]

    temporary_path.replace(output_path)


def selected_regions() -> list[Path]:
    """Retorna as pastas de regiao solicitadas, validando a configuracao."""
    if REGIONS_TO_TRANSFORM is None:
        return sorted(path for path in INPUT_DIRECTORY.iterdir() if path.is_dir())

    regions = [INPUT_DIRECTORY / name for name in REGIONS_TO_TRANSFORM]
    missing = [str(path) for path in regions if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Regiao(oes) inexistente(s): {', '.join(missing)}")
    return regions


def main() -> None:
    """Executa a transformacao para todas as regioes configuradas."""
    if not INPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Diretorio de entrada inexistente: {INPUT_DIRECTORY}")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    transformed_files = 0
    for region in selected_regions():
        relative_region = region.relative_to(INPUT_DIRECTORY)
        for variable_name in VARIABLES_TO_TRANSFORM:
            input_path = region / f"{variable_name}.nc"
            if not input_path.is_file():
                print(f"Ignorado {relative_region}/{variable_name}.nc: inexistente.")
                continue

            output_path = OUTPUT_DIRECTORY / relative_region / input_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
                print(f"Ignorado {output_path}: arquivo de saida ja existe.")
                continue

            print(f"Transformando {relative_region}/{variable_name}.nc")
            transform_file(input_path, output_path, variable_name)
            transformed_files += 1

    print(f"Concluido: {transformed_files} arquivo(s) em {OUTPUT_DIRECTORY}.")


if __name__ == "__main__":
    main()
