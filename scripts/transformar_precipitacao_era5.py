"""Converte ``tp`` acumulado do ERA5-Land para milimetros por intervalo.

Os arquivos de entrada sao preservados. O script cria uma copia de cada
``tp.nc`` em uma nova arvore de diretorios, mantendo a estrutura das regioes.
Ele remove a acumulacao horaria de precipitacao e converte cada intervalo de
metros para milimetros. O ponto 00:00 fecha o ciclo anterior; 01:00 inicia o
novo ciclo diario de acumulacao.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CONFIGURACAO EDITAVEL
# ---------------------------------------------------------------------------
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado-mm"

# ``None`` processa todas as regioes. Exemplo para limitar: ("BA_SE", "CE").
REGIONS_TO_TRANSFORM: tuple[str, ...] | None = None

# Um metro de altura de agua equivale a mil milimetros. O eixo auxiliar de
# ``time - 1 minuto`` coloca 00:00 no ciclo diario que acabou de terminar.
METRES_TO_MILLIMETRES = 1000.0
ACCUMULATION_CYCLE_OFFSET = timedelta(minutes=1)
TIME_CHUNK_SIZE = 24
OVERWRITE_EXISTING_OUTPUTS = False


def interval_precipitation_m_to_mm(
    accumulated: np.ndarray,
    timestamps: list[object],
    previous_accumulated: np.ndarray | None = None,
    previous_timestamp: object | None = None,
) -> np.ndarray:
    """Separa ``tp`` usando o ciclo auxiliar e converte m para mm.

    O ponto 00:00 e diferenciado contra 23:00 por pertencer ao ciclo anterior;
    01:00 usa o proprio acumulado como primeiro intervalo do novo ciclo.
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
            elapsed_seconds = (timestamp - last_timestamp).total_seconds()
            if elapsed_seconds > 0:
                result[index] = current - last_accumulated
        last_accumulated = current
        last_timestamp = timestamp
    return np.maximum(result * METRES_TO_MILLIMETRES, 0.0)


def read_datetimes(source: NetCDFDataset) -> list[object]:
    """Le a coordenada temporal como datetime/cftime datetime."""
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


def validate_source_unit(precipitation: object) -> None:
    """Evita converter novamente um arquivo que ja esteja em milimetros."""
    if getattr(precipitation, "units", None) != "m":
        raise ValueError(
            "tp deve estar em metros (units='m') para esta conversao; "
            "o arquivo pode ja ter sido tratado pelo recorte."
        )


def create_variable_like(target: NetCDFDataset, source: object) -> object:
    """Cria uma variavel com tipo, dimensoes e atributos da origem."""
    fill_value = source.getncattr("_FillValue") if "_FillValue" in source.ncattrs() else None
    copied = target.createVariable(
        source.name, source.datatype, source.dimensions, fill_value=fill_value
    )
    for name in source.ncattrs():
        if name != "_FillValue":
            copied.setncattr(name, source.getncattr(name))
    return copied


def transform_file(input_path: Path, output_path: Path) -> None:
    """Grava ``tp`` em milimetros por intervalo, sem acumulacao horaria."""
    temporary_path = output_path.with_suffix(".tmp.nc")
    if temporary_path.exists():
        temporary_path.unlink()

    with NetCDFDataset(input_path) as source, NetCDFDataset(
        temporary_path, mode="w", format="NETCDF4"
    ) as target:
        if "tp" not in source.variables:
            raise ValueError(f"{input_path.name} nao contem a variavel 'tp'.")
        precipitation = source.variables["tp"]
        if not precipitation.dimensions or precipitation.dimensions[0] != "time":
            raise ValueError("tp precisa ter 'time' como primeira dimensao.")
        validate_source_unit(precipitation)

        for name, dimension in source.dimensions.items():
            target.createDimension(name, None if dimension.isunlimited() else len(dimension))
        for name in source.ncattrs():
            target.setncattr(name, source.getncattr(name))

        target_precipitation = None
        for name, source_variable in source.variables.items():
            target_variable = create_variable_like(target, source_variable)
            if name == "tp":
                target_precipitation = target_variable
            else:
                target_variable[...] = source_variable[...]
        if target_precipitation is None:
            raise AssertionError("A variavel tp deveria ter sido criada.")

        target_precipitation.setncattr("units", "mm")
        target_precipitation.setncattr("GRIB_units", "mm")
        target_precipitation.setncattr(
            "long_name",
            f"{getattr(precipitation, 'long_name', 'Total precipitation')}: accumulation per interval",
        )
        target_precipitation.setncattr(
            "comment",
            "Converted from ERA5-Land accumulated values to millimetres per interval. "
            "00:00 closes the previous cycle and 01:00 starts the next one.",
        )

        time_size = len(source.dimensions["time"])
        timestamps = read_datetimes(source)
        previous_accumulated: np.ndarray | None = None
        previous_timestamp: object | None = None
        for start in range(0, time_size, TIME_CHUNK_SIZE):
            stop = min(start + TIME_CHUNK_SIZE, time_size)
            values = np.ma.filled(precipitation[start:stop, ...], np.nan)
            converted = interval_precipitation_m_to_mm(
                values,
                timestamps[start:stop],
                previous_accumulated,
                previous_timestamp,
            )
            target_precipitation[start:stop, ...] = converted.astype(precipitation.dtype)
            previous_accumulated = values[-1].copy()
            previous_timestamp = timestamps[stop - 1]

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
    """Executa a conversao configurada para todas as regioes selecionadas."""
    if not INPUT_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Diretorio de entrada inexistente: {INPUT_DIRECTORY}")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    converted_files = 0
    for region in selected_regions():
        relative_region = region.relative_to(INPUT_DIRECTORY)
        input_path = region / "tp.nc"
        if not input_path.is_file():
            print(f"Ignorado {relative_region}/tp.nc: inexistente.")
            continue

        output_path = OUTPUT_DIRECTORY / relative_region / "tp.nc"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not OVERWRITE_EXISTING_OUTPUTS:
            print(f"Ignorado {output_path}: arquivo de saida ja existe.")
            continue

        print(f"Convertendo {relative_region}/tp.nc")
        transform_file(input_path, output_path)
        converted_files += 1

    print(f"Concluido: {converted_files} arquivo(s) em {OUTPUT_DIRECTORY}.")


if __name__ == "__main__":
    main()
