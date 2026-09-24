"""Trata e recorta NetCDFs meteorológicos pelos shapefiles dos estados.

Antes do recorte espacial, ``tp`` deixa de ser acumulado diariamente e passa
de metros para milimetros por intervalo. ``ssr``/``str`` tambem deixam de ser
acumulados diarios e se tornam fluxos medios por intervalo em W m-2. Em
seguida, Salem reduz a grade ao retangulo de cada shapefile e mascara as
celulas fora do poligono. Execute pela IDE, sem argumentos de terminal.
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
SHAPES_DIRECTORY = ROOT / "data" / "processed" / "shapes-estados"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"

# Use ``None`` para todos os shapefiles disponíveis ou informe nomes sem a
# extensão, por exemplo: ("BA_SE", "PE").
SHAPES_TO_PROCESS: tuple[str, ...] | None = ["BA_SE", "AL_PE","PB_RN", "CE", 'MA', "PI"]

# Processa poucos instantes por vez para limitar o uso de memória.
TIME_CHUNK_SIZE = 24 * 31 * 48

# Inclui também células da grade tocadas pela borda do estado. Isto evita que
# estados pequenos desapareçam após um coarsening mais agressivo.
ALL_TOUCHED = True
OVERWRITE_EXISTING_OUTPUTS = True

# Transformacoes aplicadas antes do recorte espacial. Desative somente se for
# necessario reproduzir os valores brutos dos NetCDFs de entrada.
CONVERT_PRECIPITATION_TO_INTERVAL_MILLIMETRES = True
CONVERT_RADIATION_TO_WATTS_PER_SQUARE_METRE = True
RADIATION_VARIABLES = ("ssr", "str")
# O eixo auxiliar aloca 00:00 no ciclo que terminou nesse instante. Assim,
# 23:00 e o 00:00 seguinte pertencem ao mesmo ciclo; 01:00 inicia o proximo.
ACCUMULATION_CYCLE_OFFSET = np.timedelta64(1, "m")


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


def remove_daily_accumulation(data_array: xr.DataArray) -> xr.DataArray:
    """Retorna a grandeza acumulada somente no intervalo de cada ``time``.

    A classificacao do ciclo usa ``time - 1 minuto``: portanto, o acumulado
    rotulado 00:00 fecha o dia anterior e e diferenciado contra 23:00; 01:00
    pertence ao novo ciclo e usa o proprio acumulado.
    """
    if "time" not in data_array.dims:
        raise ValueError(f"{data_array.name} nao possui a dimensao 'time'.")

    time = data_array.time
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError(f"{data_array.name} possui coordenada 'time' nao suportada.")

    auxiliary_cycle = (time - ACCUMULATION_CYCLE_OFFSET).dt.floor("D")
    same_cycle_as_previous = auxiliary_cycle == auxiliary_cycle.shift(time=1)
    is_cycle_start = time.dt.hour == 1
    interval = data_array - data_array.shift(time=1)
    return xr.where(
        is_cycle_start,
        data_array,
        xr.where(same_cycle_as_previous, interval, np.nan),
    )


def convert_accumulated_precipitation_to_interval_millimetres(
    data_array: xr.DataArray,
) -> xr.DataArray:
    """Remove acumulacao de ``tp`` e converte m para mm por intervalo.

    O ponto de 00:00 encerra o ciclo diario anterior e e calculado pela
    diferenca em relacao a 23:00. O ponto de 01:00 inicia o novo ciclo, logo o
    proprio acumulado representa o primeiro intervalo.
    """
    # Precipitacao por intervalo nao pode ser negativa. ``clip`` tambem evita
    # residuos negativos minimos introduzidos por ponto flutuante.
    converted = (remove_daily_accumulation(data_array) * 1000.0).clip(min=0.0)
    converted.attrs = dict(data_array.attrs)
    converted.attrs["units"] = "mm"
    converted.attrs["GRIB_units"] = "mm"
    converted.attrs["long_name"] = (
        f"{data_array.attrs.get('long_name', data_array.name)}: accumulation per interval"
    )
    converted.attrs["comment"] = (
        "Converted from ERA5-Land accumulated values to millimetres per interval. "
        "00:00 closes the previous cycle and 01:00 starts the next one."
    )
    return converted


def convert_accumulated_radiation_to_flux(data_array: xr.DataArray) -> xr.DataArray:
    """Converte acumulados diarios de radiacao para fluxo medio em W m-2.

    Nos dados horarios do ERA5-Land, o ciclo de acumulacao reinicia as 01:00
    UTC. Nesse instante, o valor acumulado representa o primeiro intervalo; nos
    demais, usa-se a diferenca para o instante anterior. Lacunas e o primeiro
    instante sem acumulado anterior permanecem como valores ausentes.
    """
    time = data_array.time
    if not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError(f"{data_array.name} possui coordenada 'time' nao suportada.")

    auxiliary_cycle = (time - ACCUMULATION_CYCLE_OFFSET).dt.floor("D")
    same_cycle_as_previous = auxiliary_cycle == auxiliary_cycle.shift(time=1)
    is_cycle_start = time.dt.hour == 1
    interval_energy = remove_daily_accumulation(data_array)
    elapsed_seconds = (time - time.shift(time=1)) / np.timedelta64(1, "s")
    seconds_since_midnight = (
        time.dt.hour * 3600 + time.dt.minute * 60 + time.dt.second
    )
    interval_seconds = xr.where(
        is_cycle_start,
        seconds_since_midnight,
        xr.where(same_cycle_as_previous, elapsed_seconds, np.nan),
    )
    interval_seconds = interval_seconds.where(interval_seconds > 0)

    converted = interval_energy / interval_seconds
    # SSR e energia solar liquida descendente, portanto nao deve ser negativa.
    # STR e radiacao termica liquida e tem sinal fisico; ela nao e truncada.
    if data_array.name == "ssr":
        converted = converted.clip(min=0.0)
    converted.attrs = dict(data_array.attrs)
    converted.attrs["units"] = "W m-2"
    converted.attrs["GRIB_units"] = "W m**-2"
    converted.attrs["GRIB_stepType"] = "avg"
    converted.attrs["long_name"] = (
        f"{data_array.attrs.get('long_name', data_array.name)}: mean over interval"
    )
    converted.attrs["comment"] = (
        "Converted from ERA5-Land daily accumulated values; each value "
        "represents the interval ending at its time coordinate."
    )
    return converted


def prepare_data_before_crop(dataset: xr.Dataset) -> xr.Dataset:
    """Aplica conversoes fisicas aos dados antes do recorte espacial."""
    prepared = dataset.copy(deep=False)
    if CONVERT_PRECIPITATION_TO_INTERVAL_MILLIMETRES and "tp" in prepared.data_vars:
        prepared["tp"] = convert_accumulated_precipitation_to_interval_millimetres(
            prepared["tp"]
        )

    if CONVERT_RADIATION_TO_WATTS_PER_SQUARE_METRE:
        for variable_name in RADIATION_VARIABLES:
            if variable_name in prepared.data_vars:
                prepared[variable_name] = convert_accumulated_radiation_to_flux(
                    prepared[variable_name]
                )
    return prepared


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

        # A conversao vem antes de ``subset``/``roi`` para que a etapa fisica
        # seja independente da regiao solicitada e preceda o recorte espacial.
        prepared = prepare_data_before_crop(source)
        # ``subset`` reduz a área processada; ``roi`` aplica a máscara precisa
        # do polígono e preserva a caixa delimitadora resultante.
        subset = prepared.salem.subset(shape=str(shape_path))
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
