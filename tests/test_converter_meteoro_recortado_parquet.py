from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import xarray as xr

import ml.scripts.converter_meteoro_recortado_parquet as converter
from ml.scripts.converter_meteoro_recortado_parquet import (
    convert_all_regions,
    netcdf_to_dataframe,
)


def write_netcdf(path: Path) -> xr.Dataset:
    """Cria um NetCDF mínimo para testar a conversão tabular direta."""
    dataset = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), [[[300.0, 301.0]]]),
        },
        coords={
            "time": pd.to_datetime(["2025-01-01T00:00:00"]),
            "latitude": [-10.0],
            "longitude": [-40.0, -39.5],
        },
    )
    dataset.to_netcdf(path)
    return dataset


def test_netcdf_to_dataframe_preserves_coordinates_and_values(tmp_path) -> None:
    source = tmp_path / "t2m.nc"
    dataset = write_netcdf(source)

    result = netcdf_to_dataframe(source)

    with xr.open_dataset(source) as persisted:
        expected = persisted.to_dataframe().reset_index()
    pdt.assert_frame_equal(result, expected)


def test_convert_all_regions_writes_raw_parquet_per_region(tmp_path, monkeypatch) -> None:
    input_directory = tmp_path / "meteoro-recortado"
    region = input_directory / "CE"
    region.mkdir(parents=True)
    write_netcdf(region / "t2m.nc")
    output_directory = tmp_path / "ml-data"
    monkeypatch.setattr(converter, "REGIONS_TO_CONVERT", None)
    monkeypatch.setattr(converter, "OVERWRITE_EXISTING_OUTPUTS", False)

    outputs = convert_all_regions(input_directory, output_directory)

    destination = output_directory / "CE" / "t2m.parquet"
    assert outputs == [destination]
    assert list(pd.read_parquet(destination).columns) == [
        "time", "latitude", "longitude", "t2m"
    ]


def test_existing_output_is_not_overwritten_by_default(tmp_path, monkeypatch) -> None:
    input_directory = tmp_path / "meteoro-recortado"
    region = input_directory / "CE"
    region.mkdir(parents=True)
    write_netcdf(region / "t2m.nc")
    output_directory = tmp_path / "ml-data"
    destination = output_directory / "CE" / "t2m.parquet"
    destination.parent.mkdir(parents=True)
    pd.DataFrame({"sentinel": [1]}).to_parquet(destination, index=False)
    monkeypatch.setattr(converter, "REGIONS_TO_CONVERT", None)
    monkeypatch.setattr(converter, "OVERWRITE_EXISTING_OUTPUTS", False)

    outputs = convert_all_regions(input_directory, output_directory)

    assert outputs == []
    assert list(pd.read_parquet(destination).columns) == ["sentinel"]
