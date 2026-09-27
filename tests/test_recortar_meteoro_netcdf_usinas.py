from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from scripts.recortar_meteoro_netcdf_usinas import (
    crop_to_square,
    load_plant_centres,
    plant_directory_name,
)


def test_crop_to_square_handles_descending_latitude() -> None:
    dataset = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), [[[1.0] * 5] * 5])},
        coords={
            "time": [pd.Timestamp("2025-01-01")],
            "latitude": [1.0, 0.5, 0.0, -0.5, -1.0],
            "longitude": [-2.0, -1.5, -1.0, -0.5, 0.0],
        },
    )

    cropped = crop_to_square(
        dataset,
        centre_latitude=0.1,
        centre_longitude=-0.9,
    )

    assert cropped.latitude.values.tolist() == [0.5, 0.0]
    assert cropped.longitude.values.tolist() == [-1.0, -0.5]
    assert cropped.t2m.shape == (1, 2, 2)


def test_load_plant_centres_selects_requested_order(tmp_path: Path) -> None:
    path = tmp_path / "plants.parquet"
    pd.DataFrame(
        {
            "conjunto": ["Assu Sol", "Futura"],
            "latitude": [-5.5, -9.6],
            "longitude": [-37.0, -40.6],
        }
    ).to_parquet(path, index=False)

    selected = load_plant_centres(path, ("Futura", "Assu Sol"))

    assert selected["conjunto"].tolist() == ["Futura", "Assu Sol"]


def test_load_plant_centres_rejects_unknown_name(tmp_path: Path) -> None:
    path = tmp_path / "plants.parquet"
    pd.DataFrame(
        {
            "conjunto": ["Assu Sol"],
            "latitude": [-5.5],
            "longitude": [-37.0],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="não encontrado"):
        load_plant_centres(path, ("Inexistente",))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Assu Sol", "conj_assu_sol"),
        ("Futura", "conj_futura"),
        ("Sol do Sertão", "conj_sol_do_sertão"),
    ],
)
def test_plant_directory_name_matches_individual_datasets(
    name: str, expected: str
) -> None:
    assert plant_directory_name(name) == expected
