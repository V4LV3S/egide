import numpy as np
import xarray as xr

import scripts.consolidar_meteoro_netcdf as consolidator
from scripts.consolidar_meteoro_netcdf import (
    consolidate_variable,
    flatten_era5land_time,
    interpolate_to_reference_grid,
    monthly_files_by_variable,
    requires_interpolation,
    select_variables,
)
from scripts.download_era5 import END_DATE, START_DATE, iter_months


def test_interpolate_to_reference_grid_uses_latitude_and_longitude() -> None:
    dataset = xr.Dataset(
        {"u100": (("latitude", "longitude"), [[0.0, 2.0], [2.0, 4.0]])},
        coords={"latitude": [0.0, 2.0], "longitude": [0.0, 2.0]},
    )
    reference = xr.DataArray(
        np.zeros((1, 1)),
        coords={"latitude": [1.0], "longitude": [1.0]},
        dims=("latitude", "longitude"),
    )

    interpolated = interpolate_to_reference_grid(dataset, reference)

    assert interpolated.latitude.values.tolist() == [1.0]
    assert interpolated.longitude.values.tolist() == [1.0]
    assert interpolated.u100.item() == 2.0


def test_select_variables_preserves_requested_order() -> None:
    groups = {"100u": [], "100v": [], "tcc": []}

    selected = select_variables(groups, ["tcc", "100u", "tcc"])

    assert list(selected) == ["tcc", "100u"]


def test_only_era5_variables_require_interpolation() -> None:
    assert requires_interpolation("100u")
    assert requires_interpolation("100v")
    assert requires_interpolation("tcc")
    assert not requires_interpolation("2t")
    assert not requires_interpolation("tp")


def test_flatten_era5land_time_uses_valid_time_as_single_time_axis() -> None:
    dataset = xr.Dataset(
        {
            "t2m": (
                ("time", "step", "latitude", "longitude"),
                np.arange(4).reshape(2, 2, 1, 1),
            )
        },
        coords={
            "time": np.array(["2023-10-01", "2023-10-02"], dtype="datetime64[ns]"),
            "step": np.array([1, 2], dtype="timedelta64[h]"),
            "latitude": [0.0],
            "longitude": [0.0],
            "valid_time": (
                ("time", "step"),
                np.array(
                    [
                        ["2023-10-01T01", "2023-10-01T02"],
                        ["2023-10-02T01", "2023-10-02T02"],
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
        },
    )

    flattened = flatten_era5land_time(dataset)

    assert flattened.sizes == {"time": 4, "latitude": 1, "longitude": 1}
    np.testing.assert_array_equal(
        flattened.time.values,
        np.array(
            [
                "2023-10-01T01",
                "2023-10-01T02",
                "2023-10-02T01",
                "2023-10-02T02",
            ],
            dtype="datetime64[ns]",
        ),
    )
    assert flattened.t2m[:, 0, 0].values.tolist() == [0, 1, 2, 3]


def test_monthly_files_by_variable_does_not_limit_the_period(tmp_path) -> None:
    for filename in (
        "100u_2025_09.grib",
        "100u_2025_10.grib",
        "100u_2026_08.grib",
        "100u_2026_09.grib",
    ):
        (tmp_path / filename).touch()

    groups = monthly_files_by_variable(tmp_path)

    assert [path.name for path in groups["100u"]] == [
        "100u_2025_09.grib",
        "100u_2025_10.grib",
        "100u_2026_08.grib",
        "100u_2026_09.grib",
    ]


def test_consolidate_variable_writes_sorted_unique_times_in_small_chunks(
    tmp_path, monkeypatch
) -> None:
    files = [tmp_path / "tp_2024_01.grib", tmp_path / "tp_2024_02.grib"]
    datasets = {
        files[0]: xr.Dataset(
            {"tp": (("time", "latitude", "longitude"), [[[2.0]], [[1.0]]])},
            coords={
                "time": np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[ns]"),
                "latitude": [0.0],
                "longitude": [0.0],
            },
        ),
        files[1]: xr.Dataset(
            {"tp": (("time", "latitude", "longitude"), [[[99.0]], [[3.0]]])},
            coords={
                "time": np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
                "latitude": [0.0],
                "longitude": [0.0],
            },
        ),
    }

    monkeypatch.setattr(
        consolidator, "open_monthly_dataset", lambda path: datasets[path].copy(deep=True)
    )
    monkeypatch.setattr(consolidator, "TIME_CHUNK_SIZE", 1)
    output = tmp_path / "tp.nc"

    consolidate_variable("tp", files, output)

    with xr.open_dataset(output) as result:
        np.testing.assert_array_equal(
            result.time.values,
            np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
        )
        assert result.tp[:, 0, 0].values.tolist() == [1.0, 2.0, 3.0]


def test_download_period_is_october_2023_to_august_2026() -> None:
    months = list(iter_months(START_DATE, END_DATE))

    assert months[0].strftime("%Y-%m") == "2023-10"
    assert months[-1].strftime("%Y-%m") == "2026-08"
    assert len(months) == 35
