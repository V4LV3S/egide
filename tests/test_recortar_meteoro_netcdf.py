import numpy as np
import xarray as xr

from scripts.recortar_meteoro_netcdf import prepare_data_before_crop


def test_prepare_data_before_crop_converts_precipitation_and_radiation() -> None:
    time = np.array(
        ["2024-01-01T23", "2024-01-02T00", "2024-01-02T01", "2024-01-02T02"],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {
            "tp": (
                ("time", "latitude", "longitude"),
                np.array([0.100, 0.120, 0.010, 0.025]).reshape(4, 1, 1),
            ),
            "ssr": (
                ("time", "latitude", "longitude"),
                np.array([100.0, 120.0, 10.0, 25.0]).reshape(4, 1, 1),
            ),
            "str": (
                ("time", "latitude", "longitude"),
                np.array([-100.0, -120.0, -10.0, -25.0]).reshape(4, 1, 1),
            ),
        },
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )

    prepared = prepare_data_before_crop(dataset)

    assert prepared.tp.attrs["units"] == "mm"
    np.testing.assert_allclose(prepared.tp[1:, 0, 0], [20.0, 10.0, 15.0])
    assert np.isnan(prepared.tp[0, 0, 0])
    assert prepared.ssr.attrs["units"] == "W m-2"
    np.testing.assert_allclose(
        prepared.ssr[1:, 0, 0], [20.0 / 3600, 10.0 / 3600, 15.0 / 3600]
    )
    np.testing.assert_allclose(
        prepared.str[1:, 0, 0], [-20.0 / 3600, -10.0 / 3600, -15.0 / 3600]
    )
    assert np.isnan(prepared.ssr[0, 0, 0])
    assert np.isnan(prepared.str[0, 0, 0])


def test_accumulation_auxiliary_axis_keeps_00_in_the_previous_cycle() -> None:
    time = np.array(
        ["2024-01-01T23", "2024-01-02T00", "2024-01-02T01"],
        dtype="datetime64[ns]",
    )
    dataset = xr.Dataset(
        {"tp": (("time", "latitude", "longitude"), [[[0.1]], [[0.12]], [[0.01]]])},
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )

    prepared = prepare_data_before_crop(dataset)

    np.testing.assert_allclose(prepared.tp[1:, 0, 0], [20.0, 10.0])
