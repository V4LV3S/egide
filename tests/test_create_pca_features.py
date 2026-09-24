import numpy as np
import pandas as pd
import xarray as xr

from ml.scripts.create_pca_features import (
    apply_log1p_transform,
    build_ml_feature_matrix,
    create_data_features,
    normalize_data,
    prepare_feature_matrix,
    process_all_regions,
    save_date_features_parquet,
    split_data,
)


def write_meteorological_file(
    path,
    variable_name: str,
    multiplier: float,
    first_hour: int = 0,
    has_missing_value: bool = False,
) -> None:
    """Cria um NetCDF espacial pequeno para exercitar o fluxo PCA."""
    time_index = np.arange(first_hour, 20, dtype=float)
    values = multiplier * (
        time_index[:, None, None]
        + np.array([[[0.0, 1.0], [2.0, 4.0]]])
    )
    if has_missing_value:
        values[1, 0, 0] = np.nan
    dataset = xr.Dataset(
        {
            variable_name: (
                ("time", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "time": np.datetime64("2024-01-01T00")
            + time_index.astype("timedelta64[h]"),
            "latitude": [-10.0, -9.0],
            "longitude": [-40.0, -39.0],
        },
    )
    dataset.to_netcdf(path)


def test_process_all_regions_combines_pca_variables_by_region(tmp_path) -> None:
    input_directory = tmp_path / "meteoro-recortado"
    region_directory = input_directory / "BA_SE"
    region_directory.mkdir(parents=True)
    write_meteorological_file(
        region_directory / "2t.nc", "t2m", 1.0, has_missing_value=True
    )
    write_meteorological_file(region_directory / "sp.nc", "sp", 2.0, first_hour=1)
    output_directory = tmp_path / "ml-data"

    outputs = process_all_regions(
        input_directory=input_directory,
        output_directory=output_directory,
        pca_configuration={"wind": {"t2m": 1}, "solar": {"sp": 2}},
    )

    assert outputs == [
        output_directory / "BA_SE" / "wind" / "wind_pca.nc",
        output_directory / "BA_SE" / "solar" / "solar_pca.nc",
    ]
    with xr.open_dataset(outputs[0]) as wind_result:
        assert set(wind_result.data_vars) == {"t2m_pca"}
        assert wind_result.sizes["time"] == 20
        assert wind_result.sizes["t2m_pca_component"] == 1
        assert wind_result.attrs["dataset_name"] == "wind"
        assert wind_result.attrs["variables"] == "t2m"
        assert wind_result.attrs["pca_components_requested"] == '{"t2m": 1}'

    with xr.open_dataset(outputs[1]) as solar_result:
        assert set(solar_result.data_vars) == {"sp_pca"}
        assert solar_result.sizes["time"] == 19
        assert solar_result.sizes["sp_pca_component"] == 2
        assert solar_result.attrs["dataset_name"] == "solar"
        assert solar_result.attrs["variables"] == "sp"
        assert solar_result.attrs["pca_components_requested"] == '{"sp": 2}'

    with xr.open_dataset(outputs[1]) as result:
        assert result.sizes["time"] == 19
        assert result.split.values.tolist() == (
            ["train"] * 13 + ["validation"] * 3 + ["test"] * 3
        )
        assert result.attrs["region"] == "BA_SE"
        assert result.attrs["time_alignment"] == "inner"

    wind_parquet = pd.read_parquet(output_directory / "BA_SE" / "wind" / "t2m_pca.parquet")
    solar_parquet = pd.read_parquet(output_directory / "BA_SE" / "solar" / "sp_pca.parquet")
    assert wind_parquet.columns.tolist() == ["time", "split", "t2m_pca"]
    assert solar_parquet.columns.tolist() == ["time", "split", "sp_pca_0", "sp_pca_1"]
    assert wind_parquet.shape == (20, 3)
    assert solar_parquet.shape == (19, 4)


def test_normalization_is_fitted_only_with_the_training_partition(tmp_path) -> None:
    source_path = tmp_path / "t2m.nc"
    write_meteorological_file(source_path, "t2m", 1.0)
    with xr.open_dataset(source_path) as dataset:
        matrix = prepare_feature_matrix(dataset.t2m.load(), source_path)

    split = split_data(matrix, source_path)
    normalized = normalize_data(split)

    np.testing.assert_allclose(normalized.train.mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(normalized.validation.mean(axis=0), 0.0)


def test_normalization_can_skip_standard_scaler_for_tcc(tmp_path) -> None:
    source_path = tmp_path / "tcc.nc"
    write_meteorological_file(source_path, "tcc", 0.01)
    with xr.open_dataset(source_path) as dataset:
        matrix = prepare_feature_matrix(dataset.tcc.load(), source_path)

    split = split_data(matrix, source_path)
    normalized = normalize_data(split, standardize=False)

    np.testing.assert_allclose(normalized.train, split.train)
    np.testing.assert_allclose(normalized.validation, split.validation)
    np.testing.assert_allclose(normalized.test, split.test)


def test_log1p_for_precipitation_preserves_zero_and_compresses_high_values(tmp_path) -> None:
    source_path = tmp_path / "tp.nc"
    matrix = xr.DataArray(
        [[0.0, np.nan], [1.0, 99.0]],
        dims=("time", "spatial_point"),
    )

    transformed = apply_log1p_transform(matrix, source_path)

    np.testing.assert_allclose(transformed.values[[0, 1], [0, 0]], [0.0, np.log(2)])
    assert transformed.values[1, 1] == np.log(100)
    assert np.isnan(transformed.values[0, 1])


def test_pca_configuration_accepts_a_different_value_for_each_variable(tmp_path) -> None:
    input_directory = tmp_path / "meteoro-recortado"
    region_directory = input_directory / "BA_SE"
    region_directory.mkdir(parents=True)
    write_meteorological_file(region_directory / "2t.nc", "t2m", 1.0)
    write_meteorological_file(region_directory / "sp.nc", "sp", 2.0)

    output_path = process_all_regions(
        input_directory=input_directory,
        output_directory=tmp_path / "ml-data",
        pca_configuration={"weather": {"t2m": 1, "sp": 2}},
    )[0]

    with xr.open_dataset(output_path) as result:
        assert result.sizes["t2m_pca_component"] == 1
        assert result.sizes["sp_pca_component"] == 2
        assert result.attrs["pca_components_requested"] == '{"t2m": 1, "sp": 2}'


def test_create_data_features_merges_hourly_and_prefixed_pca_features(tmp_path) -> None:
    time = np.datetime64("2024-01-01T00") + np.arange(20).astype("timedelta64[h]")
    hourly_path = tmp_path / "date_features.nc"
    xr.Dataset({"hour_sin": ("time", np.arange(20.0))}, coords={"time": time}).to_netcdf(
        hourly_path
    )

    wind_path = tmp_path / "BA_SE_wind_pca.nc"
    xr.Dataset(
        {"tp_pca": (("time", "tp_pca_component"), np.ones((20, 1)))},
        coords={
            "time": time,
            "tp_pca_component": [0],
            "split": ("time", ["train"] * 20),
        },
        attrs={"region": "BA_SE", "dataset_name": "wind"},
    ).to_netcdf(wind_path)

    solar_path = tmp_path / "CE_solar_pca.nc"
    xr.Dataset(
        {"ssr_pca": (("time", "ssr_pca_component"), np.ones((19, 2)))},
        coords={
            "time": time[1:],
            "ssr_pca_component": [0, 1],
            "split": ("time", ["train"] * 19),
        },
        attrs={"region": "CE", "dataset_name": "solar"},
    ).to_netcdf(solar_path)

    output_path = create_data_features(
        hourly_path, [wind_path, solar_path], tmp_path / "data_features.nc"
    )

    with xr.open_dataset(output_path) as result:
        assert set(result.data_vars) == {
            "hour_sin",
            "BA_SE_wind_tp_pca",
            "CE_solar_ssr_pca",
            "X",
        }
        assert result.sizes["time"] == 19
        assert result.X.dims == ("time", "feature")
        assert result.X.sizes["feature"] == 4
        assert result.feature.values.tolist() == [
            "hour_sin",
            "BA_SE_wind_tp_pca",
            "CE_solar_ssr_pca_0",
            "CE_solar_ssr_pca_1",
        ]
        assert result.time.values[0] == time[1]
        assert result.attrs["time_alignment"] == "inner"
        assert result.split.values.tolist() == (
            ["train"] * 13 + ["validation"] * 3 + ["test"] * 3
        )

    for partition, expected_size in (("train", 13), ("validation", 3), ("test", 3)):
        with xr.open_dataset(tmp_path / f"data_features_{partition}.nc") as split:
            assert set(split.data_vars) == {"X"}
            assert split.X.sizes == {"time": expected_size, "feature": 4}
            assert split.attrs["partition"] == partition

        parquet_partition = pd.read_parquet(
            tmp_path / f"data_features_{partition}.parquet"
        )
        assert parquet_partition.shape == (expected_size, 6)
        assert parquet_partition.columns.tolist() == [
            "time",
            "split",
            "hour_sin",
            "BA_SE_wind_tp_pca",
            "CE_solar_ssr_pca_0",
            "CE_solar_ssr_pca_1",
        ]
        assert parquet_partition["split"].eq(partition).all()

    parquet = pd.read_parquet(tmp_path / "data_features.parquet")
    assert parquet.shape == (19, 6)


def test_save_date_features_parquet_writes_shared_time_features(tmp_path) -> None:
    time = np.datetime64("2024-01-01T00") + np.arange(3).astype("timedelta64[h]")
    input_path = tmp_path / "date_features.nc"
    xr.Dataset(
        {
            "hour_sin": ("time", [0.0, 1.0, 0.0]),
            "weekday": ("time", [0, 0, 0]),
        },
        coords={"time": time},
    ).to_netcdf(input_path)

    output_path = save_date_features_parquet(
        input_path, tmp_path / "date_features.parquet"
    )

    result = pd.read_parquet(output_path)
    assert result.columns.tolist() == ["time", "hour_sin", "weekday"]
    assert result.shape == (3, 3)
    assert result["time"].tolist() == time.tolist()


def test_build_ml_feature_matrix_flattens_each_component_into_a_column() -> None:
    dataset = xr.Dataset(
        {
            "hour_sin": ("time", [0.0, 1.0]),
            "wind_pca": (("time", "wind_component"), [[2.0, 3.0], [4.0, 5.0]]),
        },
        coords={"time": [0, 1], "wind_component": [0, 1]},
    )

    matrix = build_ml_feature_matrix(dataset)

    assert matrix.dims == ("time", "feature")
    assert matrix.feature.values.tolist() == [
        "hour_sin",
        "wind_pca_0",
        "wind_pca_1",
    ]
    np.testing.assert_allclose(matrix.values, [[0.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
