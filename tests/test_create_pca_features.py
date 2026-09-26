import numpy as np
import pandas as pd
import xarray as xr

import ml.scripts.create_pca_features as runner
from ml.scripts.pca_pipeline import (
    build_pca_dataset,
    create_temporal_partitions,
    pca_dataset_to_dataframe,
    transform_variable,
)


def write_meteorological_file(
    path, variable_name: str, multiplier: float, first_hour: int = 0
) -> None:
    """Cria um NetCDF espacial pequeno para exercitar o pipeline PCA."""
    time_index = np.arange(first_hour, 20, dtype=float)
    values = multiplier * (time_index[:, None, None] + np.array([[[0.0, 1.0]]]))
    dataset = xr.Dataset(
        {variable_name: (("time", "latitude", "longitude"), values)},
        coords={
            "time": np.datetime64("2024-01-01T00")
            + time_index.astype("timedelta64[h]"),
            "latitude": [-10.0],
            "longitude": [-40.0, -39.0],
        },
    )
    dataset.to_netcdf(path)


def test_temporal_partitions_are_chronological_and_disjoint() -> None:
    partitions = create_temporal_partitions(20, 0.70, 0.15)

    assert partitions.train == slice(0, 14)
    assert partitions.validation == slice(14, 17)
    assert partitions.test == slice(17, 20)


def test_pca_fits_preprocessing_only_with_training_data(tmp_path) -> None:
    time = pd.date_range("2024-01-01", periods=10, freq="h")
    values = np.arange(10.0)[:, None]
    data_array = xr.DataArray(
        values,
        dims=("time", "latitude"),
        coords={"time": time, "latitude": [-10.0]},
        name="t2m",
    )
    partitions = create_temporal_partitions(10, 0.60, 0.20)

    result = transform_variable(
        data_array,
        tmp_path / "t2m.nc",
        partitions,
        1,
        apply_log1p=False,
    )

    # A PCA ajustada apenas no treino produz componentes centradas nesse bloco.
    np.testing.assert_allclose(result.values[partitions.train].mean(), 0.0, atol=1e-12)
    assert not np.isclose(result.values[partitions.validation].mean(), 0.0)
    assert result.attrs["pca_fit_scope"] == "train_only"


def test_build_dataset_assigns_split_and_aligns_sources(tmp_path) -> None:
    region = tmp_path / "BA_SE"
    region.mkdir()
    write_meteorological_file(region / "2t.nc", "t2m", 1.0)
    write_meteorological_file(region / "sp.nc", "sp", 2.0, first_hour=1)

    result = build_pca_dataset(
        region,
        {"t2m": 1, "sp": 1},
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    assert result.sizes["time"] == 19
    assert result.split.values.tolist() == ["train"] * 13 + ["validation"] * 3 + ["test"] * 3
    assert result.attrs["preprocessing_fit_scope"] == "train_only"
    assert result.attrs["final_normalization_fit_scope"] == "train_only"
    assert set(result.data_vars) == {"t2m_pca", "sp_pca"}


def test_parquet_matrix_contains_time_split_and_all_components(tmp_path) -> None:
    region = tmp_path / "BA_SE"
    region.mkdir()
    write_meteorological_file(region / "2t.nc", "t2m", 1.0)
    write_meteorological_file(region / "sp.nc", "sp", 2.0)
    dataset = build_pca_dataset(
        region,
        {"t2m": 1, "sp": 2},
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    result = pca_dataset_to_dataframe(dataset)

    assert result.columns.tolist() == ["time", "split", "t2m_pca", "sp_pca_0", "sp_pca_1"]
    assert result["split"].value_counts().to_dict() == {"train": 14, "validation": 3, "test": 3}

    train_features = result.loc[result["split"] == "train", ["t2m_pca", "sp_pca_0"]]
    np.testing.assert_allclose(train_features.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(train_features.std(ddof=0).to_numpy(), 1.0, atol=1e-12)


def test_runner_writes_one_parquet_per_region(tmp_path, monkeypatch) -> None:
    input_directory = tmp_path / "meteoro-recortado"
    region = input_directory / "CE"
    region.mkdir(parents=True)
    write_meteorological_file(region / "2t.nc", "t2m", 1.0)
    output_directory = tmp_path / "meteoro"
    monkeypatch.setattr(runner, "REGIONS_TO_PROCESS", None)
    monkeypatch.setattr(runner, "PCA_COMPONENTS_BY_VARIABLE", {"t2m": 1})
    monkeypatch.setattr(runner, "OVERWRITE_EXISTING_OUTPUTS", False)

    monkeypatch.setattr(runner, "INPUT_DIRECTORY", input_directory)
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", output_directory)

    outputs = runner.main()

    output_path = output_directory / "CE" / "pca_features.parquet"
    assert outputs == [output_path]
    assert list(pd.read_parquet(output_path).columns) == ["time", "split", "t2m_pca"]
