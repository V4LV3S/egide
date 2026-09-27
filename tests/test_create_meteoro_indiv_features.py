from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ml.scripts.create_meteoro_indiv_features import (
    build_grid_dataset,
    create_plant_parquet_files,
    grid_dataset_to_dataframe,
)


VARIABLES = ("t2m", "ssr", "tcc", "tp")


def write_grid_files(directory: Path) -> None:
    """Cria quatro variáveis com grade 2x2 e dez instantes."""
    directory.mkdir()
    time = pd.date_range("2025-01-01", periods=10, freq="h")
    base = np.arange(40, dtype=float).reshape(10, 2, 2) + 1
    for offset, variable in enumerate(VARIABLES):
        values = base + offset * 100
        if variable == "tp":
            values = values / 1000
        xr.Dataset(
            {
                variable: (
                    ("time", "latitude", "longitude"),
                    values,
                )
            },
            coords={
                "time": time,
                "latitude": [-5.5, -5.8],
                "longitude": [-37.3, -37.0],
            },
        ).to_netcdf(directory / f"{variable}.nc")


def test_build_grid_dataset_keeps_four_features_per_variable(
    tmp_path: Path,
) -> None:
    plant_directory = tmp_path / "conj_teste"
    write_grid_files(plant_directory)

    dataset = build_grid_dataset(
        plant_directory,
        VARIABLES,
        train_fraction=0.60,
        validation_fraction=0.20,
        variables_with_log1p=("tp",),
    )
    frame = grid_dataset_to_dataframe(dataset)

    assert frame.columns.tolist() == [
        "time",
        "split",
        *[
            f"{variable}_{index}"
            for variable in VARIABLES
            for index in range(4)
        ],
    ]
    assert frame.shape == (10, 18)
    assert np.isfinite(frame.iloc[:, 2:].to_numpy()).all()
    np.testing.assert_allclose(
        frame.loc[frame["split"] == "train", "t2m_0"].mean(),
        0.0,
        atol=1e-12,
    )
    assert dataset.attrs["spatial_reduction"] == "none"


def test_create_plant_parquet_files_writes_output_per_plant(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    plant_directory = input_directory / "conj_teste"
    input_directory.mkdir()
    write_grid_files(plant_directory)

    outputs = create_plant_parquet_files(
        input_directory,
        tmp_path / "output",
        VARIABLES,
        train_fraction=0.60,
        validation_fraction=0.20,
        variables_with_log1p=("tp",),
    )

    assert outputs == [
        tmp_path / "output" / "conj_teste" / "grid_features.parquet"
    ]
