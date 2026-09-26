import numpy as np
import pandas as pd

from ml.scripts.create_window_datasets import create_window_files


def write_training_parquet(path) -> None:
    """Escreve uma série contínua com três splits para testar o executor."""
    timestamps = pd.date_range("2024-01-01", periods=150, freq="h")
    pd.DataFrame(
        {
            "time": timestamps,
            "split": np.repeat(["train", "validation", "test"], 50),
            "val_cargammgd": np.arange(150, dtype=float),
            "t2m_pca_0": np.arange(150, dtype=float),
            "hour_sin": np.arange(150, dtype=float),
            "hour_cos": np.arange(150, dtype=float),
            "week_day_sin": np.arange(150, dtype=float),
            "week_day_cos": np.arange(150, dtype=float),
            "month_day_sin": np.arange(150, dtype=float),
            "month_day_cos": np.arange(150, dtype=float),
            "month_sin": np.arange(150, dtype=float),
            "month_cos": np.arange(150, dtype=float),
        }
    ).to_parquet(path, index=False)


def test_create_window_files_writes_compressed_npz(tmp_path) -> None:
    input_directory = tmp_path / "training"
    input_directory.mkdir()
    write_training_parquet(input_directory / "BA_SE_ml_input.parquet")
    output_directory = tmp_path / "windows"

    output_paths = create_window_files(input_directory, output_directory)

    output_path = output_directory / "BA_SE_ml_input_windows_24h.npz"
    assert output_paths == [output_path]
    with np.load(output_path) as output:
        assert sorted(output.files) == [
            "X_calendar_test",
            "X_calendar_train",
            "X_calendar_val",
            "X_past_test",
            "X_past_train",
            "X_past_val",
            "y_test",
            "y_train",
            "y_val",
        ]
        assert output["X_past_train"].shape == (3, 24, 2)
        assert output["X_calendar_val"].shape == (3, 24, 8)
        assert output["y_test"].shape == (3, 24)


def test_create_window_files_does_not_overwrite_outputs(tmp_path) -> None:
    input_directory = tmp_path / "training"
    input_directory.mkdir()
    write_training_parquet(input_directory / "BA_SE_ml_input.parquet")
    output_directory = tmp_path / "windows"

    create_window_files(input_directory, output_directory)

    assert create_window_files(input_directory, output_directory) == []
