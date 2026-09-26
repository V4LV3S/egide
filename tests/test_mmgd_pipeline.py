import numpy as np
import pandas as pd
import pytest

import ml.scripts.create_mmgd_features as runner
from ml.scripts.mmgd_pipeline import build_mmgd_dataset


def write_mmgd_file(path) -> None:
    """Escreve uma série mínima de geração MMGD para os testes."""
    data = pd.DataFrame(
        {
            "time": [
                *pd.date_range("2023-10-01", periods=9, freq="h"),
                pd.Timestamp("2026-08-31 23:00:00"),
            ],
            "val_cargammgd": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0],
        }
    )
    data.to_parquet(path, index=False)


def test_mmgd_dataset_uses_train_only_rolling_p99(tmp_path) -> None:
    source_path = tmp_path / "BA_SE.parquet"
    write_mmgd_file(source_path)

    result = build_mmgd_dataset(
        source_path,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert result.columns.tolist() == ["time", "split", "val_cargammgd"]
    assert result["split"].tolist() == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    assert result.loc[0, "val_cargammgd"] == 0.0
    # O P99 do treino [0, ..., 5] é 4,95. Os valores extremos futuros não
    # entram na janela e, portanto, não alteram a escala de validação/teste.
    np.testing.assert_allclose(result.loc[5, "val_cargammgd"], 5.0 / 4.95)
    np.testing.assert_allclose(result.loc[6, "val_cargammgd"], 10_000.0 / 4.95)
    np.testing.assert_allclose(result.loc[9, "val_cargammgd"], 10_000.0 / 4.95)


def test_mmgd_dataset_crops_to_expected_time_range(tmp_path) -> None:
    source_path = tmp_path / "BA_SE.parquet"
    write_mmgd_file(source_path)
    data = pd.read_parquet(source_path)
    outside_range = pd.DataFrame(
        {
            "time": pd.to_datetime(["2023-09-30 23:00:00", "2026-09-01 00:00:00"]),
            "val_cargammgd": [100.0, 100.0],
        }
    )
    pd.concat([data, outside_range], ignore_index=True).to_parquet(source_path, index=False)

    result = build_mmgd_dataset(
        source_path,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert len(result) == 10
    assert result["time"].iloc[0] == pd.Timestamp("2023-10-01 00:00:00")
    assert result["time"].iloc[-1] == pd.Timestamp("2026-08-31 23:00:00")


@pytest.mark.parametrize(
    ("row", "timestamp"),
    [
        (0, "2023-10-01 00:30:00"),
        (-1, "2026-08-31 22:00:00"),
    ],
)
def test_mmgd_dataset_rejects_unexpected_time_range(tmp_path, row, timestamp) -> None:
    source_path = tmp_path / "BA_SE.parquet"
    write_mmgd_file(source_path)
    data = pd.read_parquet(source_path)
    data.loc[data.index[row], "time"] = pd.Timestamp(timestamp)
    data.to_parquet(source_path, index=False)

    with pytest.raises(ValueError, match="A série deve cobrir exatamente"):
        build_mmgd_dataset(
            source_path,
            train_fraction=0.60,
            validation_fraction=0.20,
        )


def test_runner_writes_one_normalized_file_per_area(tmp_path, monkeypatch) -> None:
    input_directory = tmp_path / "carga-horaria-mmgd"
    input_directory.mkdir()
    write_mmgd_file(input_directory / "CE_cargaverificada.parquet")
    output_directory = tmp_path / "mmgd"
    monkeypatch.setattr(runner, "INPUT_DIRECTORY", input_directory)
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", output_directory)
    monkeypatch.setattr(runner, "FILES_TO_PROCESS", None)
    monkeypatch.setattr(runner, "OVERWRITE_EXISTING_OUTPUTS", False)
    monkeypatch.setattr(runner, "TRAIN_FRACTION", 0.60)
    monkeypatch.setattr(runner, "VALIDATION_FRACTION", 0.20)
    monkeypatch.setattr(runner, "ROLLING_WINDOW_DAYS", 60)

    outputs = runner.main()

    output_path = output_directory / "CE_cargaverificada.parquet"
    assert outputs == [output_path]
    assert list(pd.read_parquet(output_path).columns) == ["time", "split", "val_cargammgd"]
