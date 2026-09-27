import numpy as np
import pandas as pd
import pytest

import ml.scripts.create_indv_features as runner
from ml.scripts.indv_pipeline import build_indv_dataset


def write_indv_file(path, *, plant_name: str = "USINA TESTE") -> None:
    """Escreve uma série individual mínima para os testes."""
    data = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=10, freq="30min"),
            "nom_usina": [plant_name] * 10,
            "val_geracaoreferencia": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                10_000.0,
                10_000.0,
                10_000.0,
                10_000.0,
            ],
        }
    )
    data.to_parquet(path, index=False)


def test_indv_dataset_uses_train_only_standard_scaler(tmp_path) -> None:
    source_path = tmp_path / "usina_teste.parquet"
    write_indv_file(source_path)

    result = build_indv_dataset(
        source_path,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert result.columns.tolist() == [
        "time",
        "split",
        "val_geracaoreferencia",
    ]
    assert result["split"].tolist() == (
        ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    )
    train_values = result.loc[:5, "val_geracaoreferencia"]
    np.testing.assert_allclose(train_values.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(train_values.std(ddof=0), 1.0)
    train_mean = 2.5
    train_standard_deviation = np.std([0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(
        result.loc[9, "val_geracaoreferencia"],
        (10_000.0 - train_mean) / train_standard_deviation,
    )


def test_indv_dataset_rejects_multiple_plants(tmp_path) -> None:
    source_path = tmp_path / "duas_usinas.parquet"
    write_indv_file(source_path)
    data = pd.read_parquet(source_path)
    data.loc[data.index[-1], "nom_usina"] = "OUTRA USINA"
    data.to_parquet(source_path, index=False)

    with pytest.raises(ValueError, match="exatamente uma usina"):
        build_indv_dataset(
            source_path,
            train_fraction=0.60,
            validation_fraction=0.20,
        )


def test_runner_writes_one_normalized_file_per_plant(
    tmp_path, monkeypatch
) -> None:
    input_directory = tmp_path / "geracao_indv"
    input_directory.mkdir()
    write_indv_file(input_directory / "usina_teste.parquet")
    output_directory = tmp_path / "individual"
    monkeypatch.setattr(runner, "INPUT_DIRECTORY", input_directory)
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", output_directory)
    monkeypatch.setattr(runner, "FILES_TO_PROCESS", None)
    monkeypatch.setattr(runner, "OVERWRITE_EXISTING_OUTPUTS", False)
    monkeypatch.setattr(runner, "TRAIN_FRACTION", 0.60)
    monkeypatch.setattr(runner, "VALIDATION_FRACTION", 0.20)

    outputs = runner.main()

    output_path = output_directory / "usina_teste.parquet"
    assert outputs == [output_path]
    assert list(pd.read_parquet(output_path).columns) == [
        "time",
        "split",
        "val_geracaoreferencia",
    ]
