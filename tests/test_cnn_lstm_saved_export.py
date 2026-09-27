import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from ml.scripts.cnn_lstm_saved_export import export_saved_results


@dataclass(frozen=True)
class FakeConfig:
    """Configuração mínima usada pelo modelo de teste."""


class FakeModel(nn.Module):
    """Modelo mínimo cujo estado representa uma previsão constante."""

    def __init__(
        self,
        past_feature_count: int,
        calendar_feature_count: int,
        config: FakeConfig,
    ) -> None:
        super().__init__()
        assert past_feature_count == 1
        assert calendar_feature_count == 1
        assert isinstance(config, FakeConfig)
        self.prediction = nn.Parameter(torch.tensor(0.0))


def fake_load_window_data(_path):
    target = np.array(
        [[10.0, 11.0], [11.0, 12.0], [12.0, 13.0]],
        dtype=np.float32,
    )
    return {
        "X_past_test": np.zeros((3, 2, 1), dtype=np.float32),
        "X_calendar_test": np.zeros((3, 2, 1), dtype=np.float32),
        "y_test": target,
    }


def fake_make_loader(*arrays, shuffle):
    assert shuffle is False
    return arrays


def fake_predict(model, loader, device):
    assert device == "cpu"
    target = loader[2]
    prediction = np.full(target.shape, model.prediction.item())
    return target, prediction


def test_export_saved_results_loads_selected_ensemble(tmp_path) -> None:
    input_directory = tmp_path / "data"
    input_directory.mkdir()
    input_path = input_directory / "usina_a_ml_input_windows_24h.npz"
    input_path.touch()
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="h"),
            "split": ["test"] * 6,
            "val_target": np.arange(8.0, 14.0),
        }
    ).to_parquet(input_directory / "usina_a_ml_input.parquet", index=False)

    original_path = tmp_path / "usina_a.parquet"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="h"),
            "val_target": np.arange(8.0, 14.0) * 2.0 + 5.0,
        }
    ).to_parquet(original_path, index=False)
    dataset_model_directory = tmp_path / "models" / "usina_a"
    dataset_model_directory.mkdir(parents=True)
    (dataset_model_directory / "best_model_metrics.json").write_text(
        json.dumps(
            {
                "best_configuration": "best",
                "seeds": [11, 22],
            }
        ),
        encoding="utf-8",
    )
    for seed, prediction in ((11, 1.0), (22, 3.0)):
        model = FakeModel(1, 1, FakeConfig())
        model.prediction.data.fill_(prediction)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {},
                "seed": seed,
            },
            dataset_model_directory / f"best_final_seed{seed}.pt",
        )

    outputs = export_saved_results(
        input_directory,
        tmp_path / "models",
        datasets=None,
        window_file_suffix="_ml_input_windows_24h.npz",
        entity_type="usina",
        model_class=FakeModel,
        config_class=FakeConfig,
        load_window_data=fake_load_window_data,
        make_loader=fake_make_loader,
        predict=fake_predict,
        original_data_path=lambda _dataset_name: original_path,
        device="cpu",
    )

    output_path = dataset_model_directory / "test_results.json"
    assert outputs == {"usina_a": output_path}
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["usina"] == "usina_a"
    assert payload["escala_valores"] == "original"
    assert payload["resultados"][0] == {
        "time0": "2026-01-01T02:00:00",
        "valores_previstos": [9.0, 9.0],
        "valores_observados": [25.0, 27.0],
    }
