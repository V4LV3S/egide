import json

import numpy as np
import pandas as pd
import pytest

from ml.scripts.cnn_lstm_results import (
    fit_original_target_transform,
    load_split_target_times,
    write_test_results,
)


def test_load_split_target_times_matches_contiguous_windows(tmp_path) -> None:
    parquet_path = tmp_path / "dataset_ml_input.parquet"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=6, freq="h"),
            "split": ["test"] * 6,
        }
    ).to_parquet(parquet_path, index=False)

    result = load_split_target_times(
        parquet_path,
        split="test",
        lookback=2,
        horizon=2,
        expected_sample_count=3,
    )

    expected = np.array(
        [
            ["2026-01-01T02:00", "2026-01-01T03:00"],
            ["2026-01-01T03:00", "2026-01-01T04:00"],
            ["2026-01-01T04:00", "2026-01-01T05:00"],
        ],
        dtype="datetime64[m]",
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("entity_type", "dataset_name"),
    [("regiao_geoletrica", "BA_SE"), ("usina", "conj_assu_sol")],
)
def test_write_test_results_uses_entity_specific_metadata(
    tmp_path, entity_type, dataset_name
) -> None:
    output_path = tmp_path / "test_results.json"
    target_times = np.array(
        [
            ["2026-01-01T02:00", "2026-01-01T03:00"],
            ["2026-01-01T03:00", "2026-01-01T04:00"],
        ],
        dtype="datetime64[m]",
    )

    write_test_results(
        output_path,
        dataset_name=dataset_name,
        entity_type=entity_type,
        target_times=target_times,
        observed_values=np.array([[10.0, 11.0], [11.0, 12.0]]),
        predicted_values=np.array([[9.5, 10.5], [10.8, 11.8]]),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tipo_serie"] == entity_type
    assert payload[entity_type] == dataset_name
    assert payload["intervalo_horas"] == 1
    assert payload["horizonte_horas"] == 2
    assert payload["resultados"][0] == {
        "time0": "2026-01-01T02:00:00",
        "valores_previstos": [9.5, 10.5],
        "valores_observados": [10.0, 11.0],
    }


def test_write_test_results_rejects_misaligned_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="mesmo formato"):
        write_test_results(
            tmp_path / "test_results.json",
            dataset_name="BA_SE",
            entity_type="regiao_geoletrica",
            target_times=np.empty((1, 2), dtype="datetime64[ns]"),
            observed_values=np.ones((1, 2)),
            predicted_values=np.ones((1, 3)),
        )


def test_fit_original_target_transform_handles_subhourly_source(
    tmp_path,
) -> None:
    normalized_path = tmp_path / "normalized.parquet"
    original_path = tmp_path / "original.parquet"
    pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01 01:00", periods=3, freq="h"),
            "split": ["test"] * 3,
            "val_target": [0.0, 1.0, 2.0],
        }
    ).to_parquet(normalized_path, index=False)
    pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01 00:30", periods=6, freq="30min"
            ),
            "val_target": [4.0, 6.0, 6.0, 8.0, 8.0, 10.0],
        }
    ).to_parquet(original_path, index=False)

    transform = fit_original_target_transform(
        normalized_path,
        original_path,
    )

    assert transform.multiplier == pytest.approx(2.0)
    assert transform.offset == pytest.approx(5.0)
    np.testing.assert_allclose(
        transform.inverse(np.array([[-1.0, 0.0, 1.0]])),
        [[3.0, 5.0, 7.0]],
    )
