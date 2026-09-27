import json

from ml.scripts.copy_cnn_lstm_plot_results import (
    ResultGroup,
    copy_group_results,
)


def test_copy_group_results_suffixes_name_and_clips_negatives(tmp_path) -> None:
    source_directory = tmp_path / "models"
    dataset_directory = source_directory / "BA_SE"
    dataset_directory.mkdir(parents=True)
    input_path = dataset_directory / "test_results.json"
    original_payload = {
        "tipo_serie": "regiao_geoletrica",
        "regiao_geoletrica": "BA_SE",
        "escala_valores": "original",
        "resultados": [
            {
                "time0": "2026-01-01T00:00:00",
                "valores_previstos": [-2.5, 3.0, -0.1],
                "valores_observados": [0.0, -1.0, 4.0],
            }
        ],
    }
    input_path.write_text(
        json.dumps(original_payload),
        encoding="utf-8",
    )
    group = ResultGroup(
        source_directory=source_directory,
        file_prefix="mmgd",
        entity_type="regiao_geoletrica",
        entity_field="regiao_geoletrica",
    )

    output_paths = copy_group_results(group, tmp_path / "plot_results")

    expected_path = (
        tmp_path / "plot_results" / "test_results_mmgd_BA_SE.json"
    )
    assert output_paths == [expected_path]
    copied = json.loads(expected_path.read_text(encoding="utf-8"))
    assert copied["resultados"][0]["valores_previstos"] == [0.0, 3.0, 0.0]
    assert copied["resultados"][0]["valores_observados"] == [0.0, 0.0, 4.0]
    assert copied["limite_inferior"] == 0.0
    assert copied["quantidade_valores_negativos_ajustados"] == 3

    unchanged = json.loads(input_path.read_text(encoding="utf-8"))
    assert unchanged == original_payload
