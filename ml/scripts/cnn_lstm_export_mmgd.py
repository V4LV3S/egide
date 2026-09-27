"""Gera JSONs de teste usando os checkpoints CNN-LSTM de MMGD."""
from __future__ import annotations

from pathlib import Path

try:  # Permite importar nos testes e executar diretamente pela IDE.
    from .cnn_lstm_saved_export import export_saved_results
    from .cnn_lstm_training_mmgd import (
        CNNLSTM,
        CNNLSTMConfig,
        DEVICE as TRAINING_DEVICE,
        load_window_data,
        make_loader,
        predict,
    )
except ImportError:  # pragma: no cover - caminho usado na execução direta.
    from cnn_lstm_saved_export import export_saved_results
    from cnn_lstm_training_mmgd import (
        CNNLSTM,
        CNNLSTMConfig,
        DEVICE as TRAINING_DEVICE,
        load_window_data,
        make_loader,
        predict,
    )


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "ml" / "data" / "training_mmgd"
MODEL_DIRECTORY = ROOT / "ml" / "models" / "cnn_lstm_mmgd"
ORIGINAL_DATA_DIRECTORY = ROOT / "ml" / "data" / "carga-horaria-mmgd"
WINDOW_FILE_SUFFIX = "_ml_input_windows_24h.npz"

# None exporta todas as regiões. Exemplo: ("BA_SE", "CE").
DATASETS: tuple[str, ...] | None = ("BA_SE", "AL_PE", "PB_RN", "CE", "MA", "PI")
DEVICE = TRAINING_DEVICE


def original_data_path(dataset_name: str) -> Path:
    """Resolve a série física de uma região geoelétrica."""
    return ORIGINAL_DATA_DIRECTORY / f"{dataset_name}_cargaverificada.parquet"


def main() -> dict[str, Path]:
    """Exporta os resultados das regiões sem retreinar os modelos."""
    return export_saved_results(
        INPUT_DIRECTORY,
        MODEL_DIRECTORY,
        datasets=DATASETS,
        window_file_suffix=WINDOW_FILE_SUFFIX,
        entity_type="regiao_geoletrica",
        model_class=CNNLSTM,
        config_class=CNNLSTMConfig,
        load_window_data=load_window_data,
        make_loader=make_loader,
        predict=predict,
        original_data_path=original_data_path,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
