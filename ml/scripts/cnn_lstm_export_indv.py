"""Gera JSONs de teste usando os checkpoints CNN-LSTM de usinas."""
from __future__ import annotations

from pathlib import Path

try:  # Permite importar nos testes e executar diretamente pela IDE.
    from .cnn_lstm_saved_export import export_saved_results
    from .cnn_lstm_training_indv import (
        CNNLSTM,
        CNNLSTMConfig,
        DEVICE as TRAINING_DEVICE,
        load_window_data,
        make_loader,
        predict,
    )
except ImportError:  # pragma: no cover - caminho usado na execução direta.
    from cnn_lstm_saved_export import export_saved_results
    from cnn_lstm_training_indv import (
        CNNLSTM,
        CNNLSTMConfig,
        DEVICE as TRAINING_DEVICE,
        load_window_data,
        make_loader,
        predict,
    )


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "ml" / "data" / "training_indiv"
MODEL_DIRECTORY = ROOT / "ml" / "models" / "cnn_lstm_indv"
ORIGINAL_DATA_DIRECTORY = ROOT / "data" / "processed" / "geracao_indv"
WINDOW_FILE_SUFFIX = "_ml_input_windows_24h.npz"

# None exporta todos os datasets. Exemplo: ("conj_lagoa_dos_ventos",).
DATASETS: tuple[str, ...] | None = None
DEVICE = TRAINING_DEVICE


def original_data_path(dataset_name: str) -> Path:
    """Resolve a série física, removendo o prefixo usado em testes antigos."""
    original_name = dataset_name.removeprefix("testing_")
    return ORIGINAL_DATA_DIRECTORY / f"{original_name}.parquet"


def main() -> dict[str, Path]:
    """Exporta os resultados das usinas sem retreinar os modelos."""
    return export_saved_results(
        INPUT_DIRECTORY,
        MODEL_DIRECTORY,
        datasets=DATASETS,
        window_file_suffix=WINDOW_FILE_SUFFIX,
        entity_type="usina",
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
