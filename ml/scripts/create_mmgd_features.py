"""Configuração editável para normalizar as séries de geração MMGD."""
from __future__ import annotations

from pathlib import Path

try:  # Permite importar o módulo nos testes e executá-lo diretamente pela IDE.
    from .mmgd_pipeline import VALUE_COLUMN, create_mmgd_parquet_files
except ImportError:  # pragma: no cover - caminho usado somente na execução direta.
    from mmgd_pipeline import VALUE_COLUMN, create_mmgd_parquet_files


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "ml" / "data" / "carga-horaria-mmgd"
OUTPUT_DIRECTORY = ROOT / "ml" / "data" / "mmgd"
FILES_TO_PROCESS: tuple[str, ...] | None = None
OVERWRITE_EXISTING_OUTPUTS = False
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.20
ROLLING_WINDOW_DAYS = 60


def main() -> list[Path]:
    """Executa a divisão e normalização conforme a configuração deste módulo."""
    return create_mmgd_parquet_files(
        input_directory=INPUT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        value_column=VALUE_COLUMN,
        rolling_window_days=ROLLING_WINDOW_DAYS,
        files_to_process=FILES_TO_PROCESS,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
    )


if __name__ == "__main__":
    main()
