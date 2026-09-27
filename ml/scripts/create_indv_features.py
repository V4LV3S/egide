"""Configuração editável para normalizar geração por usina."""
from __future__ import annotations

from pathlib import Path

try:  # Permite importar o módulo nos testes e executá-lo diretamente pela IDE.
    from .indv_pipeline import create_indv_parquet_files
except ImportError:  # pragma: no cover - caminho usado somente na execução direta.
    from indv_pipeline import create_indv_parquet_files


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "ml" / "data" / "geracao_horaria_indv"
OUTPUT_DIRECTORY = ROOT / "ml" / "data" / "indiv"
FILES_TO_PROCESS: tuple[str, ...] | None = None
OVERWRITE_EXISTING_OUTPUTS = False
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.20


def main() -> list[Path]:
    """Normaliza cada série de usina conforme a configuração do módulo."""
    return create_indv_parquet_files(
        input_directory=INPUT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        files_to_process=FILES_TO_PROCESS,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
    )


if __name__ == "__main__":
    main()
