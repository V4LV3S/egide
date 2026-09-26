"""Configuração editável para gerar features PCA meteorológicas."""
from __future__ import annotations

from pathlib import Path

from pca_pipeline import create_region_parquet_files


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"
OUTPUT_DIRECTORY = ROOT / "ml" / "data" / "meteoro"
REGIONS_TO_PROCESS: tuple[str, ...] | None = None
OVERWRITE_EXISTING_OUTPUTS = False

# Inteiro = quantidade de componentes; float em (0, 1] = variância explicada.
PCA_COMPONENTS_BY_VARIABLE: dict[str, int | float] = {
    "t2m": 0.95,
    "tp": 0.90,
    "ws100": 0.90,
    "sp": 0.90,
    "ssr": 0.90,
    "tcc": 0.90,
}
VARIABLES_WITH_LOG1P = ("tp",)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.20


def main() -> list[Path]:
    """Executa o pipeline com a configuração editável deste módulo."""
    return create_region_parquet_files(
        input_directory=INPUT_DIRECTORY,
        output_directory=OUTPUT_DIRECTORY,
        pca_configuration=PCA_COMPONENTS_BY_VARIABLE,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        variables_with_log1p=VARIABLES_WITH_LOG1P,
        regions_to_process=REGIONS_TO_PROCESS,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
    )


if __name__ == "__main__":
    main()
