"""Gera arquivos NPZ comprimidos com janelas para cada Parquet consolidado."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # Permite importar nos testes e executar diretamente pela IDE.
    from .window_pipeline import build_windows_from_parquet
except ImportError:  # pragma: no cover - caminho usado somente na execução direta.
    from window_pipeline import build_windows_from_parquet


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "ml" / "data" / "training"
OUTPUT_DIRECTORY = INPUT_DIRECTORY
LOOKBACK = 24
HORIZON = 24
OVERWRITE_EXISTING_OUTPUTS = False


def create_window_files(
    input_directory: Path,
    output_directory: Path,
    *,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
    overwrite_existing_outputs: bool = False,
) -> list[Path]:
    """Cria um ``.npz`` comprimido de janelas para cada Parquet de entrada."""
    if not input_directory.is_dir():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_directory}")

    input_paths = sorted(input_directory.glob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {input_directory}")

    output_paths: list[Path] = []
    for input_path in input_paths:
        output_path = output_directory / f"{input_path.stem}_windows_{lookback}h.npz"
        if output_path.exists() and not overwrite_existing_outputs:
            print(f"Ignorando saída existente: {output_path}")
            continue

        (
            x_past_train,
            x_calendar_train,
            y_train,
            x_past_val,
            x_calendar_val,
            y_val,
            x_past_test,
            x_calendar_test,
            y_test,
        ) = build_windows_from_parquet(
            input_path,
            lookback=lookback,
            horizon=horizon,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            X_past_train=x_past_train,
            X_calendar_train=x_calendar_train,
            y_train=y_train,
            X_past_val=x_past_val,
            X_calendar_val=x_calendar_val,
            y_val=y_val,
            X_past_test=x_past_test,
            X_calendar_test=x_calendar_test,
            y_test=y_test,
        )
        print(f"{input_path.name} -> {output_path}")
        output_paths.append(output_path)
    return output_paths


def main() -> list[Path]:
    """Executa a criação conforme a configuração deste módulo."""
    return create_window_files(
        INPUT_DIRECTORY,
        OUTPUT_DIRECTORY,
        lookback=LOOKBACK,
        horizon=HORIZON,
        overwrite_existing_outputs=OVERWRITE_EXISTING_OUTPUTS,
    )


if __name__ == "__main__":
    main()
