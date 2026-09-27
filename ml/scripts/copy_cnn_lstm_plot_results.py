"""Prepara cópias dos resultados CNN-LSTM para ferramentas de plotagem."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
OUTPUT_DIRECTORY = ROOT / "cnn_lstm_plot_results"


@dataclass(frozen=True)
class ResultGroup:
    """Configuração de uma família de resultados CNN-LSTM."""

    source_directory: Path
    file_prefix: str
    entity_type: str
    entity_field: str


RESULT_GROUPS = (
    ResultGroup(
        source_directory=ROOT / "ml" / "models" / "cnn_lstm_indv",
        file_prefix="indv",
        entity_type="usina",
        entity_field="usina",
    ),
    ResultGroup(
        source_directory=ROOT / "ml" / "models" / "cnn_lstm_mmgd",
        file_prefix="mmgd",
        entity_type="regiao_geoletrica",
        entity_field="regiao_geoletrica",
    ),
)


def replace_negative_values(payload: dict[str, Any]) -> int:
    """Substitui valores negativos previstos e observados por zero."""
    results = payload.get("resultados")
    if not isinstance(results, list):
        raise ValueError("O JSON deve possuir uma lista em 'resultados'.")

    replaced_count = 0
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Cada resultado deve ser um objeto JSON.")
        for field in ("valores_previstos", "valores_observados"):
            values = result.get(field)
            if not isinstance(values, list):
                raise ValueError(f"Campo ausente ou inválido: {field}.")
            clipped_values = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"Valor não numérico encontrado em {field}.")
                numeric_value = float(value)
                if numeric_value < 0:
                    numeric_value = 0.0
                    replaced_count += 1
                clipped_values.append(numeric_value)
            result[field] = clipped_values
    return replaced_count


def copy_group_results(
    group: ResultGroup,
    output_directory: Path,
) -> list[Path]:
    """Copia e pós-processa todos os resultados de uma família."""
    input_paths = sorted(group.source_directory.glob("*/test_results.json"))
    if not input_paths:
        raise FileNotFoundError(
            f"Nenhum test_results.json encontrado em {group.source_directory}."
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for input_path in input_paths:
        with input_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("tipo_serie") != group.entity_type:
            raise ValueError(
                f"Tipo de série incompatível em {input_path}: "
                f"{payload.get('tipo_serie')!r}."
            )
        entity_name = payload.get(group.entity_field)
        if not isinstance(entity_name, str) or not entity_name:
            raise ValueError(
                f"Identificador '{group.entity_field}' ausente em {input_path}."
            )
        if entity_name != input_path.parent.name:
            raise ValueError(
                f"Identificador {entity_name!r} não corresponde à pasta "
                f"{input_path.parent.name!r}."
            )

        replaced_count = replace_negative_values(payload)
        payload["limite_inferior"] = 0.0
        payload["quantidade_valores_negativos_ajustados"] = replaced_count

        output_path = (
            output_directory
            / f"test_results_{group.file_prefix}_{entity_name}.json"
        )
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(
            f"{input_path} -> {output_path} "
            f"({replaced_count} valores ajustados)"
        )
        output_paths.append(output_path)
    return output_paths


def main() -> list[Path]:
    """Gera uma pasta consolidada de resultados prontos para plots."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return [
        output_path
        for group in RESULT_GROUPS
        for output_path in copy_group_results(group, OUTPUT_DIRECTORY)
    ]


if __name__ == "__main__":
    main()
