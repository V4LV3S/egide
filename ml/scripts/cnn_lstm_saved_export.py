"""Inferência e exportação a partir de checkpoints CNN-LSTM existentes."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # Permite importar como pacote e executar os wrappers diretamente.
    from .cnn_lstm_results import (
        EntityType,
        fit_original_target_transform,
        load_split_target_times,
        write_test_results,
    )
except ImportError:  # pragma: no cover - caminho usado na execução direta.
    from cnn_lstm_results import (
        EntityType,
        fit_original_target_transform,
        load_split_target_times,
        write_test_results,
    )


def find_export_datasets(
    input_directory: Path,
    *,
    window_file_suffix: str,
    datasets: tuple[str, ...] | None,
) -> list[tuple[str, Path]]:
    """Resolve os datasets que terão previsões exportadas."""
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Diretório de entrada não encontrado: {input_directory}"
        )

    if datasets is None:
        input_paths = sorted(input_directory.glob(f"*{window_file_suffix}"))
        resolved = [
            (path.name.removesuffix(window_file_suffix), path)
            for path in input_paths
        ]
    else:
        resolved = [
            (name, input_directory / f"{name}{window_file_suffix}")
            for name in datasets
        ]

    if not resolved:
        raise FileNotFoundError(
            f"Nenhum arquivo *{window_file_suffix} encontrado em "
            f"{input_directory}"
        )
    missing = [str(path) for _, path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Datasets configurados não encontrados:\n" + "\n".join(missing)
        )
    return resolved


def export_dataset_from_checkpoints(
    dataset_name: str,
    input_path: Path,
    model_directory: Path,
    *,
    entity_type: EntityType,
    model_class: type[nn.Module],
    config_class: Callable[..., Any],
    load_window_data: Callable[[Path], dict[str, np.ndarray]],
    make_loader: Callable[..., DataLoader],
    predict: Callable[
        [nn.Module, DataLoader, str], tuple[np.ndarray, np.ndarray]
    ],
    original_data_path: Callable[[str], Path],
    device: str,
) -> Path:
    """Carrega o ensemble salvo e exporta suas previsões do teste."""
    dataset_model_directory = model_directory / dataset_name
    metrics_path = dataset_model_directory / "best_model_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Metadados do modelo não encontrados: {metrics_path}"
        )

    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    try:
        best_configuration = str(metrics["best_configuration"])
        seeds = [int(seed) for seed in metrics["seeds"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Metadados inválidos em {metrics_path}: faltam configuração ou seeds."
        ) from error
    if not seeds:
        raise ValueError(f"Nenhuma seed registrada em {metrics_path}.")

    checkpoint_paths = [
        dataset_model_directory
        / f"{best_configuration}_final_seed{seed}.pt"
        for seed in seeds
    ]
    missing_checkpoints = [
        str(path) for path in checkpoint_paths if not path.is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Checkpoints do ensemble não encontrados:\n"
            + "\n".join(missing_checkpoints)
        )

    windows = load_window_data(input_path)
    test_arrays = tuple(
        windows[f"{prefix}_test"]
        for prefix in ("X_past", "X_calendar", "y")
    )
    test_loader = make_loader(*test_arrays, shuffle=False)
    source_parquet_path = input_path.with_name(
        f"{dataset_name}_ml_input.parquet"
    )
    test_target_times = load_split_target_times(
        source_parquet_path,
        split="test",
        lookback=windows["X_past_test"].shape[1],
        horizon=windows["y_test"].shape[1],
        expected_sample_count=len(windows["y_test"]),
    )

    predictions = []
    test_true = windows["y_test"]
    for seed, checkpoint_path in zip(seeds, checkpoint_paths, strict=True):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if int(checkpoint.get("seed", seed)) != seed:
            raise ValueError(
                f"Seed interna incompatível no checkpoint: {checkpoint_path}"
            )
        try:
            config = config_class(**checkpoint["model_config"])
            model_state = checkpoint["model_state_dict"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Checkpoint inválido ou incompleto: {checkpoint_path}"
            ) from error

        model = model_class(
            past_feature_count=windows["X_past_test"].shape[-1],
            calendar_feature_count=windows["X_calendar_test"].shape[-1],
            config=config,
        ).to(device)
        model.load_state_dict(model_state)
        checkpoint_true, checkpoint_prediction = predict(
            model, test_loader, device
        )
        if not np.array_equal(checkpoint_true, test_true):
            raise ValueError(
                f"Alvos desalinhados ao processar {checkpoint_path}."
            )
        predictions.append(checkpoint_prediction)

    ensemble_prediction = np.mean(predictions, axis=0)
    inverse_transform = fit_original_target_transform(
        source_parquet_path,
        original_data_path(dataset_name),
    )
    output_path = dataset_model_directory / "test_results.json"
    write_test_results(
        output_path,
        dataset_name=dataset_name,
        entity_type=entity_type,
        target_times=test_target_times,
        observed_values=inverse_transform.inverse(test_true),
        predicted_values=inverse_transform.inverse(ensemble_prediction),
        value_scale="original",
    )
    print(
        f"{dataset_name}: {len(checkpoint_paths)} checkpoints -> {output_path}"
    )
    return output_path


def export_saved_results(
    input_directory: Path,
    model_directory: Path,
    *,
    datasets: tuple[str, ...] | None,
    window_file_suffix: str,
    entity_type: EntityType,
    model_class: type[nn.Module],
    config_class: Callable[..., Any],
    load_window_data: Callable[[Path], dict[str, np.ndarray]],
    make_loader: Callable[..., DataLoader],
    predict: Callable[
        [nn.Module, DataLoader, str], tuple[np.ndarray, np.ndarray]
    ],
    original_data_path: Callable[[str], Path],
    device: str,
) -> dict[str, Path]:
    """Exporta todos os datasets selecionados sem realizar treinamento."""
    resolved_datasets = find_export_datasets(
        input_directory,
        window_file_suffix=window_file_suffix,
        datasets=datasets,
    )
    print(f"Dispositivo: {device}")
    print("Datasets: " + ", ".join(name for name, _ in resolved_datasets))

    return {
        dataset_name: export_dataset_from_checkpoints(
            dataset_name,
            input_path,
            model_directory,
            entity_type=entity_type,
            model_class=model_class,
            config_class=config_class,
            load_window_data=load_window_data,
            make_loader=make_loader,
            predict=predict,
            original_data_path=original_data_path,
            device=device,
        )
        for dataset_name, input_path in resolved_datasets
    }
