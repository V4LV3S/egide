from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # salva PNG sem precisar de tela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset

# ---------- CONFIG ----------
TRAIN_DIR = Path("data/training")
SEQUENCE_DIR = TRAIN_DIR / "lstm_cnn"
MODEL_DIR = TRAIN_DIR / "model_lstm_cnn"
STATE_HOURLY_PATH = Path("data/processed/state_hourly.parquet")

SEQUENCE_LENGTH = 168  # 7 dias de histórico horário
BATCH_SIZE, EPOCHS, PATIENCE = 128, 60, 10
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
LSTM_HIDDEN, CNN_FILTERS, CNN_KERNEL, DENSE_UNITS, STATE_EMBED_DIM = 64, 32, 3, 32, 4

# Split temporal: os últimos HOLDOUT_FRACTION do dataset ficam de fora do TimeSeriesSplit,
# como um teste final nunca visto durante o desenvolvimento do modelo. O TimeSeriesSplit
# (n_splits=2 -> 3 blocos: treino/val/teste) roda só no restante. gap é em HORAS distintas,
# não em amostras, pois várias unidades espaciais compartilham a mesma hora.
HOLDOUT_FRACTION = 0.10
N_SPLITS = 2
GAP_HOURS = 168  # igual a SEQUENCE_LENGTH: nenhuma janela pode alcançar a divisão vizinha.
SEED = 42

# Plots finais (mesmo estilo do notebook de carga): um painel por dia, começando à meia-noite.
PLOT_SPLIT = "validation"      # "validation", "test" ou "holdout"
PLOT_STATE = "BA"              # id_estado mostrado nos painéis de previsão
PLOT_DAYS = 14

# Razões de restrição, na mesma ordem usada em y_reason.npy pelo pipeline de dados.
# PAR foi removida do escopo do modelo: só REL, CNF e ENE.
REASON_CODES = ["rel", "cnf", "ene"]
N_REASONS = len(REASON_CODES)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_columns() -> list[str]:
    return json.loads((TRAIN_DIR / "feature_columns.json").read_text(encoding="utf-8"))


def load_state_mapping() -> pd.DataFrame:
    return pd.read_json(TRAIN_DIR / "state_mapping.json")

# ---------- DATASET ----------
SEQUENCE_ARRAYS = {
    "X": "X.npy", "y_occ": "y_curtailment_flag.npy", "y_mw": "y_curtailment_mwmed.npy",
    "y_minutes": "y_curtailment_minutes.npy", "y_reason": "y_reason.npy", "state_idx": "state_idx.npy",
}


class SequenceDataset(Dataset):
    """
    Lê os .npy do pipeline como memmap.

    X: [N, SEQUENCE_LENGTH, n_features]
    Alvos: ocorrência, curtailment_mwmed, curtailment_minutes, reason_rel/cnf/ene/par
    state_idx: unidade espacial usada no embedding do modelo.
    """

    def __init__(self, indices: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray,
                 mw_mean: float, mw_std: float, minutes_mean: float, minutes_std: float) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.arrays = {name: np.load(SEQUENCE_DIR / path, mmap_mode="r") for name, path in SEQUENCE_ARRAYS.items()}
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.mw_mean, self.mw_std = np.float32(mw_mean), np.float32(mw_std)
        self.minutes_mean, self.minutes_std = np.float32(minutes_mean), np.float32(minutes_std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        idx = int(self.indices[position])

        x = np.asarray(self.arrays["X"][idx], dtype=np.float32).copy()
        x = (x - self.feature_mean[None, :]) / self.feature_std[None, :]  # padronização só do treino

        y_mw = (np.float32(self.arrays["y_mw"][idx]) - self.mw_mean) / self.mw_std
        y_minutes = (np.float32(self.arrays["y_minutes"][idx]) - self.minutes_mean) / self.minutes_std

        return (
            torch.from_numpy(x),
            torch.tensor(np.float32(self.arrays["y_occ"][idx])),
            torch.tensor(y_mw, dtype=torch.float32),
            torch.tensor(y_minutes, dtype=torch.float32),
            torch.from_numpy(np.asarray(self.arrays["y_reason"][idx], dtype=np.float32)),
            torch.tensor(int(self.arrays["state_idx"][idx]), dtype=torch.long),
        )

# ---------- ARQUITETURA ----------
class LSTMCNNOraculo(nn.Module):
    """
    LSTM-CNN híbrida: ramo LSTM(64)x2 + ramo Conv1D(32,k=3)-ReLU-MaxPool(2), concatenados
    com o embedding do estado e passados por uma camada densa antes das quatro cabeças.

    curtailment_mwh não é uma cabeça independente: em resolução horária, MWh = MWmed × 1h.

    Saídas:
        occurrence_logit      probabilidade de curtailment
        curtailment_mw_scaled volume (padronizado)
        curtailment_minutes_scaled  minutos de restrição (padronizado)
        reason_logits          uma probabilidade por razão (REASON_CODES)
    """

    def __init__(self, n_features: int, n_states: int, n_reasons: int = N_REASONS,
                 lstm_hidden: int = LSTM_HIDDEN, cnn_filters: int = CNN_FILTERS,
                 cnn_kernel: int = CNN_KERNEL, dense_units: int = DENSE_UNITS,
                 state_embed_dim: int = STATE_EMBED_DIM) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, lstm_hidden, num_layers=2, batch_first=True, dropout=0.20)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.state_embedding = nn.Embedding(n_states, state_embed_dim)

        cnn_time = (SEQUENCE_LENGTH - cnn_kernel + 1) // 2  # comprimento após Conv1D valid + MaxPool(2)
        fusion_input = lstm_hidden + cnn_filters * cnn_time + state_embed_dim

        self.dense = nn.Sequential(nn.Linear(fusion_input, dense_units), nn.ReLU(), nn.Dropout(0.20))
        self.head_occurrence = nn.Linear(dense_units, 1)
        self.head_mw = nn.Linear(dense_units, 1)
        self.head_minutes = nn.Linear(dense_units, 1)
        self.head_reason = nn.Linear(dense_units, n_reasons)

    def forward(self, x: torch.Tensor, state_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        lstm_repr = self.lstm(x)[0][:, -1, :]                  # [B, T, F] -> último timestamp, [B, lstm_hidden]
        cnn_repr = self.cnn(x.transpose(1, 2)).flatten(1)      # Conv1D espera [B, F, T] -> [B, cnn_filters*T']
        state_repr = self.state_embedding(state_idx)           # [B, state_embed_dim]

        fused = self.dense(torch.cat([lstm_repr, cnn_repr, state_repr], dim=1))
        return {
            "occurrence_logit": self.head_occurrence(fused).squeeze(1),
            "curtailment_mw_scaled": self.head_mw(fused).squeeze(1),
            "curtailment_minutes_scaled": self.head_minutes(fused).squeeze(1),
            "reason_logits": self.head_reason(fused),
        }

# ---------- FEATURES / SCALERS ----------
def fit_scalers(state_hourly: pd.DataFrame, feature_columns: list[str], train_end: pd.Timestamp) -> dict:
    """Ajusta os scalers só no treino (time <= train_end); não altera o parquet."""
    train = state_hourly[state_hourly["time"] <= train_end]
    if train.empty:
        raise RuntimeError("O conjunto de treinamento está vazio.")

    x = train[feature_columns].astype(np.float64)
    feature_mean = x.mean(axis=0).to_numpy(dtype=np.float32)
    feature_std = np.where(x.std(axis=0, ddof=0) < 1e-8, 1.0, x.std(axis=0, ddof=0)).astype(np.float32)

    def target_scaler(col: str) -> tuple[float, float]:
        return float(train[col].mean()), max(float(train[col].std(ddof=0)), 1e-8)

    mw_mean, mw_std = target_scaler("curtailment_mwmed")
    minutes_mean, minutes_std = target_scaler("curtailment_minutes")

    return {
        "feature_columns": feature_columns, "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "mw_mean": mw_mean, "mw_std": mw_std, "minutes_mean": minutes_mean, "minutes_std": minutes_std,
    }

# ---------- SPLIT TEMPORAL ----------
def _trim_tail(hours: np.ndarray, gap: int) -> np.ndarray:
    """Remove as últimas `gap` horas de um bloco, para abrir um buffer antes do próximo bloco."""
    return hours[: len(hours) - gap] if gap else hours


def build_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    """
    Reserva primeiro os últimos HOLDOUT_FRACTION (10%) das horas como um holdout final —
    um teste extra, além do TimeSeriesSplit, nunca visto durante o desenvolvimento do modelo
    (treino, validação, seleção de hiperparâmetros ou early stopping).

    O restante (90%) é dividido em treino/validação/teste com
    TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_HOURS).

    O split é feito sobre as HORAS distintas de end_datetime, não sobre as amostras: várias
    unidades espaciais compartilham a mesma hora, então aplicar o gap direto nas amostras
    faria 24 amostras corresponderem a poucas horas reais, não a 24h. Trabalhando em horas
    únicas, gap=24 deixa de fato 24h de buffer entre os blocos, evitando que uma janela de
    168h "veja" horas do bloco vizinho.

    TimeSeriesSplit(n_splits=N_SPLITS) sobre as horas únicas gera 3 blocos cronológicos
    (fold0/fold1/fold2); o parâmetro `gap` só corta a cauda do TRAIN de cada divisão, então
    ele já garante o buffer entre fold0 (treino) e fold1 (validação). Para o mesmo buffer
    existir também entre fold1/fold2 (val/teste) e entre o pool e o holdout, aplicamos o
    mesmo corte manualmente nessas duas fronteiras — replicando a própria regra do `gap`
    do sklearn.
    """
    end_datetime = pd.to_datetime(np.load(SEQUENCE_DIR / "end_datetime.npy", mmap_mode="r"), unit="ns")
    all_hours = np.sort(pd.unique(end_datetime))

    n_holdout = round(len(all_hours) * HOLDOUT_FRACTION)
    holdout_hours = all_hours[len(all_hours) - n_holdout:]
    pool_hours = _trim_tail(all_hours[: len(all_hours) - n_holdout], GAP_HOURS)

    splits = list(TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_HOURS).split(pool_hours))
    train_hour_idx, fold1_hour_idx = splits[0]
    _, fold2_hour_idx = splits[-1]

    train_hours = pool_hours[train_hour_idx]
    val_hours = _trim_tail(pool_hours[fold1_hour_idx], GAP_HOURS)
    test_hours = pool_hours[fold2_hour_idx]

    train_idx = np.flatnonzero(np.isin(end_datetime, train_hours))
    val_idx = np.flatnonzero(np.isin(end_datetime, val_hours))
    test_idx = np.flatnonzero(np.isin(end_datetime, test_hours))
    holdout_idx = np.flatnonzero(np.isin(end_datetime, holdout_hours))

    train_end, val_end = pd.Timestamp(train_hours.max()), pd.Timestamp(val_hours.max())
    logger.info("Split | train=%d (até %s) | val=%d (até %s) | test=%d (%s -> %s) | gap=%dh",
               len(train_idx), train_end, len(val_idx), val_end, len(test_idx),
               pd.Timestamp(test_hours.min()), pd.Timestamp(test_hours.max()), GAP_HOURS)
    logger.info("Holdout final | n=%d (%s -> %s) | %.0f%% das horas, fora do TimeSeriesSplit",
               len(holdout_idx), pd.Timestamp(holdout_hours.min()), pd.Timestamp(holdout_hours.max()),
               HOLDOUT_FRACTION * 100)

    for name, idx in (("Train", train_idx), ("Validation", val_idx), ("Test", test_idx), ("Holdout", holdout_idx)):
        if not len(idx):
            raise RuntimeError(f"{name} vazio.")
    return train_idx, val_idx, test_idx, holdout_idx, train_end, val_end

# ---------- CLASS WEIGHTS ----------
def compute_class_weights(train_idx: np.ndarray) -> tuple[float, np.ndarray]:
    """
    pos_weight = negativos / positivos, para compensar o forte desbalanço zero-inflacionado.

    O peso das razões (REL/CNF/ENE) é calculado só sobre as horas com curtailment
    (y_occ > 0.5), pois é exatamente nessas horas que a reason loss é computada
    (ver `reason_mask` em OraculoLoss.forward) — horas sem curtailment não entram
    nessa estatística.
    """
    y_occ = np.asarray(np.load(SEQUENCE_DIR / "y_curtailment_flag.npy", mmap_mode="r")[train_idx], dtype=np.float32)
    y_reason = np.asarray(np.load(SEQUENCE_DIR / "y_reason.npy", mmap_mode="r")[train_idx], dtype=np.float32)

    occurrence_weight = (len(y_occ) - y_occ.sum()) / max(y_occ.sum(), 1.0)

    y_reason_occurred = y_reason[y_occ > 0.5]
    positive_reason = y_reason_occurred.sum(axis=0)
    reason_weights = (len(y_reason_occurred) - positive_reason) / np.maximum(positive_reason, 1.0)
    return float(occurrence_weight), reason_weights.astype(np.float32)

# ---------- LOSS ----------
class OraculoLoss(nn.Module):
    """
    Multi-task: BCE ponderada (ocorrência e razões) + SmoothL1 (MWmed e minutos).
    As regressões recebem peso extra nas horas com curtailment, dado o alvo zero-inflacionado.

    A reason loss (REL/CNF/ENE) só é calculada nas amostras com curtailment
    (`reason_mask = y_occ > 0.5`): a razão de uma restrição não tem sentido
    definido quando não houve restrição, então essas horas não entram no
    gradiente da cabeça de razões.
    """

    def __init__(self, occurrence_pos_weight: float, reason_pos_weight: np.ndarray) -> None:
        super().__init__()
        self.occurrence_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(occurrence_pos_weight, dtype=torch.float32))
        self.reason_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(reason_pos_weight, dtype=torch.float32))
        self.regression_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, out: dict[str, torch.Tensor], y_occ, y_mw, y_minutes, y_reason
                ) -> tuple[torch.Tensor, dict[str, float]]:
        positive_weight = 1.0 + 4.0 * y_occ

        loss_occ = self.occurrence_loss(out["occurrence_logit"], y_occ)
        loss_mw = (self.regression_loss(out["curtailment_mw_scaled"], y_mw) * positive_weight).mean()
        loss_minutes = (self.regression_loss(out["curtailment_minutes_scaled"], y_minutes) * positive_weight).mean()

        reason_mask = y_occ > 0.5
        if reason_mask.any():
            loss_reason = self.reason_loss(out["reason_logits"][reason_mask], y_reason[reason_mask])
        else:
            loss_reason = torch.zeros((), device=y_occ.device, dtype=loss_occ.dtype)

        total = loss_occ + loss_mw + 0.5 * loss_minutes + 0.5 * loss_reason
        parts = {"occurrence": loss_occ, "mw": loss_mw, "minutes": loss_minutes, "reason": loss_reason}
        return total, {k: float(v.detach().cpu()) for k, v in parts.items()}

# ---------- EPOCH / PREDICT ----------
def _to_device(*tensors: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    return [t.to(device) for t in tensors]


def run_epoch(model: nn.Module, loader: DataLoader, criterion: OraculoLoss,
              optimizer: torch.optim.Optimizer | None, device: torch.device) -> dict[str, float]:
    """Roda uma época e retorna a loss total e as 4 losses por tarefa, todas ponderadas por amostra."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "occurrence": 0.0, "mw": 0.0, "minutes": 0.0, "reason": 0.0}
    total_samples = 0
    for x, y_occ, y_mw, y_minutes, y_reason, state_idx in loader:
        x, y_occ, y_mw, y_minutes, y_reason, state_idx = _to_device(
            x, y_occ, y_mw, y_minutes, y_reason, state_idx, device=device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model(x, state_idx)
            loss, parts = criterion(out, y_occ, y_mw, y_minutes, y_reason)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.size(0)
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        total_samples += batch_size

    n = max(total_samples, 1)
    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            mw_mean: float, mw_std: float, minutes_mean: float, minutes_std: float) -> dict[str, np.ndarray]:
    model.eval()
    out_batches = {k: [] for k in ("occurrence_probability", "mw_scaled", "minutes_scaled", "reason_probability")}
    y_batches = {k: [] for k in ("y_occ", "y_mw", "y_minutes", "y_reason")}

    for x, b_occ, b_mw, b_minutes, b_reason, state_idx in loader:
        x, state_idx = _to_device(x, state_idx, device=device)
        out = model(x, state_idx)

        out_batches["occurrence_probability"].append(torch.sigmoid(out["occurrence_logit"]).cpu().numpy())
        out_batches["mw_scaled"].append(out["curtailment_mw_scaled"].cpu().numpy())
        out_batches["minutes_scaled"].append(out["curtailment_minutes_scaled"].cpu().numpy())
        out_batches["reason_probability"].append(torch.sigmoid(out["reason_logits"]).cpu().numpy())
        for key, batch in (("y_occ", b_occ), ("y_mw", b_mw), ("y_minutes", b_minutes), ("y_reason", b_reason)):
            y_batches[key].append(batch.numpy())

    result = {k: np.concatenate(v) for k, v in {**out_batches, **y_batches}.items()}

    # Volta às unidades físicas; curtailment fisicamente não pode ser negativo.
    pred_mw = np.clip(result["mw_scaled"] * mw_std + mw_mean, 0.0, None)
    pred_minutes = np.clip(result["minutes_scaled"] * minutes_std + minutes_mean, 0.0, 60.0)
    true_mw = result["y_mw"] * mw_std + mw_mean
    true_minutes = result["y_minutes"] * minutes_std + minutes_mean

    return {
        "occurrence_probability": result["occurrence_probability"],
        "pred_mw": pred_mw, "pred_mwh": pred_mw.copy(), "pred_minutes": pred_minutes,
        "reason_probability": result["reason_probability"],
        "y_occ": result["y_occ"], "y_mw": true_mw, "y_minutes": true_minutes, "y_reason": result["y_reason"],
    }

# ---------- MÉTRICAS ----------
REGRESSION_METRIC_NAMES = ("mae", "rmse", "nrmse_percent", "r2", "mape_percent", "smape_percent")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Mesmas métricas do notebook de carga (cnn_lstm_training.ipynb), nas unidades físicas.

    MAPE/SMAPE seguem a fórmula do notebook (denominador mínimo 1e-6). Como o curtailment
    é zero-inflacionado, elas explodem nas horas com y_true = 0; por isso também são
    reportadas `*_positive`, calculadas só nas horas com curtailment real (y_true > 0).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    absolute_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(np.square(error))))
    target_range = float(np.ptp(y_true)) if y_true.size else 0.0
    residual_sum_of_squares = float(np.sum(np.square(error)))
    total_sum_of_squares = float(np.sum(np.square(y_true - y_true.mean())))

    def mape(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float(100 * np.mean(absolute_error[mask] / np.maximum(np.abs(y_true[mask]), 1e-6)))

    def smape(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        denominator = np.maximum(np.abs(y_true[mask]) + np.abs(y_pred[mask]), 1e-6)
        return float(100 * np.mean(2 * absolute_error[mask] / denominator))

    everything, positive = np.ones_like(y_true, dtype=bool), y_true > 0
    return {
        "mae": float(absolute_error.mean()),
        "rmse": rmse,
        "nrmse_percent": float(100 * rmse / max(target_range, 1e-6)),
        "r2": float(1 - residual_sum_of_squares / max(total_sum_of_squares, 1e-6)),
        "mape_percent": mape(everything),
        "smape_percent": smape(everything),
        "mape_percent_positive": mape(positive),
        "smape_percent_positive": smape(positive),
    }


def calculate_metrics(pred: dict[str, np.ndarray]) -> dict:
    y_occ, p_occ = pred["y_occ"].astype(int), pred["occurrence_probability"]
    occurrence_binary = (p_occ >= 0.5).astype(int)

    metrics = {}
    if len(np.unique(y_occ)) > 1:
        metrics["occurrence_roc_auc"] = float(roc_auc_score(y_occ, p_occ))
        metrics["occurrence_pr_auc"] = float(average_precision_score(y_occ, p_occ))
    else:
        metrics["occurrence_roc_auc"] = metrics["occurrence_pr_auc"] = np.nan

    metrics["occurrence_precision"] = float(precision_score(y_occ, occurrence_binary, zero_division=0))
    metrics["occurrence_recall"] = float(recall_score(y_occ, occurrence_binary, zero_division=0))
    metrics["occurrence_f1"] = float(f1_score(y_occ, occurrence_binary, zero_division=0))

    for target, pred_key, true_key in (("mw", "pred_mw", "y_mw"), ("minutes", "pred_minutes", "y_minutes")):
        for metric_name, value in regression_metrics(pred[true_key], pred[pred_key]).items():
            metrics[f"curtailment_{target}_{metric_name}"] = value

    y_reason, p_reason = pred["y_reason"].astype(int), pred["reason_probability"]
    reason_binary = (p_reason >= 0.5).astype(int)
    for i, name in enumerate(REASON_CODES):
        metrics[f"reason_{name}_f1"] = float(f1_score(y_reason[:, i], reason_binary[:, i], zero_division=0))
    return metrics

# ---------- PIPELINE ----------
def make_dataset(idx: np.ndarray, scalers: dict, feature_mean: np.ndarray, feature_std: np.ndarray) -> SequenceDataset:
    return SequenceDataset(idx, feature_mean, feature_std, scalers["mw_mean"], scalers["mw_std"],
                           scalers["minutes_mean"], scalers["minutes_std"])


def make_loader(dataset: SequenceDataset, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
                       pin_memory=device.type == "cuda")


def evaluate_split(name: str, idx: np.ndarray, loader: DataLoader, model: nn.Module,
                    device: torch.device, scalers: dict) -> dict:
    """Roda predict + métricas num split e salva `{name}_metrics.json` e `{name}_predictions.parquet`."""
    predictions = predict(model, loader, device, scalers["mw_mean"], scalers["mw_std"],
                         scalers["minutes_mean"], scalers["minutes_std"])
    metrics = calculate_metrics(predictions)
    (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("===== %s =====", name.upper())
    for metric_name, value in metrics.items():
        logger.info("%s = %.6f", metric_name, value)

    end_datetime = np.load(SEQUENCE_DIR / "end_datetime.npy", mmap_mode="r")[idx]
    state_idx = np.load(SEQUENCE_DIR / "state_idx.npy", mmap_mode="r")[idx]
    pred_df = pd.DataFrame({
        "time": pd.to_datetime(np.asarray(end_datetime), unit="ns"),
        "state_idx": np.asarray(state_idx),
        "curtailment_flag_true": predictions["y_occ"].astype("int8"),
        "curtailment_probability": predictions["occurrence_probability"],
        "curtailment_mwmed_true": predictions["y_mw"],
        "curtailment_mwmed_pred": predictions["pred_mw"],
        "curtailment_mwh_true": predictions["y_mw"],
        "curtailment_mwh_pred": predictions["pred_mwh"],
        "curtailment_minutes_true": predictions["y_minutes"],
        "curtailment_minutes_pred": predictions["pred_minutes"],
        **{f"reason_{r}_probability": predictions["reason_probability"][:, i] for i, r in enumerate(REASON_CODES)},
    })
    pred_df.to_parquet(MODEL_DIR / f"{name}_predictions.parquet", index=False)
    return metrics


def main() -> None:
    seed_everything()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- features e alvos ----
    feature_columns = load_feature_columns()
    n_features = len(feature_columns)
    logger.info("Features (%d): %s", n_features, feature_columns)

    state_hourly = pd.read_parquet(STATE_HOURLY_PATH)
    state_hourly["time"] = pd.to_datetime(state_hourly["time"])

    required_targets = ["curtailment_flag", "curtailment_mwmed", "curtailment_minutes",
                        *[f"reason_{c}" for c in REASON_CODES]]
    if target_missing := {c: int(state_hourly[c].isna().sum()) for c in required_targets if state_hourly[c].isna().any()}:
        raise RuntimeError(f"Targets com NaN: {target_missing}")

    # ---- sanity check das sequências ----
    x_shape = np.load(SEQUENCE_DIR / "X.npy", mmap_mode="r").shape
    logger.info("X.shape = %s", x_shape)
    if x_shape[1:] != (SEQUENCE_LENGTH, n_features):
        raise RuntimeError(f"X esperado com shape (*, {SEQUENCE_LENGTH}, {n_features}), recebido {x_shape}")

    # ---- split (holdout final + TimeSeriesSplit) e scalers (ajustados só até train_end) ----
    train_idx, val_idx, test_idx, holdout_idx, train_end, _val_end = build_splits()

    scalers = fit_scalers(state_hourly, feature_columns, train_end)
    (MODEL_DIR / "scalers.json").write_text(json.dumps(scalers, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_mean = np.asarray(scalers["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(scalers["feature_std"], dtype=np.float32)

    occurrence_pos_weight, reason_pos_weight = compute_class_weights(train_idx)
    logger.info("occurrence pos_weight=%.4f | reason pos_weight=%s", occurrence_pos_weight, reason_pos_weight)

    train_loader = make_loader(make_dataset(train_idx, scalers, feature_mean, feature_std), True, device)
    val_loader = make_loader(make_dataset(val_idx, scalers, feature_mean, feature_std), False, device)
    test_loader = make_loader(make_dataset(test_idx, scalers, feature_mean, feature_std), False, device)
    holdout_loader = make_loader(make_dataset(holdout_idx, scalers, feature_mean, feature_std), False, device)

    # ---- modelo ----
    mapping = load_state_mapping()
    n_states = len(mapping)
    logger.info("Estados no embedding (%d): %s", n_states, mapping["id_estado"].tolist())

    model = LSTMCNNOraculo(n_features=n_features, n_states=n_states).to(device)
    logger.info("Parâmetros treináveis: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = OraculoLoss(occurrence_pos_weight, reason_pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # ---- treino com early stopping ----
    # Além de train_loss/val_loss, registramos occurrence_loss, mw_loss, minutes_loss e
    # reason_loss por época (train e val), para identificar qual tarefa piora a validation loss.
    best_val_loss, best_epoch, epochs_without_improvement, history = float("inf"), 0, 0, []
    for epoch in range(1, EPOCHS + 1):
        train_parts = run_epoch(model, train_loader, criterion, optimizer, device)
        val_parts = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_parts["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_parts["loss"], "val_loss": val_parts["loss"],
            "train_occurrence_loss": train_parts["occurrence"], "val_occurrence_loss": val_parts["occurrence"],
            "train_mw_loss": train_parts["mw"], "val_mw_loss": val_parts["mw"],
            "train_minutes_loss": train_parts["minutes"], "val_minutes_loss": val_parts["minutes"],
            "train_reason_loss": train_parts["reason"], "val_reason_loss": val_parts["reason"],
        })
        logger.info(
            "Epoch %03d | loss train=%.6f val=%.6f | occurrence=%.4f/%.4f mw=%.4f/%.4f "
            "minutes=%.4f/%.4f reason=%.4f/%.4f | lr=%.2e",
            epoch, train_parts["loss"], val_parts["loss"],
            train_parts["occurrence"], val_parts["occurrence"], train_parts["mw"], val_parts["mw"],
            train_parts["minutes"], val_parts["minutes"], train_parts["reason"], val_parts["reason"], current_lr,
        )

        val_loss = val_parts["loss"]
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "n_features": n_features, "n_states": n_states,
                "n_reasons": N_REASONS, "sequence_length": SEQUENCE_LENGTH, "feature_columns": feature_columns,
                "architecture": ("Parallel LSTM(64)x2 + Conv1D(32,kernel=3)-ReLU-MaxPool(2) "
                                 "-> concat -> Dense(32,ReLU)"),
            }, MODEL_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.info("Early stopping na época %d.", epoch)
                break

    pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)

    # ---- avaliação no melhor checkpoint: teste (do TimeSeriesSplit) e holdout final ----
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    split_metrics = {
        "validation": evaluate_split("validation", val_idx, val_loader, model, device, scalers),
        "test": evaluate_split("test", test_idx, test_loader, model, device, scalers),
        "holdout": evaluate_split("holdout", holdout_idx, holdout_loader, model, device, scalers),
    }
    summary = build_metrics_summary(split_metrics)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    (MODEL_DIR / "best_model_metrics.json").write_text(
        json.dumps({"best_val_loss": best_val_loss, "best_epoch": best_epoch, **split_metrics},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.6f}".format):
        for target in ("mw", "minutes"):
            logger.info("===== curtailment_%s =====\n%s", target, summary[summary["target"] == target]
                        .drop(columns="target").to_string(index=False))
        logger.info("===== classificação =====\n%s", build_classification_summary(split_metrics).to_string(index=False))
    plot_training_history(pd.DataFrame(history))
    for split in ("validation", "test", "holdout"):
        plot_predictions(split, PLOT_STATE, PLOT_DAYS)
    logger.info("Artifacts salvos em: %s", MODEL_DIR)


# ---------- PLOTS ----------
def plot_training_history(history: pd.DataFrame) -> None:
    """Loss total (treino × validação) e loss de validação por tarefa, por época."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="validação")
    axes[0].set(title="Loss total", xlabel="Época", ylabel="Loss")
    for task in ("occurrence", "mw", "minutes", "reason"):
        axes[1].plot(history["epoch"], history[f"val_{task}_loss"], label=task)
    axes[1].set(title="Loss de validação por tarefa", xlabel="Época", ylabel="Loss")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "training_history.png", dpi=120)
    plt.close(fig)


def plot_predictions(split: str, state: str, n_days: int) -> None:
    """
    Real × previsto por dia (24h por painel), como no notebook de carga.
    Eixo esquerdo: curtailment em MWmed; eixo direito: probabilidade de ocorrência.
    """
    predictions = pd.read_parquet(MODEL_DIR / f"{split}_predictions.parquet")
    mapping = load_state_mapping()
    state_idx = dict(zip(mapping["id_estado"], mapping["state_idx"])).get(state)
    if state_idx is None:
        logger.warning("PLOT_STATE=%s não existe no mapeamento; plots de previsão pulados.", state)
        return

    data = predictions[predictions["state_idx"] == state_idx].sort_values("time").set_index("time")
    days = [d for d in pd.DatetimeIndex(data.index.normalize().unique())
            if len(data.loc[d:d + pd.Timedelta(hours=23)]) == 24][:n_days]
    if not days:
        logger.warning("Nenhum dia completo de %s em %s; plots de previsão pulados.", state, split)
        return

    n_columns = min(2, len(days))
    n_rows = int(np.ceil(len(days) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(14, 3.5 * n_rows), squeeze=False)
    for axis, day in zip(axes.flat, days):
        day_data = data.loc[day:day + pd.Timedelta(hours=23)]
        axis.plot(day_data.index, day_data["curtailment_mwmed_true"], marker="o", label="real (MWmed)")
        axis.plot(day_data.index, day_data["curtailment_mwmed_pred"], marker="o", label="previsão (MWmed)")
        probability_axis = axis.twinx()
        probability_axis.plot(day_data.index, day_data["curtailment_probability"], color="gray",
                              linestyle="--", alpha=0.7, label="P(curtailment)")
        probability_axis.set_ylim(0, 1)
        probability_axis.set_ylabel("Probabilidade")
        axis.set_title(f"{state} | {split} | {day:%Y-%m-%d}")
        axis.set_xlabel("Hora")
        axis.set_ylabel("Curtailment (MWmed)")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.3)
        lines = axis.get_legend_handles_labels()
        extra = probability_axis.get_legend_handles_labels()
        axis.legend(lines[0] + extra[0], lines[1] + extra[1], fontsize=8, loc="upper left")
    for axis in axes.flat[len(days):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"{split}_predictions_{state}.png", dpi=120)
    plt.close(fig)


def build_metrics_summary(split_metrics: dict[str, dict]) -> pd.DataFrame:
    """Tabela split × alvo com as métricas de regressão do notebook (mesmo layout)."""
    rows = []
    for split, metrics in split_metrics.items():
        for target in ("mw", "minutes"):
            prefix = f"curtailment_{target}_"
            rows.append({"split": split, "target": target,
                         **{k[len(prefix):]: v for k, v in metrics.items() if k.startswith(prefix)}})
    return pd.DataFrame(rows)


def build_classification_summary(split_metrics: dict[str, dict]) -> pd.DataFrame:
    """Métricas de ocorrência e razão, que não têm equivalente no notebook de carga."""
    return pd.DataFrame([
        {"split": split, **{k: v for k, v in metrics.items() if k.startswith(("occurrence_", "reason_"))}}
        for split, metrics in split_metrics.items()
    ])


if __name__ == "__main__":
    main()
