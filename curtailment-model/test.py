"""
Baseline XGBoost (tabular, sem sequência): mesmos splits, mesmas amostras e mesmos targets do
LSTM+CNN, mas cada amostra usa só as 18 features da hora `t` (a última posição da janela de
X.npy), sem as 167 horas de histórico. Serve de referência: se o XGBoost tabular chegar perto
do LSTM+CNN, a parte sequencial não está agregando muito valor sobre as features pontuais; se
ficar bem atrás, a dependência temporal está de fato contribuindo.

Requer: pip install xgboost

Uso:
    python train_xgboost.py
"""
from __future__ import annotations

import json
import logging

import numpy as np
import xgboost as xgb

from training_pipe import (
    REASON_CODES, SEQUENCE_DIR, TRAIN_DIR, build_splits, calculate_metrics, load_feature_columns,
)

MODEL_DIR = TRAIN_DIR / "model_xgboost"
LSTM_MODEL_DIR = TRAIN_DIR / "model_lstm_cnn"
SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_tabular(idx: np.ndarray) -> dict[str, np.ndarray]:
    """Extrai, para cada amostra, só a última hora (t) da janela — sem histórico."""
    x_last = np.asarray(np.load(SEQUENCE_DIR / "X.npy", mmap_mode="r")[idx, -1, :], dtype=np.float32)
    state_idx = np.asarray(np.load(SEQUENCE_DIR / "state_idx.npy", mmap_mode="r")[idx], dtype=np.float32)
    features = np.column_stack([x_last, state_idx])  # state_idx entra como feature ordinal
    return {
        "X": features,
        "y_occ": np.asarray(np.load(SEQUENCE_DIR / "y_curtailment_flag.npy", mmap_mode="r")[idx], dtype=np.float32),
        "y_mw": np.asarray(np.load(SEQUENCE_DIR / "y_curtailment_mwmed.npy", mmap_mode="r")[idx], dtype=np.float32),
        "y_minutes": np.asarray(np.load(SEQUENCE_DIR / "y_curtailment_minutes.npy", mmap_mode="r")[idx], dtype=np.float32),
        "y_reason": np.asarray(np.load(SEQUENCE_DIR / "y_reason.npy", mmap_mode="r")[idx], dtype=np.float32),
    }


def train_occurrence(train: dict, val: dict) -> xgb.XGBClassifier:
    pos, neg = train["y_occ"].sum(), len(train["y_occ"]) - train["y_occ"].sum()
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float(neg / max(pos, 1.0)), eval_metric="aucpr",
        early_stopping_rounds=30, random_state=SEED, n_jobs=-1,
    )
    model.fit(train["X"], train["y_occ"], eval_set=[(val["X"], val["y_occ"])], verbose=False)
    return model


def train_regression(train: dict, val: dict, target: str) -> xgb.XGBRegressor:
    # Mesmo peso extra nas horas com curtailment usado na OraculoLoss (1 + 4*y_occ), para
    # manter a comparação justa com o LSTM+CNN.
    sample_weight = 1.0 + 4.0 * train["y_occ"]
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="mae", early_stopping_rounds=30, random_state=SEED, n_jobs=-1,
    )
    model.fit(train["X"], train[target], sample_weight=sample_weight,
              eval_set=[(val["X"], val[target])], verbose=False)
    return model


def train_reason(train: dict, val: dict, reason_idx: int) -> xgb.XGBClassifier | None:
    """Treina só nas horas com curtailment — mesma máscara (`y_occ > 0.5`) usada na OraculoLoss."""
    occurred_train, occurred_val = train["y_occ"] > 0.5, val["y_occ"] > 0.5
    y_train = train["y_reason"][occurred_train, reason_idx]
    if occurred_train.sum() == 0 or len(np.unique(y_train)) < 2:
        logger.warning("Razão %d sem exemplos suficientes no treino; pulando.", reason_idx)
        return None
    pos, neg = y_train.sum(), len(y_train) - y_train.sum()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=float(neg / max(pos, 1.0)), eval_metric="logloss",
        early_stopping_rounds=30, random_state=SEED, n_jobs=-1,
    )
    model.fit(train["X"][occurred_train], y_train,
              eval_set=[(val["X"][occurred_val], val["y_reason"][occurred_val, reason_idx])], verbose=False)
    return model


def predict_all(models: dict, data: dict) -> dict[str, np.ndarray]:
    occurrence_probability = models["occurrence"].predict_proba(data["X"])[:, 1]
    pred_mw = np.clip(models["mw"].predict(data["X"]), 0.0, None)
    pred_minutes = np.clip(models["minutes"].predict(data["X"]), 0.0, 60.0)

    reason_probability = np.zeros((len(data["X"]), len(REASON_CODES)), dtype=np.float32)
    for i, name in enumerate(REASON_CODES):
        model = models["reason"][name]
        if model is not None:
            reason_probability[:, i] = model.predict_proba(data["X"])[:, 1]

    return {
        "occurrence_probability": occurrence_probability, "pred_mw": pred_mw, "pred_minutes": pred_minutes,
        "reason_probability": reason_probability,
        "y_occ": data["y_occ"], "y_mw": data["y_mw"], "y_minutes": data["y_minutes"], "y_reason": data["y_reason"],
    }


def compare_with_lstm(name: str, xgb_metrics: dict) -> None:
    """Se o LSTM+CNN já rodou, imprime as duas métricas lado a lado."""
    lstm_path = LSTM_MODEL_DIR / f"{name}_metrics.json"
    if not lstm_path.exists():
        return
    lstm_metrics = json.loads(lstm_path.read_text(encoding="utf-8"))
    logger.info("--- Comparação XGBoost vs LSTM+CNN (%s) ---", name.upper())
    for key in xgb_metrics:
        if key in lstm_metrics:
            logger.info("%-28s xgboost=%.4f | lstm_cnn=%.4f", key, xgb_metrics[key], lstm_metrics[key])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    load_feature_columns()  # só para logar/validar que o pipeline já rodou
    train_idx, val_idx, test_idx, holdout_idx, _, _ = build_splits()

    train, val, test, holdout = (load_tabular(idx) for idx in (train_idx, val_idx, test_idx, holdout_idx))

    logger.info("Treinando XGBoost (ocorrência)...")
    occurrence_model = train_occurrence(train, val)

    logger.info("Treinando XGBoost (MWmed)...")
    mw_model = train_regression(train, val, "y_mw")

    logger.info("Treinando XGBoost (minutos)...")
    minutes_model = train_regression(train, val, "y_minutes")

    reason_models = {}
    for i, name in enumerate(REASON_CODES):
        logger.info("Treinando XGBoost (razão %s)...", name.upper())
        reason_models[name] = train_reason(train, val, i)

    models = {"occurrence": occurrence_model, "mw": mw_model, "minutes": minutes_model, "reason": reason_models}

    for name, data in (("test", test), ("holdout", holdout)):
        predictions = predict_all(models, data)
        metrics = calculate_metrics(predictions)
        (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("===== XGBOOST %s =====", name.upper())
        for metric_name, value in metrics.items():
            logger.info("%s = %.6f", metric_name, value)
        compare_with_lstm(name, metrics)

    occurrence_model.save_model(str(MODEL_DIR / "occurrence.json"))
    mw_model.save_model(str(MODEL_DIR / "mw.json"))
    minutes_model.save_model(str(MODEL_DIR / "minutes.json"))
    for name, model in reason_models.items():
        if model is not None:
            model.save_model(str(MODEL_DIR / f"reason_{name}.json"))
    logger.info("Modelos salvos em: %s", MODEL_DIR)


if __name__ == "__main__":
    main()