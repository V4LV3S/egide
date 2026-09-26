# Treinamento CNN-LSTM

O script `cnn_lstm_training.py` executa o mesmo treinamento do notebook para todos os arquivos `*_ml_input_windows_24h.npz` encontrados em `ml/data/training`.

Para executar manualmente:

```bash
uv run python ml/scripts/cnn_lstm_training.py
```

Também é possível abrir o arquivo e usar **Run Python File**. Para treinar somente alguns datasets, altere `DATASETS` no início do script, por exemplo:

```python
DATASETS = ("BA_SE", "CE")
```

Os modelos e as métricas são salvos em `ml/models/cnn_lstm/<dataset>`.
