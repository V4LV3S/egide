# Testes

A suíte cobre transformações meteorológicas, agregações ONS, criação de
features e janelas, além da exportação CNN-LSTM.

Execute tudo com:

```bash
uv run pytest
```

Para uma área específica:

```bash
uv run pytest tests/test_window_pipeline.py -q
uv run pytest tests/test_cnn_lstm_results.py -q
```

Os testes usam dados sintéticos e diretórios temporários; não devem depender de
downloads, GPU ou dos grandes artefatos locais.
