# Resultados para plotagem

Cópias consolidadas dos resultados de teste CNN-LSTM, prontas para ferramentas
externas.

Padrões de nome:

- `test_results_indv_<usina>.json`;
- `test_results_mmgd_<regiao_geoletrica>.json`.

As previsões e observações estão na escala original. Nesta pasta, valores
negativos são limitados a `0.0`; a quantidade alterada é registrada em
`quantidade_valores_negativos_ajustados`. Os resultados originais em
`ml/models` não são modificados.

Regere a pasta com:

```bash
uv run python ml/scripts/copy_cnn_lstm_plot_results.py
```
