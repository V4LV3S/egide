# Machine Learning

Pipeline de previsão de geração renovável com horizonte de 24 horas.

Há dois recortes:

- geração MMGD por região geoelétrica;
- geração por usina individual.

## Fluxo

```text
data/processed
  -> scripts de agregação e features
  -> ml/data/*_ml_input.parquet
  -> janelas NPZ de 24 h
  -> treino CNN-LSTM com múltiplas seeds
  -> ml/models
  -> JSONs em escala original
  -> cnn_lstm_plot_results
```

| Pasta | Conteúdo |
| --- | --- |
| [`data/`](data/README.md) | Séries normalizadas, features, datasets consolidados e janelas. |
| [`models/`](models/README.md) | Checkpoints, seleção de hiperparâmetros, métricas e previsões. |
| [`notebooks/`](notebooks/README.md) | Exploração e versões históricas dos experimentos. |
| [`scripts/`](scripts/README.md) | Preparação, treino, inferência e exportação. |

Os treinadores usam ensemble de seeds e seleção por validação. O teste não é
usado para escolher configuração. Os JSONs finais retornam previsões e
observações na escala original.

Comandos operacionais e opções de `DATASETS` estão em
[scripts/README.md](scripts/README.md).
