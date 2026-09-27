# Scripts de Machine Learning

Scripts reproduzíveis para preparar dados, treinar CNN-LSTM e exportar
resultados. Execute-os a partir da raiz com `uv run python <script>`.

## Preparação

| Script | Função |
| --- | --- |
| `create_mmgd_features.py` | Divide e normaliza geração MMGD por P99 do treino. |
| `create_indv_features.py` | Divide e padroniza geração por usina com StandardScaler do treino. |
| `create_pca_features.py` | Imputa, padroniza e aplica PCA aos campos meteorológicos regionais. |
| `create_meteoro_indiv_features.py` | Cria features meteorológicas locais por usina. |
| `create_window_datasets.py` | Converte Parquets consolidados em janelas NPZ. |

Os módulos `mmgd_pipeline.py`, `indv_pipeline.py`, `pca_pipeline.py` e
`window_pipeline.py` concentram a lógica reutilizável e validada por testes.
Antes de executar `create_window_datasets.py`, ajuste `INPUT_DIRECTORY` e
`TARGET_COLUMN` para o fluxo MMGD ou individual desejado.

As janelas padrão usam 24 horas de histórico e produzem as 24 horas seguintes,
sem atravessar partições ou lacunas temporais.

## Treinamento CNN-LSTM

| Script | Escopo | Saída |
| --- | --- | --- |
| `cnn_lstm_training_mmgd.py` | MMGD por região geoelétrica | `ml/models/cnn_lstm_mmgd/<dataset>` |
| `cnn_lstm_training_indv.py` | Geração por usina | `ml/models/cnn_lstm_indv/<dataset>` |

```bash
uv run python ml/scripts/cnn_lstm_training_mmgd.py
uv run python ml/scripts/cnn_lstm_training_indv.py
```

Cada treinamento seleciona a configuração pela validação, ajusta o modelo final
em treino + validação e avalia uma única vez no teste. Os artefatos incluem
checkpoints por seed, tabelas de seleção, métricas e `test_results.json`.

Por padrão são usadas múltiplas seeds e redes com convoluções e LSTMs. GPU é
recomendada; em CPU, o treinamento completo pode ser demorado. Para limitar a
execução, edite `DATASETS` e, para experimentos rápidos, `N_SEEDS`,
`MAX_EPOCHS` e `MODEL_CONFIG_GRID`.

## Exportação sem retreinar

Quando os checkpoints já existem:

```bash
uv run python ml/scripts/cnn_lstm_export_mmgd.py
uv run python ml/scripts/cnn_lstm_export_indv.py
```

Os exportadores leem `best_model_metrics.json`, carregam apenas a configuração
vencedora, executam todas as seeds registradas e recalculam o ensemble. As
previsões e observações são convertidas para a escala da série original.

Formato resumido:

```json
{
  "tipo_serie": "regiao_geoletrica",
  "regiao_geoletrica": "BA_SE",
  "escala_valores": "original",
  "intervalo_horas": 1,
  "horizonte_horas": 24,
  "resultados": [
    {
      "time0": "2026-05-18T09:00:00",
      "valores_previstos": [1064.25, 1483.99],
      "valores_observados": [1165.17, 1681.42]
    }
  ]
}
```

Cada item é uma janela; as posições das listas avançam uma hora a partir de
`time0`.

## Pasta para plots

```bash
uv run python ml/scripts/copy_cnn_lstm_plot_results.py
```

O comando copia todos os resultados para `cnn_lstm_plot_results/`, acrescenta
o nome da usina ou região ao arquivo e troca valores negativos por `0.0`
somente nas cópias.

## Testes relacionados

```bash
uv run pytest tests/test_window_pipeline.py -q
uv run pytest tests/test_cnn_lstm_results.py \
  tests/test_cnn_lstm_saved_export.py \
  tests/test_copy_cnn_lstm_plot_results.py -q
```
