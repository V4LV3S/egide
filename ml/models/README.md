# Modelos CNN-LSTM

Artefatos gerados pelos treinamentos regionais e individuais.

Cada pasta de dataset pode conter:

| Artefato | Conteúdo |
| --- | --- |
| `*_final_seed<seed>.pt` | Estado do modelo, configuração, seed e épocas. |
| `validation_selection_runs.csv` | Métricas de cada execução de validação. |
| `validation_selection_summary.csv` | Média e desvio por configuração. |
| `best_model_metrics.json` | Configuração vencedora e métricas do ensemble. |
| `test_results.json` | Previsões e observações do teste na escala original. |

Os checkpoints podem ser reutilizados pelos scripts
`cnn_lstm_export_indv.py` e `cnn_lstm_export_mmgd.py`, sem novo treino.
Não mova um checkpoint para outro dataset: a quantidade e a ordem das features
precisam coincidir com o NPZ correspondente.
