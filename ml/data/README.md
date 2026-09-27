# Dados de Machine Learning

Artefatos intermediários e entradas finais dos modelos.

| Caminho | Conteúdo |
| --- | --- |
| `carga-horaria-mmgd/` | Séries MMGD horárias antes da normalização. |
| `geracao_horaria_indv/` | Geração individual agregada por hora. |
| `mmgd/` e `indiv/` | Alvos divididos e normalizados sem vazamento temporal. |
| `meteoro_mmgd/` | Componentes PCA meteorológicas por região. |
| `meteoro_indiv/` | Features meteorológicas locais por usina. |
| `training_mmgd/` | Parquets consolidados e janelas NPZ regionais. |
| `training_indiv/` | Parquets consolidados e janelas NPZ por usina. |
| `date_features.*` | Variáveis cíclicas de calendário. |

Os arquivos `*_ml_input_windows_24h.npz` contêm entradas passadas,
calendário futuro e alvo para treino, validação e teste. A criação das janelas
é feita por `ml/scripts/create_window_datasets.py`.
