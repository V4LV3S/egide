# Pipeline ONS: curtailment eólico (Nordeste)

Pipeline em dois scripts: baixa os dados abertos do ONS e gera as sequências para treinar um modelo LSTM/CNN que prevê o *constrained-off* (curtailment) eólico por estado.

```
extract_ons.py  ->  data/raw/         (CSVs brutos)
process_ons.py  ->  data/processed/   (dataset horário)
                    data/training/    (sequências e metadados)
```

## Requisitos

- Python 3.10+
- `requests`, `pandas`, `numpy`, `pyarrow` (para gravar o parquet)

```bash
pip install requests pandas numpy pyarrow
```

## Uso

```bash
python extract_ons.py   # 1. baixa os CSVs (pula os que já existem)
python process_ons.py   # 2. processa e gera os tensores
```

## 1. `extract_ons.py`

Baixa os arquivos brutos do bucket público do ONS, sem transformação. Usa retry com backoff para erros transitórios e não baixa de novo arquivos já existentes.

| Dataset | Pasta em `data/raw/` | Periodicidade | Período |
|---|---|---|---|
| Restrição de geração eólica | `constrained_off_eolico` | mensal | 2021-10 a 2026-09 |
| Curva de carga | `curva_carga` | anual | 2021 a 2026 |
| Intercâmbio entre subsistemas | `intercambio_subsistemas` | anual | 2021 a 2026 |

**Adicionar um dataset:** inclua uma linha `Dataset(...)` em `DATASETS`, informando pasta, caminho no bucket, padrão do nome do arquivo, período e se é mensal. Não é preciso escrever função nova.

Arquivos ausentes (404) geram aviso e são contados como falha no log final de cada dataset.

## 2. `process_ons.py`

### Etapas

1. **Eólico:** lê cada CSV mensal em chunks e agrega 30 min por entidade, depois 30 min por estado e por fim hora por estado.
2. **Carga:** carga horária por subsistema.
3. **Intercâmbio:** uma coluna por par origem_destino (`interchange_verified_<origem>_<destino>_MWmed`), em grade horária contínua. Hora sem registro do par vale 0.
4. **Grade:** grade contínua Estado × Hora. `GerRenEOL_MW` ausente **não** é preenchido, o que permite distinguir ausência de registro de geração zero.
5. **Features:** variáveis de tempo cíclicas e, para `GerRenEOL_MW`, lags, estatísticas de janela móvel e variações. As janelas usam `shift(1)`, então não há vazamento do instante atual.
6. **Sequências:** janelas de 168 h por estado, gravadas em arrays `.npy` mapeados em memória.

### Configuração (topo do arquivo)

| Constante | Função |
|---|---|
| `SUBSYSTEM` | `id_subsistema` mantido; as demais linhas são descartadas (padrão `"NE"`) |
| `STATE_MERGE` | mapa `id_estado` original -> id do grupo (padrão `BA` e `SE` -> `BA_SE`) |
| `STATE_NAMES` | nome exibido do grupo (`Bahia+Sergipe`) |
| `SEQUENCE_LENGTH` | horas de histórico por amostra (168) |
| `LAGS`, `WINDOWS` | lags e janelas móveis das features de geração |
| `CHUNKSIZE` | linhas por chunk na leitura dos CSVs eólicos |

> Os valores `"NE"`, `"BA"` e `"SE"` devem ser conferidos contra os CSVs reais (`df["id_subsistema"].unique()`, `df["id_estado"].unique()`). Se não baterem, o filtro ou a junção não acontecem e o script não gera erro.

### Como o pipeline se estende

As agregações são declarativas (`ENTITY_SPEC`, `STATE_SPEC`, `MERGE_SPEC`, `HOURLY_SPEC`, no formato `saída=(coluna, função)`). Para criar uma métrica nova, edite a linha correspondente. Para unir outros estados, acrescente entradas em `STATE_MERGE` e `STATE_NAMES`.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `data/processed/state_hourly.parquet` | dataset horário por estado, com features e alvos |
| `data/training/feature_columns.json` | lista ordenada das features de `X` |
| `data/training/state_mapping.json` | `id_estado`, `nom_estado`, `state_idx` |
| `data/training/coverage_report.csv` | horas completas e valores ausentes por estado |
| `data/training/lstm_cnn/X.npy` | `[amostras, 168, features]`, `float32` |
| `data/training/lstm_cnn/y_curtailment_flag.npy` | ocorrência de curtailment na hora (0/1) |
| `data/training/lstm_cnn/y_curtailment_mwmed.npy` | curtailment em MWmed |
| `data/training/lstm_cnn/y_curtailment_mwh.npy` | curtailment em MWh (igual ao MWmed em 1 h) |
| `data/training/lstm_cnn/y_curtailment_minutes.npy` | minutos de restrição |
| `data/training/lstm_cnn/y_reason.npy` | `[amostras, 4]`, razões REL, CNF, ENE, PAR |
| `data/training/lstm_cnn/state_idx.npy` | índice do estado, para embedding |
| `data/training/lstm_cnn/end_datetime.npy` | instante final da janela (`int64`, ns) |

### Features de entrada

- `GerRenEOL_*`: geração de referência, lags, médias/desvios/máximos móveis, variações absolutas e percentuais.
- `load_subsystem_MWmed`: carga do subsistema.
- `interchange_verified_*`: intercâmbio verificado por par de subsistemas.
- `hour`, `dow` e `doy` em seno e cosseno, mais `is_weekend`.

Ficam **fora** das features: alvos de curtailment, razões, origens e intercâmbio programado.

### Regras de validade das amostras

- Uma janela exige 168 horas consecutivas, sem lacunas, com features completas e com as duas meias-horas eólicas presentes (`complete_eolic_hour == 1`).
- Só o instante final da janela precisa ter alvo válido. Alvos ausentes no meio do histórico não descartam a amostra.
- As linhas do `X` correspondem posição a posição às dos demais `y_*.npy`, `state_idx.npy` e `end_datetime.npy`.

### Observações

- Bahia e Sergipe unidos: geração de referência, GNR e minutos são somados por instante. As flags de razão e origem usam máximo (1 se qualquer um teve a ocorrência).
- Carga e intercâmbio não são filtrados por subsistema na leitura. A carga entra pelo subsistema do estado, e o intercâmbio é uma variável do sistema inteiro.
- Os arquivos `.npy` são gravados com `open_memmap` e podem ser lidos com `np.load(path, mmap_mode="r")`.