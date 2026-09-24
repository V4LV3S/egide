# Egide

Projeto para coleta, processamento e visualizacao de dados energeticos e
meteorologicos.

## Estrutura

```text
egide/
|-- data/
|   |-- raw/             # downloads locais, ignorados pelo Git
|   `-- processed/       # dados tratados em Parquet
|-- docs/hackathon/      # documentos e material de referencia
|-- notebooks/           # exploracao, processamento e visualizacao
|   `-- sandbox/         # experimentos temporarios
|-- ml/                  # notebooks, scripts e artefatos de Machine Learning
|   |-- notebooks/       # Ridge, MLP, CNN+LSTM e XGBoost
|   `-- scripts/         # preparacao, treino e persistencia reutilizaveis
|-- scripts/             # rotinas executaveis de coleta
|-- main.py
|-- pyproject.toml
`-- uv.lock
```

Execute os comandos a partir da raiz do repositorio para que os caminhos de
dados sejam resolvidos corretamente.

## Instalacao

```sh
git clone https://github.com/YOUR-USERNAME/egide.git
cd egide
uv sync
```

## Download ERA5

O script `scripts/download_era5.py` baixa medias mensais do ERA5-Land pela API
do Copernicus CDS. Antes de executa-lo, crie uma conta no CDS, aceite os termos
do conjunto de dados e configure as credenciais em `~/.cdsapirc`.

As variaveis, o periodo e a area geografica podem ser alterados no inicio do
script. Para executar:

```sh
uv run python scripts/download_era5.py
```

O arquivo resultante e salvo em `data/raw/era5/` e nao e versionado.

## Dados ONS

O notebook `notebooks/download_carga.ipynb` coleta dados de carga verificada e
programada da API do ONS. Os resultados sao gravados em
`data/processed/carga/`.

Os notebooks `notebooks/visual_load.ipynb` e `notebooks/visual_coff.ipynb`
concentram as analises e visualizacoes dos dados processados.

## Machine Learning

Os notebooks em `ml/notebooks/` oferecem esqueletos para regressao temporal,
regressao + classificacao e classificacao com Ridge/Logistic Regression, MLP,
CNN+LSTM e XGBoost. Consulte [ml/README.md](ml/README.md) para configuracao,
paralelismo manual e uso com Parquet ou CSV.

### Features meteorologicas com PCA

O fluxo abaixo processa todos os NetCDFs em cada pasta de
`data/processed/meteoro-recortado/` e grava uma unica saida por regiao em
`ml/data/{REGIAO}_pca.nc`. O escalonamento e a PCA sao ajustados apenas com os
70% iniciais da serie; os 15% seguintes sao rotulados como validacao e os 15%
finais como teste. O fluxo mantem somente os instantes presentes em todas as
variaveis da regiao e preenche lacunas pontuais pela mediana aprendida no treino.

Em cada grupo de variaveis, o script executa explicitamente: (1) carregamento
e alinhamento temporal, (2) separacao cronologica em treino, validacao e teste,
(3) imputacao e normalizacao ajustadas apenas no treino, (4) ajuste da PCA no
treino e projecao das tres particoes e (5) gravacao do NetCDF comprimido com os
rotulos da particao e metadados.

Por padrao, `tcc` nao recebe padronizacao: como ja e uma fracao entre 0 e 1,
ele passa apenas pela imputacao de lacunas antes da PCA. A tupla
`VARIABLES_WITHOUT_STANDARDIZATION` permite ajustar essa regra no codigo.

Para `tp`, o script aplica `log1p` aos valores em mm antes da imputacao e da
normalizacao: zero permanece zero, enquanto eventos de precipitacao muito altos
tem sua influencia reduzida. A tupla `VARIABLES_WITH_LOG1P` controla quais
variaveis recebem essa transformacao.

Abra `ml/scripts/create_pca_features.py` na IDE e altere
`PCA_CONFIGURATION` na secao `CONFIGURACAO EDITAVEL`. Ela define, para cada
dataset e variavel, a variancia explicada (valor entre 0 e 1) ou o numero fixo
de componentes. Por exemplo, `{"solar": {"ssr": 0.95, "tcc": 5}}` usa 95%
da variancia para `ssr` e cinco componentes para `tcc`. Cada chave externa
gera um arquivo `{REGIAO}_{DATASET}_pca.nc`; os nomes das variaveis devem ser
os internos dos NetCDFs, como `ws100`, `ssr`, `str` e `tcc`. Em seguida,
execute o arquivo pela propria IDE.

Por padrao, a saida e organizada para montagem manual do dataset de ML:

```text
ml/data/
`-- BA_SE/
    |-- wind/
    |   |-- wind_pca.nc
    |   |-- t2m_pca.parquet
    |   `-- tp_pca.parquet
    `-- solar/
        |-- solar_pca.nc
        |-- ssr_pca.parquet
        `-- tcc_pca.parquet
```

Cada Parquet possui `time`, `split` e uma coluna para cada componente PCA da
variavel. Assim, e possivel escolher variaveis, unir com `date_features.nc` e
incluir alvos manualmente. Defina `CREATE_COMBINED_DATASET = True` se tambem
quiser gerar a combinacao automatica em `data_features.nc` e Parquet.

O mesmo script tambem converte `ml/data/date_features.nc` em
`ml/data/date_features.parquet`. Esse arquivo compartilhado possui `time` e as
features de calendario, sem `split`, para que possa ser unido manualmente aos
Parquets de qualquer estado.

Exemplo de uso:

```python
import xarray as xr

with xr.open_dataset("ml/data/BA_SE/wind/wind_pca.nc") as wind:
    componentes_de_vento = wind.t2m_pca
```

```python
import pandas as pd

tp = pd.read_parquet("ml/data/BA_SE/wind/tp_pca.parquet")
X_tp = tp.drop(columns=["time", "split"])
```

## Documentacao

Os cadernos do desafio, instrucoes de acesso ao ERA5 e o notebook de referencia
estao em `docs/hackathon/`. O [estado atual do projeto e o pipeline de
dados](docs/estado-atual-do-projeto.md) consolidam os dados, transformacoes,
cobertura e pendencias mapeados no repositorio.

## Licenca

Este projeto e distribuido sob a [licenca MIT](LICENSE).
