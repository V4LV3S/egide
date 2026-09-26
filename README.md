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

O fluxo e dividido em `ml/scripts/pca_pipeline.py`, que concentra as etapas
reutilizaveis, e `ml/scripts/create_pca_features.py`, que contem a configuracao
editavel na IDE. Para cada regiao, as variaveis NetCDF configuradas sao
alinhadas nos instantes em comum e separadas cronologicamente em treino (70%),
validacao (15%) e teste (15%).

A sequencia evita vazamento de dados:

1. Para cada variavel, valores ausentes sao imputados e o primeiro
   `StandardScaler` e ajustado apenas no treino; validacao e teste recebem a
   mesma transformacao.
2. A PCA de cada variavel e ajustada somente no treino normalizado e projeta
   treino, validacao e teste.
3. As componentes de todas as variaveis sao concatenadas para formar
   `X_train`, `X_validation` e `X_test`.
4. Um segundo `StandardScaler` e ajustado em `X_train` e transforma as tres
   matrizes, resultando nas features finais do modelo.

`tp` recebe `log1p` antes dessas etapas; a transformacao e configurada por
`VARIABLES_WITH_LOG1P`. Edite tambem `PCA_COMPONENTS_BY_VARIABLE`,
`TRAIN_FRACTION` e `VALIDATION_FRACTION` diretamente em
`ml/scripts/create_pca_features.py` e execute o arquivo pela IDE.

Cada regiao gera um unico arquivo:

```text
ml/data/
`-- meteoro/
    `-- BA_SE/
        `-- pca_features.parquet
```

O Parquet contem `time`, `split` (`train`, `validation` ou `test`) e uma coluna
por componente PCA, como `t2m_pca_0` e `tp_pca_0`. Ele e a matriz meteorologica
final, pronta para ser unida aos alvos e demais features no fluxo do modelo.

## Documentacao

Os cadernos do desafio, instrucoes de acesso ao ERA5 e o notebook de referencia
estao em `docs/hackathon/`. O [estado atual do projeto e o pipeline de
dados](docs/estado-atual-do-projeto.md) consolidam os dados, transformacoes,
cobertura e pendencias mapeados no repositorio.

## Licenca

Este projeto e distribuido sob a [licenca MIT](LICENSE).
