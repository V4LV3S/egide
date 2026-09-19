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

## Documentacao

Os cadernos do desafio, instrucoes de acesso ao ERA5 e o notebook de referencia
estao em `docs/hackathon/`.

## Licenca

Este projeto e distribuido sob a [licenca MIT](LICENSE).
