# Égide

Pipeline de dados e Machine Learning para previsão de geração renovável e
volume de constrained-off no sistema elétrico brasileiro.

## Demo

- **Link da demo:** (se houver, ex: Vercel, Netlify, etc.)

## Estado atual

O projeto já possui:

- aquisição e tratamento de dados ONS, ERA5/ERA5-Land e SIGA/ANEEL;
- features meteorológicas regionais e por usina;
- datasets temporais de treino, validação e teste;
- modelos CNN-LSTM para previsão de 24 horas;
- ensembles para MMGD por região geoelétrica e geração por usina;
- exportação das previsões na escala original para ferramentas externas.

## Fluxo principal

```text
fontes ONS / CDS / ANEEL
        -> scripts/
        -> data/processed/
        -> ml/scripts/
        -> ml/data/
        -> ml/models/
        -> cnn_lstm_plot_results/
```

## Como rodar o projeto

Requisitos: Python 3.14 ou superior e [uv](https://docs.astral.sh/uv/).

```bash
git clone <url-do-repositorio>
cd egide
uv sync
uv run pytest
```

Execute os comandos a partir da raiz. As entradas, saídas e seleções de datasets
são configuradas no início de cada script.

## Operações CNN-LSTM

```bash
# Treinar
uv run python ml/scripts/cnn_lstm_training_mmgd.py
uv run python ml/scripts/cnn_lstm_training_indv.py

# Reutilizar checkpoints e exportar o teste
uv run python ml/scripts/cnn_lstm_export_mmgd.py
uv run python ml/scripts/cnn_lstm_export_indv.py

# Reunir JSONs para plots e limitar negativos a zero
uv run python ml/scripts/copy_cnn_lstm_plot_results.py
```

## Próximas melhorias

- investigar variáveis meteorológicas mais representativas para complementar a
  previsão das fontes intermitentes;
- organizar melhor as pastas por processo;
- aprimorar o fluxo e o tratamento dos dados, buscando o máximo desempenho
  computacional e menor consumo de memória;
- automatizar a seleção de variáveis, hiperparâmetros e arquiteturas dos modelos;
- adicionar métricas por horizonte de previsão, região e usina para facilitar
  a identificação de cenários com maior erro;
- avaliar previsões probabilísticas e intervalos de confiança, além das
  previsões pontuais;
- fortalecer a validação dos dados, a rastreabilidade dos experimentos e o
  monitoramento de degradação dos modelos;
- comparar a CNN-LSTM com modelos de referência e arquiteturas temporais mais
  recentes.

## Pastas

| Pasta | Resumo |
| --- | --- |
| [`data/`](data/README.md) | Fontes e dados processados. |
| [`scripts/`](scripts/README.md) | Aquisição e transformações gerais. |
| [`ml/`](ml/README.md) | Pipeline de features, treino, modelos e inferência. |
| [`cnn_lstm_plot_results/`](cnn_lstm_plot_results/README.md) | JSONs finais para plotagem. |
| [`notebooks/`](notebooks/README.md) | Exploração e visualização. |
| [`tests/`](tests/README.md) | Testes automatizados. |
| [`docs/`](docs/README.md) | Referências e documentação conceitual. |
| [`curtailment-model/`](curtailment-model/README.md) | Pipeline experimental de curtailment eólico. |

## Licença

Este projeto está sob a licença MIT — veja o arquivo [LICENSE](./LICENSE) para mais detalhes.
