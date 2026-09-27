# Dados

Esta pasta reúne fontes de entrada e artefatos intermediários anteriores ao
pipeline de Machine Learning.

| Caminho | Conteúdo |
| --- | --- |
| `cadastro/` | Cadastro SIGA/ANEEL e pontos representativos de usinas. |
| `raw/` | Downloads brutos regeneráveis; o conteúdo é ignorado pelo Git. |
| `processed/` | Carga ONS, geração individual, constrained-off, meteorologia e malhas já tratadas. |

Os NetCDFs e GRIBs meteorológicos podem ocupar dezenas de gigabytes. Não edite
esses arquivos manualmente: use os scripts descritos em
[../scripts/README.md](../scripts/README.md).

Documentação específica:

- [Carga ONS](processed/carga/README.md)
- [Dados meteorológicos](processed/meteoro/README.md)
- [Semântica de constrained-off](../docs/constrained_off.md)

Os dados prontos para treino ficam separados em [../ml/data](../ml/data).
