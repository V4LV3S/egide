# TODO

## Previsão Renovável

**Responsáveis:** Almeida e Daniel

- [ ] Montar um esqueleto das arquiteturas de ML com foco em paralelismo
- [ ] LSTM + CNN
- [ ] MLP básico
- [ ] Regressão linear regularizada (verificar)
- [ ] Verificar se será necessária outra arquitetura

**Saída:** 24h pelo modelo MLx + 48h MLx com ajuste de erros

## Previsão Curtailment

**Responsáveis:** Almeida e Daniel

- [ ] LSTM + CNN
- [ ] MLP básico
- [ ] Regressão linear regularizada (verificar)
- [ ] XGBoost
- [ ] Arquitetura de classificação

**Saída:** 24h pelo modelo MLy + 48h MLy com ajuste de erros

# Dados
**Responsáveis:** Almeida 

- [x] Variáveis meteorológicas (Almeida)


**ERA5** 
1. Criar a variável de intensidade do vento vent100 = raiz(v²+u²)
2. Montar os mesmos pontos de grade do ERA5-land



## Regularizar os dados

| Dado                    | Tratamento que faz mais sentido                             |
| ----------------------- | ----------------------------------------------------------- |
| **Geração eólica**      | \(P/P_{capacidade}\) → \([0,1]\)                            |
| **Geração solar**       | \(P/P_{capacidade}\) ou capacity factor                     |
| **Irradiância solar**   | **Clearness Index** \(GHI/GHI_{clear-sky}\)                 |
| **Temperatura**         | StandardScaler / Z-score                                    |
| **Pressão**             | StandardScaler / Z-score                                    |
| **Umidade**             | StandardScaler ou MinMax                                    |
| **Velocidade do vento** | StandardScaler, dependendo do modelo                        |
| **u10/v10/u100/v100**   | StandardScaler individual                                   |
| **Carga**               | Z-score ou Instance Normalization                           |
| **Hora/dia/mês**        | Melhor usar codificação cíclica que simplesmente normalizar |
