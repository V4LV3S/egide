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
3. Alinhar o eixo temporal dos NetCDFs ERA5 e ERA5-Land antes da junção: os
   arquivos ERA5-Land atuais incluem uma hora adicional em cada borda
   (`2023-09-30 01:00` a `2026-09-01 00:00`), enquanto ERA5 cobre
   `2023-10-01 00:00` a `2026-08-31 23:00`.



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
