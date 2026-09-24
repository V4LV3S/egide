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


# TODO — Pipeline Meteorológico

## 1. Preparar dados ERA5-Land

* [x] Carregar variáveis:

  * Temperatura
  * Pressão
  * `u100`
  * `v100`
  * Precipitação
  * `ssr`
* [x] Calcular velocidade do vento:

  * `wind_speed = sqrt(u100² + v100²)`
* [x] Remover `u100` e `v100` após o cálculo

## 2. Reduzir resolução espacial

* [x] Aplicar coarsening:

  * `0.1° → 0.3°`
  * média espacial em blocos `3 x 3`

## 3. Recortar por estado

* [x] Carregar shapes dos estados
* [x] Selecionar células/pontos `0.5°` dentro de cada estado
* [x] Manter cada célula espacial separada
* [x] Não calcular média única do estado

## 4. Criar features meteorológicas

Para cada célula espacial:

* [X] Temperatura
* [X] Pressão
* [X] Velocidade do vento
* [X] Precipitação
* [X] Radiação `ssr`

Formato esperado:

```text
T_1 ... T_n
P_1 ... P_n
V_1 ... V_n
Prec_1 ... Prec_n
SSR_1 ... SSR_n
```

## 5. Pré-processamento

* [X] Aplicar `log1p()` na precipitação
* [X] Manter demais variáveis sem transformação adicional

## 6. Separar dados

* [X] Dividir temporalmente:

  * Train
  * Validation
  * Test
* [X] Não usar shuffle

## 7. Padronização

* [X] Ajustar `StandardScaler` apenas no Train
* [X] Transformar Train, Validation e Test com o mesmo scaler
* [aval] Refazer o `StandardScaler` para os termos do parquet talvez seja necessario

## 8. PCA

* [ ] Ajustar PCA apenas no Train
* [ ] Usar:

  * `PCA(n_components=0.95)`
* [ ] Transformar Train, Validation e Test
* [ ] Salvar número de componentes encontrados

## 9. Variáveis temporais

Adicionar após a PCA:

* [X] `sin(hora)`
* [X] `cos(hora)`
* [X] `sin(dia_do_ano)`
* [X] `cos(dia_do_ano)`

## 10. Preparar entrada da ABiLSTM

* [ ] Criar janelas de 24 horas
* [ ] Formato de entrada:

```text
(samples, 24, features)
```

## 11. Preparar saída

Para previsão das próximas 24 horas:

```text
Y = [G(t+1), G(t+2), ..., G(t+24)]
```

* [ ] Formato:

```text
(samples, 24)
```

## 12. Treinar modelo

* [ ] Treinar ABiLSTM
* [ ] Avaliar em Validation
* [ ] Testar no conjunto Test

## Pipeline resumido

```text
ERA5-Land 0.1°
      ↓
Velocidade do vento
      ↓
Coarsening 0.5°
      ↓
Recorte por estado
      ↓
Features por célula
      ↓
log1p(precipitação)
      ↓
Train / Val / Test
      ↓
StandardScaler
      ↓
PCA 95 - 90%
      ↓
Features temporais
      ↓
Janelas de 24h
      ↓
ABiLSTM
      ↓
Previsão t+1 ... t+24
```
