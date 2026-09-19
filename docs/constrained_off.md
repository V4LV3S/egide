# Dados de constrained-off do ONS

Este conjunto descreve restrições operativas aplicadas pelo ONS a usinas eólicas
e fotovoltaicas. Cada registro corresponde a um intervalo de 30 minutos e os
valores de potência são expressos em **MWmed** (megawatt médio no intervalo).

## Como ler geração, limitação e corte

| Campo | Significado |
| --- | --- |
| `val_geracao` | **Geração verificada**: potência que a usina efetivamente injetou/produziu, medida pelo SCADA. |
| `val_geracaolimitada` | **Geração limitada**: teto de geração definido pelo ONS em tempo real. Não é o montante cortado. Valor nulo significa que não houve limitação; zero significa que o ONS solicitou injeção de potência ativa nula (usina/conjunto desligado). |
| `val_geracaoreferencia` | **Geração de referência**: estimativa de quanto a usina/conjunto poderia gerar se não houvesse limitação, calculada a partir do recurso disponível (vento ou irradiância) e das regras do ONS. |
| `val_disponibilidade` | **Disponibilidade eletromecânica verificada**: potência que a usina estava fisicamente/operacionalmente disponível para gerar. Pode ser menor que a potência instalada. |
| `val_geracaoreferenciafinal` | Referência ajustada pelas regras de apuração. Só é calculada para restrições de razão `REL` e é enviada à CCEE para a apuração de geração frustrada e ESS por constrained-off. |

Em resumo:

```text
geração de referência = potencial esperado sem restrição
geração limitada      = teto autorizado pelo ONS
geração verificada    = geração efetivamente realizada
```

## O que é geração cortada/frustrada?

Geração limitada é o **limite** solicitado pelo ONS. Geração cortada, frustrada
ou não realizada é a parte do potencial que deixou de ser produzida por causa
desse limite.

Exemplo em um intervalo de 30 minutos:

```text
Geração de referência: 150 MWmed
Geração limitada:      100 MWmed
Geração verificada:     94 MWmed
```

O corte associado à ordem de restrição é aproximadamente `150 - 100 = 50
MWmed`. A diferença `100 - 94 = 6 MWmed` **não é automaticamente corte**: ela
significa que a geração real ficou abaixo do teto e pode decorrer de recurso
menor que o esperado, indisponibilidade ou fatores operacionais da usina.

Para análise exploratória, uma estimativa simples da geração cortada é:

```text
max(val_geracaoreferencia - val_geracaolimitada, 0)
```

Para apuração oficial/financeira, deve-se usar `val_geracaoreferenciafinal`
quando disponível. Ela considera as regras do ONS, inclusive disponibilidade e
ajustes quando a geração verificada fica materialmente abaixo da limitada.

## Razão e origem da restrição

`cod_razaorestricao` indica a razão:

- `REL`: indisponibilidade externa (elétrica);
- `CNF`: atendimento a requisitos de confiabilidade;
- `ENE`: razão energética;
- `PAR`: restrição indicada no parecer de acesso.

`cod_origemrestricao` indica a abrangência:

- `LOC`: local;
- `SIS`: sistêmica.

## Unidade e conversão para energia

Os dados são apresentados em MWmed. Como cada registro representa 30 minutos,
o valor de energia do intervalo em MWh é:

```text
MWh = MWmed × 0,5
```

Por exemplo, 100 MWmed em uma linha correspondem a 50 MWh.

## Fonte

- [Dicionário de Dados — Restrição de Operação por Constrained-off de Usinas Fotovoltaicas (ONS)](https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/restricao_coff_fotovoltaica_tm/DicionarioDados_RestricaoContrainedoff_UsiFotovoltaica.pdf)
- [RO-AO.BR.13 — Apuração de Restrição de Operação por Constrained-off de Usinas Eolioelétricas e Usinas Fotovoltaicas (ONS)](https://www.ons.org.br/%2FMPO%2FDocumento%20Normativo%2F4.%20Rotinas%20Operacionais%20-%20SM%205.13%2F4.3.%20Rotinas%20P%C3%B3s-Opera%C3%A7%C3%A3o%2F4.3.2.%20Apura%C3%A7%C3%A3o%20de%20Dados%2FRO-AO.BR.13_Rev.08.pdf)
