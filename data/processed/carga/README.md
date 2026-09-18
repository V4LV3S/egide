# Dados de carga do ONS

Os arquivos desta pasta seguem o padrão:

```text
<CODIGO_DA_AREA>_<TIPO_DE_CARGA>.parquet
```

Os tipos disponíveis são:

- `cargaverificada`: carga global efetivamente observada;
- `cargaprogramada`: carga global prevista na programação diária.

## Códigos das áreas

| Código | Localidade ou área |
| --- | --- |
| `SECO` | Subsistema Sudeste/Centro-Oeste |
| `S` | Subsistema Sul |
| `NE` | Subsistema Nordeste |
| `N` | Subsistema Norte |
| `RJ` | Rio de Janeiro |
| `SP` | São Paulo |
| `MG` | Minas Gerais |
| `ES` | Espírito Santo |
| `MT` | Mato Grosso |
| `MS` | Mato Grosso do Sul |
| `DF` | Distrito Federal |
| `GO` | Goiás |
| `AC` | Acre |
| `RO` | Rondônia |
| `PR` | Paraná |
| `SC` | Santa Catarina |
| `RS` | Rio Grande do Sul |
| `BASE` | Bahia/Sergipe |
| `BAOE` | Bahia Oeste |
| `ALPE` | Alagoas/Pernambuco |
| `PBRN` | Paraíba/Rio Grande do Norte |
| `CE` | Ceará |
| `PI` | Piauí |
| `TON` | Tocantins |
| `PA` | Pará |
| `MA` | Maranhão |
| `AP` | Amapá |
| `AM` | Amazonas |
| `RR` | Roraima |
| `PESE` | Perdas do Subsistema Sudeste/Centro-Oeste |
| `PES` | Perdas do Subsistema Sul |
| `PENE` | Perdas do Subsistema Nordeste |
| `PEN` | Perdas do Subsistema Norte |

> **Atenção:** `SE_cargaverificada.parquet` usa o código `SE`, que não consta
> na relação atual da API. O código documentado para o Subsistema
> Sudeste/Centro-Oeste é `SECO`; confirme a origem do arquivo antes de
> renomeá-lo ou combiná-lo com dados de `SECO`.

Fonte: [API Carga Global do ONS](http://ons-dl-prod-opendata-swagger.s3-website-us-east-1.amazonaws.com/).
