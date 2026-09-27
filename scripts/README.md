# Scripts de dados

Rotinas executáveis de aquisição e transformação anteriores ao treino. Execute
sempre a partir da raiz do repositório e revise as constantes no início de cada
arquivo.

## Fluxo meteorológico

```text
CDS -> data/raw
    -> separação/consolidação
    -> coarsening
    -> recorte por região ou usina
    -> data/processed
```

| Script | Função |
| --- | --- |
| `download_era5.py` | Baixa ERA5 e ERA5-Land pela API CDS. |
| `era5_data_handler.py` | Separa variáveis dos GRIBs ERA5. |
| `era5land_data_handler.py` | Separa variáveis dos GRIBs ERA5-Land. |
| `consolidar_meteoro_netcdf.py` | Consolida os arquivos mensais em NetCDF por variável. |
| `coarsen_meteoro_netcdf.py` | Reduz a resolução espacial. |
| `recortar_meteoro_netcdf.py` | Recorta os campos por região geoelétrica. |
| `recortar_meteoro_netcdf_usinas.py` | Seleciona os pontos próximos das usinas. |
| `transformar_precipitacao_era5.py` | Converte precipitação acumulada para mm por intervalo. |
| `transformar_radiacao_era5.py` | Converte radiação acumulada para fluxo por intervalo. |

## ONS e cadastro

| Script | Função |
| --- | --- |
| `download_cadastro_aneel.py` | Baixa o cadastro SIGA/ANEEL. |
| `transformar_carga_patamares_ons.py` | Agrega carga semihorária em valores horários para MMGD. |
| `transformar_geracao_patamares.py` | Agrega geração individual semihorária em valores horários. |

Exemplo:

```bash
uv run python scripts/consolidar_meteoro_netcdf.py
uv run python scripts/recortar_meteoro_netcdf.py
uv run python scripts/transformar_carga_patamares_ons.py
```

As etapas seguintes estão em [../ml/README.md](../ml/README.md).
