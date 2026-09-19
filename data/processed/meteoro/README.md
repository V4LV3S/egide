# Dados meteorológicos

Os arquivos seguem o padrão `<variavel>_<AAAA>_<MM>.grib`: cada arquivo contém
os dados horários da variável indicada para o mês correspondente.

| Variável | Fonte | Representa |
| --- | --- | --- |
| `100u` | ERA5 | Componente leste-oeste do vento a 100 m. |
| `100v` | ERA5 | Componente norte-sul do vento a 100 m. |
| `vent100` | ERA5 | Velocidade do vento a 100 m, calculada a partir de `100u` e `100v`. |
| `tcc` | ERA5 | Fração de cobertura total de nuvens. |
| `2t` | ERA5-Land | Temperatura do ar a 2 m. |
| `sp` | ERA5-Land | Pressão à superfície. |
| `tp` | ERA5-Land | Precipitação total acumulada. |
| `ssr` | ERA5-Land | Radiação solar líquida acumulada na superfície. |
| `str` | ERA5-Land | Radiação térmica líquida acumulada na superfície. |

`coords_era5land.nc` contém as coordenadas de latitude e longitude da grade
espacial do ERA5-Land.
