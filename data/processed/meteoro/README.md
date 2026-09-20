# Dados meteorológicos

Os arquivos seguem o padrão `<variavel>_<AAAA>_<MM>.grib`: cada arquivo contém
os dados horários da variável indicada para o mês correspondente.

| Variável | Fonte | Representa |
| --- | --- | --- |
| `100u` | ERA5 | Componente leste-oeste do vento a 100 m. |
| `100v` | ERA5 | Componente norte-sul do vento a 100 m. |
| `tcc` | ERA5 | Fração de cobertura total de nuvens. |
| `2t` | ERA5-Land | Temperatura do ar a 2 m. |
| `sp` | ERA5-Land | Pressão à superfície. |
| `tp` | ERA5-Land | Precipitação total acumulada. |
| `ssr` | ERA5-Land | Radiação solar líquida acumulada na superfície. |
| `str` | ERA5-Land | Radiação térmica líquida acumulada na superfície. |

`coords_era5land.nc` contém as coordenadas de latitude e longitude da grade
espacial do ERA5-Land.

O download está configurado para o período de outubro de 2023 a agosto de
2026, inclusive. Os separadores e a consolidação não aplicam recorte temporal:
eles mantêm todos os arquivos fornecidos. O separador preserva as variáveis
presentes nos GRIBs, sem calcular a variável derivada `vent100`.

## Conversão e consolidação dos GRIBs

Para arquivos mensais no padrão `<variavel>_<AAAA>_<MM>.grib`, execute o
arquivo `scripts/consolidar_meteoro_netcdf.py` diretamente pela IDE. Ajuste
`VARIABLES_TO_CONSOLIDATE` no início do arquivo para consolidar somente uma
lista específica; use `None` para todas as variáveis. Ajuste também
`OVERWRITE_EXISTING_OUTPUTS` para recriar NetCDFs já existentes.

O script lê os GRIBs com `cfgrib` e cria um único `<variavel>.nc` por
variável em `data/processed/meteoro-total/`. Antes de gravar, ordena `time` em
ordem crescente e remove instantes duplicados (mantendo a primeira ocorrência).
Para `100u`, `100v` e `tcc`, a interpolação linear para a grade indicada em
`coords_era5land.nc` é feita em cada arquivo mensal antes da concatenação.
Ele não remove os GRIBs de origem nem cria arquivos de índice `.idx`. Para
recriar arquivos consolidados existentes, defina `OVERWRITE_EXISTING_OUTPUTS`
como `True`. Ao terminar cada variável, o script fecha todos os arquivos abertos
e executa a coleta de lixo antes de iniciar a próxima.

Somente `100u`, `100v` e `tcc` recebem interpolação para a grade ERA5-Land.
As variáveis do ERA5-Land já usam essa grade e são consolidadas sem interpolação.
Para `2t`, `sp`, `tp`, `ssr` e `str`, as dimensões `time` e `step` do GRIB são
unificadas em uma única dimensão `time`, com os datetimes de `valid_time`.
