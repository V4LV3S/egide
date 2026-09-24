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

Ao executar `scripts/recortar_meteoro_netcdf.py`, as transformações físicas
ocorrem antes do recorte: `tp` deixa de ser acumulado e passa de m para mm por
intervalo; `ssr`/`str` passam de acumulados em J m-2 para fluxos médios por
intervalo em W m-2. Internamente, o script usa um eixo auxiliar `time - 1
minuto`: assim 00:00 fecha o ciclo anterior e 01:00 inicia o novo ciclo
acumulado. Os arquivos regionais recém-gravados já contêm essas unidades.
`tp` e `ssr` são limitados a zero caso apareça uma pequena diferença negativa
numérica; `str` mantém seu sinal por ser radiação térmica líquida.

## Radiação por intervalo

`ssr` e `str` são acumulados desde 00:00 UTC em cada ciclo diário do
ERA5-Land; por isso, não devem ser usados diretamente como valores horários.
O script `scripts/transformar_radiacao_era5.py` processa todos os `ssr.nc` e
`str.nc` disponíveis em `data/processed/meteoro-recortado/` e cria cópias em
`data/processed/meteoro-recortado-intervalo/`, mantendo a mesma estrutura de
regiões e preservando os arquivos originais. Por padrão, a saída é o fluxo
médio de cada intervalo em `W m-2`; defina
`CONVERT_TO_WATTS_PER_SQUARE_METRE = False` no código para manter energia por
intervalo em `J m-2`.

## Precipitação em milímetros

O script `scripts/transformar_precipitacao_era5.py` converte todos os `tp.nc`
acumulados de `data/processed/meteoro-recortado/` para milímetros por
intervalo e grava as cópias em `data/processed/meteoro-recortado-mm/`, sem
alterar os originais. Como o recorte já aplica essa conversão, o script
independente só aceita arquivos ainda em metros e falha se receber uma saída já
convertida em mm.
