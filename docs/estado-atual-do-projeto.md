# Estado atual do projeto e pipeline de dados

> Retrato do repositório em 20 de setembro de 2026. As quantidades, tamanhos e
> intervalos abaixo foram lidos dos arquivos locais; não são apenas a intenção
> descrita nos notebooks.

## Objetivo e estágio atual

O Egide reúne dados do ONS, reanálises meteorológicas ERA5/ERA5-Land e malhas
geográficas para apoiar duas frentes ainda em construção:

1. previsão de geração renovável; e
2. previsão/análise de *constrained-off* (curtailment).

A etapa de aquisição e organização dos dados está substancialmente adiantada.
Há datasets prontos para exploração, um processo reproduzível para a parte
meteorológica e variáveis temporais prontas. Ainda não há um dataset analítico
único, junção espacial/temporal entre as fontes, engenharia de atributos de
domínio, treino de modelos ou avaliação versionada.

## Pipeline consolidado

```text
                    ┌──────────────────────────────────┐
                    │ CDS / Copernicus                 │
                    │ ERA5 e ERA5-Land, horários       │
                    └──────────────┬───────────────────┘
                                   │ download_era5.py
                                   v
                    data/raw/era5[-land]/*.grib
                    (não versionado)
                                   │ era5[_land]_data_handler.py
                                   v
                    data/processed/meteoro/
                    <variável>_<AAAA>_<MM>.grib
                                   │ consolidar_meteoro_netcdf.py
                    ┌──────────────┴─────────────────────────────┐
                    │ interpola ERA5 para a grade ERA5-Land      │
                    │ ordena e remove tempos duplicados          │
                    │ achata time + step do ERA5-Land            │
                    └──────────────┬─────────────────────────────┘
                                   v
                    data/processed/meteoro-total/<variável>.nc
                                   │
                     [pendente: alinhar, agregar e criar features]
                                   v
                              dataset de ML

ONS API de carga ──> notebooks/download_carga.ipynb ──>
                     data/processed/carga/*.parquet ──┐
                                                        │
ONS constrained-off ──> [rotina de aquisição não       │
                         presente no repositório] ──>
                     data/processed/coff/*.parquet ────┤
                                                        ├──> junção temporal/
IBGE: UF 2025 ──> filtro Nordeste em                    │     espacial pendente
notebooks/visual_shapes.ipynb ──>                       │
data/processed/shapes/estados_ne.* ─────────────────────┤
                                                        │
Calendário ──> ml/notebooks/date_var_creation.ipynb ──>
ml/data/date_features.nc ───────────────────────────────┘
```

## Inventário dos dados disponíveis

### Meteorologia

Fonte: CDS/Copernicus. A configuração em `scripts/download_era5.py` cobre
outubro de 2023 a agosto de 2026, área `[6, -74, -34, -34]` (norte, oeste,
sul, leste) e dados horários. No estado atual, `DATASETS_TO_DOWNLOAD` contém
somente `era5-land`; o mesmo script também suporta `era5` quando habilitado.

| Arquivo consolidado | Variável | Fonte | Unidade | Cobertura efetiva | Grade |
| --- | --- | --- | --- | --- | --- |
| `100u.nc` | componente zonal do vento a 100 m (`u100`) | ERA5 | m s⁻¹ | 2023-10-01 00:00 a 2026-08-31 23:00, 25.584 horas | 401 × 401 |
| `100v.nc` | componente meridional do vento a 100 m (`v100`) | ERA5 | m s⁻¹ | 2023-10-01 00:00 a 2026-08-31 23:00, 25.584 horas | 401 × 401 |
| `tcc.nc` | cobertura total de nuvens | ERA5 | 0–1 | 2023-10-01 00:00 a 2026-08-31 23:00, 25.584 horas | 401 × 401 |
| `2t.nc` | temperatura a 2 m (`t2m`) | ERA5-Land | K | 2023-09-30 01:00 a 2026-09-01 00:00, 25.608 horas | 401 × 401 |
| `sp.nc` | pressão de superfície | ERA5-Land | Pa | 2023-09-30 01:00 a 2026-09-01 00:00, 25.608 horas | 401 × 401 |
| `tp.nc` | precipitação total | ERA5-Land | m | 2023-09-30 01:00 a 2026-09-01 00:00, 25.608 horas | 401 × 401 |
| `ssr.nc` | radiação solar líquida de onda curta | ERA5-Land | J m⁻² | 2023-09-30 01:00 a 2026-09-01 00:00, 25.608 horas | 401 × 401 |
| `str.nc` | radiação térmica líquida de onda longa | ERA5-Land | J m⁻² | 2023-09-30 01:00 a 2026-09-01 00:00, 25.608 horas | 401 × 401 |

Todos esses arquivos têm latitude de -34 a 6 e longitude de -74 a -34. Os
campos ERA5 (`100u`, `100v`, `tcc`) são interpolados linearmente para a grade
de referência ERA5-Land antes da consolidação. O arquivo
`data/processed/meteoro/coords_era5land.nc` preserva essa grade de referência.

Os GRIBs ERA5-Land têm `time` e `step`; o consolidado usa `valid_time` como o
eixo temporal final. Por isso sua cobertura tem uma hora adicional no começo e
no fim em relação aos ERA5. Antes de cruzar as fontes, selecionar a interseção
`2023-10-01 00:00` a `2026-08-31 23:00`; não assumir que os oito NetCDFs têm
o mesmo índice apenas por terem o mesmo número de meses.

O diretório `data/processed/meteoro-total/` ocupa aproximadamente **135,4
GiB**. Ele contém os oito NetCDFs acima e, no instante deste levantamento,
dois compactados auxiliares (`100u.zip` e `2t.zip`). Os NetCDFs não são
ignorados pelo Git e estão como arquivos locais ainda não versionados: antes
de enviá-los à nuvem/repositório, definir a estratégia de armazenamento
(artefato externo, Git LFS ou dados regeneráveis).

### Carga elétrica

Fonte: API Carga Global do ONS. O notebook
`notebooks/download_carga.ipynb` produz Parquets por área e tipo de carga.
Estão disponíveis 15 arquivos, cerca de **30 MiB** no total, no formato
`<área>_<tipo>.parquet`.

| Conjunto | Áreas disponíveis | Registros por área | Cobertura |
| --- | --- | --- | --- |
| `cargaverificada` | `ALPE`, `BAOE`, `BASE`, `CE`, `NE`, `PBRN`, `PI` | 66.384 | 2023-01-01 a 2026-09-01 |
| `cargaprogramada` | `ALPE`, `BAOE`, `BASE`, `CE`, `NE`, `PBRN`, `PI` | 66.288 | 2023-01-01 a 2026-09-01 |
| `cargaverificada` histórico | `SE` | 41.995 | 2023-01-01 a 2025-04-30 |

O identificador temporal para junção é `din_referenciautc` (UTC). Os arquivos
verificados trazem também atualização, carga supervisionada/não supervisionada,
MMGD e indicador de consistência; os programados trazem
`val_cargaglobalprogramada`. Há uma ressalva já documentada em
`data/processed/carga/README.md`: `SE` não é o código atual documentado pelo
ONS para o subsistema Sudeste/Centro-Oeste (`SECO`). Não renomeá-lo ou
combiná-lo sem validar a origem.

### Constrained-off

Fonte: ONS. Os Parquets ficam em `data/processed/coff/` e ocupam cerca de
**0,69 GiB**. Cada registro representa 30 minutos e as potências estão em
MWmed. A semântica de geração, limite, referência, disponibilidade, razão e
origem da restrição está em [constrained_off.md](constrained_off.md).

| Arquivo | Granularidade | Registros | Cobertura |
| --- | --- | ---: | --- |
| `constrained_off_eolica_detail.parquet` | usina/modalidade; vento e geração | 63.429.221 | 2023-01 a 2026-08 |
| `constrained_off_eolica_tm.parquet` | usina; restrição e geração | 7.951.920 | 2023-10 a 2026-08 |
| `constrained_off_fotovoltaica_detail.parquet` | usina/modalidade; irradiância e geração | 19.061.960 | 2024-04 a 2026-08 |
| `constrained_off_fotovoltaica_tm.parquet` | usina; restrição e geração | 2.854.800 | 2024-04 a 2026-08 |
| `constrained_off_eolica_fotovoltaica_tm.parquet` | união eólica + fotovoltaica | 9.441.168 | 2024-04 a 2026-08 |

O notebook `notebooks/visual_coff.ipynb` explora o conjunto eólico do
Nordeste. Os Parquets já carregam campos de linhagem (`fonte`, `arquivo_origem`)
e de particionamento (`ano`, `mes`, `data`, `hora`, `minuto`, `ano_mes`), mas a
rotina que baixa e consolida os dados constrained-off não está no repositório;
portanto este trecho ainda não é reproduzível ponta a ponta.

### Geometria e atributos de calendário

* `data/raw/shapes/BR_UF_2025.*`: malha das 27 UFs brasileiras, CRS EPSG:4674.
* `data/processed/shapes/estados_ne.*`: recorte das nove UFs do Nordeste,
  derivado no notebook `notebooks/visual_shapes.ipynb`, também EPSG:4674.
* `ml/data/date_features.nc`: 25.584 timestamps horários de 2023-10-01
  00:00 a 2026-08-31 23:00, com oito atributos cíclicos: seno e cosseno de
  hora, dia da semana, dia do mês e mês. É a primeira feature de ML já
  materializada.

## O que foi implementado

O histórico Git registra, entre 11 e 19 de setembro de 2026:

* estrutura do projeto com `uv`, Git LFS para Parquet e diretórios de dados;
* consulta à API do ONS para carga e organização por área;
* inclusão dos conjuntos e documentação de constrained-off;
* organização inicial de módulos/notebooks de ML e recomendações de
  regularização em `TODO.md`;
* malha das UFs, recorte Nordeste e notebooks de visualização;
* download mensal de ERA5/ERA5-Land, separação de GRIB por variável e
  consolidação incremental em NetCDF;
* testes para a consolidação meteorológica: interpolação, ordenação/deduplicação
  temporal, achatamento `time` + `step` e intervalo de download.

Os notebooks atuais são de coleta, inspeção ou visualização: `download_carga`,
`visual_load`, `visual_coff`, `visual_era5`, `visual_meteoro`,
`visual_shapes` e `ml/notebooks/date_var_creation`. Eles não constituem ainda
um pipeline automático de treino ou inferência.

## Próxima camada: dataset de modelagem

O caminho recomendado para transformar os artefatos existentes em um conjunto
usável por modelo é:

1. Fixar uma grade temporal de referência — inicialmente a interseção horária
   2023-10-01 00:00 a 2026-08-31 23:00 em UTC.
2. Para alvos ONS de 30 minutos, escolher e documentar a regra de agregação
   para hora (por exemplo, média de MWmed) ou manter todos os preditores em
   30 minutos com interpolação/repetição explicitamente justificada.
3. Derivar `vent100 = sqrt(u100² + v100²)` depois de alinhar `100u` e `100v`;
   ela está prevista no `TODO.md`, mas não existe como arquivo/variável atual.
4. Converter unidades acumuladas quando necessário: `t2m` de K para °C;
   `tp` de m para mm; e analisar `ssr`/`str` como acumulados antes de calcular
   fluxos ou energia por intervalo.
5. Associar cada usina/estado a células da grade (ponto, máscara ou agregação
   espacial), usando identificadores `ceg`, `id_ons` e a malha; as coordenadas
   das usinas ainda não aparecem no repositório.
6. Definir alvos de curtailment, preferindo `val_geracaoreferenciafinal` quando
   disponível para apuração e documentando o tratamento de nulos. Para análise
   exploratória, o documento de constrained-off sugere
   `max(val_geracaoreferencia - val_geracaolimitada, 0)`.
7. Criar partições temporais de treino/validação/teste, sem embaralhamento e
   sem vazamento de futuro; versionar esquema, períodos, transformações e
   métricas junto do artefato do modelo.

## Pendências e riscos conhecidos

| Tema | Situação | Ação necessária |
| --- | --- | --- |
| Dataset final | Não existe | Implementar junção temporal, espacial e de alvos. |
| `vent100` | Pendente | Derivar de `100u` e `100v` após o alinhamento. |
| Alinhamento ERA5/ERA5-Land | Bordas temporais diferentes | Recortar pela interseção antes de combinar. |
| Frequência | Meteoro horário; constrained-off e carga em sub-hora | Definir uma frequência canônica e agregações. |
| Constrained-off | Dados disponíveis, ingestão ausente | Adicionar script/notebook reproduzível de aquisição e consolidação. |
| Localização das usinas | Não encontrada | Obter tabela de coordenadas para a associação à grade. |
| ML | Somente features de calendário e esqueletos planejados | Implementar baselines, treino, avaliação e persistência. |
| Documentação ML | `README.md` aponta para `ml/README.md`, inexistente | Criar o guia ou corrigir o link. |
| Armazenamento | NetCDFs grandes e locais | Definir checksum, manifesto, destino na nuvem e política de versionamento. |

## Reprodução rápida

Com o ambiente configurado e as credenciais CDS em `~/.cdsapirc`:

```sh
uv sync
uv run python scripts/download_era5.py
uv run python scripts/era5_data_handler.py
uv run python scripts/era5land_data_handler.py
uv run python scripts/consolidar_meteoro_netcdf.py
uv run pytest
```

Antes da consolidação, ajustar em `scripts/consolidar_meteoro_netcdf.py` quais
variáveis serão processadas e se arquivos existentes podem ser sobrescritos.
O processo grava primeiro um NetCDF temporário e requer espaço livre acima do
tamanho estimado da saída. A coleta de carga permanece no notebook; a de
constrained-off ainda precisa ser incorporada ao fluxo reproduzível.
