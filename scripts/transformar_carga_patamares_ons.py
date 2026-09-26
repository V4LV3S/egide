"""Consolida a carga ONS semihorária em intervalos horários por média."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BRASILIA = "America/Sao_Paulo"
AMOSTRAS_POR_HORA = 2

# Configuração editável para execução direta pela IDE.
INPUT_DIRECTORY = Path("data/processed/carga")
OUTPUT_DIRECTORY = Path("ml/data/carga-horaria-mmgd")
SUM_BAOE_AND_BASE = True


def chaves_horarias(referencia_utc: pd.Series) -> pd.DataFrame:
    """Deriva a hora de término local a partir dos instantes semihorários UTC.

    A marca temporal da API é o fim do intervalo. Assim, a hora ``D 00:00``
    agrega as amostras de ``D-1 23:30`` e ``D 00:00`` em Brasília.
    """
    utc = pd.to_datetime(referencia_utc, utc=True, errors="raise")
    brasilia = utc.dt.tz_convert(BRASILIA)
    fim_hora_brasilia = brasilia.dt.ceil("h")
    return pd.DataFrame(
        {
            "din_referenciautc": fim_hora_brasilia.dt.tz_convert("UTC"),
            "din_referenciabrasilia": fim_hora_brasilia,
        },
        index=referencia_utc.index,
    )


def transformar_carga(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcula a média de cada par de amostras semihorárias do ONS."""
    obrigatorias = {"cod_areacarga", "din_referenciautc"}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")
    valores = [coluna for coluna in dados if coluna.startswith("val_")]
    if not valores:
        raise ValueError("Não há colunas de valor iniciadas por 'val_'.")

    # A coleta pode conter páginas sobrepostas; cada instante é uma só amostra.
    dados = dados.drop_duplicates(["cod_areacarga", "din_referenciautc"], keep="last")
    base = dados[["cod_areacarga", *valores]].join(chaves_horarias(dados["din_referenciautc"]))
    chaves = ["cod_areacarga", "din_referenciautc", "din_referenciabrasilia"]
    horario = (
        base.groupby(chaves, as_index=False, dropna=False)
        .agg(
            **{coluna: (coluna, "mean") for coluna in valores},
            n_amostras=(valores[0], "size"),
        )
        .sort_values(["cod_areacarga", "din_referenciautc"], ignore_index=True)
    )
    horario["hora_completa"] = horario["n_amostras"].eq(AMOSTRAS_POR_HORA)
    return horario[[*chaves, "n_amostras", "hora_completa", *valores]]


def somar_baoe_e_base(baoe: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Soma as medidas horárias de BAOE e BASE, formando a área BA_SE."""
    chaves = ["din_referenciautc", "din_referenciabrasilia"]
    valores_baoe = {coluna for coluna in baoe if coluna.startswith("val_")}
    valores_base = {coluna for coluna in base if coluna.startswith("val_")}
    if valores_baoe != valores_base:
        raise ValueError("BAOE e BASE devem ter as mesmas colunas 'val_'.")
    valores = sorted(valores_baoe)
    if not valores:
        raise ValueError("Não há colunas de medida 'val_' para somar.")

    colunas = [*chaves, "n_amostras", "hora_completa", *valores]
    unidos = baoe[colunas].merge(
        base[colunas],
        on=chaves,
        how="inner",
        validate="one_to_one",
        suffixes=("_baoe", "_base"),
    )
    if len(unidos) != len(baoe) or len(unidos) != len(base):
        raise ValueError("BAOE e BASE possuem intervalos horários sem par para soma.")

    resultado = unidos[chaves].copy()
    resultado.insert(0, "cod_areacarga", "BA_SE")
    resultado["n_amostras"] = unidos[["n_amostras_baoe", "n_amostras_base"]].min(axis=1)
    resultado["hora_completa"] = unidos["hora_completa_baoe"] & unidos["hora_completa_base"]
    for coluna in valores:
        resultado[coluna] = unidos[f"{coluna}_baoe"] + unidos[f"{coluna}_base"]
    return resultado[["cod_areacarga", *chaves, "n_amostras", "hora_completa", *valores]]


def remover_carga_base_diaria(dados: pd.DataFrame) -> pd.DataFrame:
    """Remove a carga-base diária a partir das seis horas nas extremidades.

    Para cada dia, a referência é a média de ``val_cargammgd`` nas três
    primeiras e três últimas horas disponíveis. Essa referência é subtraída de
    todas as horas do dia; resultados menores que 1,05% da referência viram
    zero.
    """
    obrigatorias = {"time", "val_cargammgd"}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")

    resultado = dados.copy()
    resultado["time"] = pd.to_datetime(resultado["time"], errors="raise")
    resultado = resultado.sort_values("time").copy()
    dias = resultado["time"].dt.normalize()
    posicao_no_dia = resultado.groupby(dias).cumcount()
    horas_no_dia = resultado.groupby(dias)["time"].transform("size")
    horas_de_referencia = (posicao_no_dia < 3) | (posicao_no_dia >= horas_no_dia - 3)
    referencia_por_dia = (
        resultado.loc[horas_de_referencia]
        .groupby(dias[horas_de_referencia])["val_cargammgd"]
        .mean()
    )
    referencia = dias.map(referencia_por_dia)

    resultado["val_cargammgd"] = resultado["val_cargammgd"] - referencia
    resultado.loc[
        resultado["val_cargammgd"] < referencia * 0.0105,
        "val_cargammgd",
    ] = 0
    return resultado


def salvar_bases_mmgd_horaria(
    bases_horarias: dict[str, pd.DataFrame], diretorio_saida: Path
) -> list[Path]:
    """Grava uma vez a base final: ``time`` e ``val_cargammgd``.

    O nome de cada arquivo identifica a área; por isso ``cod_areacarga`` não é
    repetido como coluna. Não existem Parquets horários intermediários.
    """
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    destinos: list[Path] = []
    for nome, dados in sorted(bases_horarias.items()):
        if "val_cargammgd" not in dados.columns:
            continue
        destino = diretorio_saida / nome
        saida = dados[["din_referenciabrasilia", "val_cargammgd"]].rename(
            columns={"din_referenciabrasilia": "time"}
        )
        saida = remover_carga_base_diaria(saida)
        saida["time"] = pd.to_datetime(saida["time"], errors="raise").dt.tz_localize(None)
        saida.to_parquet(destino, index=False)
        print(f"{nome}: time + val_cargammgd -> {destino}")
        destinos.append(destino)
    return destinos


def processar_diretorio(entrada: Path, saida: Path) -> list[Path]:
    """Cria em memória as bases horárias e salva somente a saída MMGD final."""
    fontes = sorted(entrada.glob("*cargaverificada.parquet"))
    if not fontes:
        raise FileNotFoundError(f"Nenhum Parquet de carga verificada encontrado em {entrada}.")

    bases_horarias = {
        fonte.name: transformar_carga(pd.read_parquet(fonte)) for fonte in fontes
    }
    if SUM_BAOE_AND_BASE:
        bases_horarias["BA_SE_cargaverificada.parquet"] = somar_baoe_e_base(
            bases_horarias["BAOE_cargaverificada.parquet"],
            bases_horarias["BASE_cargaverificada.parquet"],
        )
    return salvar_bases_mmgd_horaria(bases_horarias, saida)


def main() -> None:
    """Executa a transformação usando os diretórios configurados acima."""
    processar_diretorio(INPUT_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
