"""Consolida a carga ONS semihorária em intervalos horários por média."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BRASILIA = "America/Sao_Paulo"
AMOSTRAS_POR_HORA = 2

# Configuração editável para execução direta pela IDE.
INPUT_DIRECTORY = Path("data/processed/carga")
OUTPUT_DIRECTORY = Path("data/processed/carga/horaria")
SUM_BAOE_AND_BASE = True
SAVE_MMGD_ONLY = True


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
    """Calcula a média de cada par de amostras semihorárias do ONS.

    A saída fica indexada pelo fim do intervalo horário: por exemplo, o valor
    de ``2025-01-01 00:00`` representa a média de 23:30 e 00:00 locais.
    """
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
    """Soma as medidas horárias de BAOE e BASE, formando a área BA_SE.

    A soma é aplicada exclusivamente às colunas ``val_*``. Metadados do
    intervalo não são somados: ``n_amostras`` permanece a quantidade de
    medições por área, e ``hora_completa`` só é verdadeira quando ambos os
    arquivos têm a hora completa.
    """
    chaves = ["din_referenciautc", "din_referenciabrasilia"]
    for nome, dados in (("BAOE", baoe), ("BASE", base)):
        ausentes = set(chaves) - set(dados.columns)
        if ausentes:
            raise ValueError(f"{nome} sem colunas de horário: {sorted(ausentes)}")

    valores_baoe = {coluna for coluna in baoe if coluna.startswith("val_")}
    valores_base = {coluna for coluna in base if coluna.startswith("val_")}
    if valores_baoe != valores_base:
        raise ValueError("BAOE e BASE devem ter as mesmas colunas 'val_'.")
    valores = sorted(valores_baoe)
    if not valores:
        raise ValueError("Não há colunas de medida 'val_' para somar.")

    colunas_baoe = [*chaves, "n_amostras", "hora_completa", *valores]
    colunas_base = [*chaves, "n_amostras", "hora_completa", *valores]
    unidos = baoe[colunas_baoe].merge(
        base[colunas_base],
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


def tipo_carga(caminho: Path) -> str:
    """Extrai o tipo de carga de ``<area>_<tipo>.parquet``."""
    partes = caminho.stem.split("_", maxsplit=1)
    if len(partes) != 2:
        raise ValueError(f"Nome de arquivo inválido: {caminho.name}")
    return partes[1]


def consolidar_baoe_e_base(diretorio: Path) -> list[Path]:
    """Lê BAOE e BASE horários e grava um Parquet BA_SE para cada tipo comum."""
    destinos: list[Path] = []
    for fonte_baoe in sorted(diretorio.glob("BAOE_*.parquet")):
        tipo = tipo_carga(fonte_baoe)
        fonte_base = diretorio / f"BASE_{tipo}.parquet"
        if not fonte_base.exists():
            raise FileNotFoundError(f"Arquivo BASE correspondente não encontrado: {fonte_base}")
        resultado = somar_baoe_e_base(
            pd.read_parquet(fonte_baoe), pd.read_parquet(fonte_base)
        )
        resultado.insert(1, "tipo_carga", tipo)
        destino = diretorio / f"BA_SE_{tipo}.parquet"
        resultado.to_parquet(destino, index=False)
        print(f"{fonte_baoe.name} + {fonte_base.name} -> {destino.name}")
        destinos.append(destino)
    return destinos


def salvar_base_mmgd_horaria(diretorio: Path) -> list[Path]:
    """Salva versões enxutas com data/hora de Brasília e carga MMGD."""
    destino_diretorio = diretorio / "mmgd"
    destinos: list[Path] = []
    for fonte in sorted(diretorio.glob("*.parquet")):
        dados = pd.read_parquet(fonte)
        if "val_cargammgd" not in dados.columns:
            continue
        destino_diretorio.mkdir(parents=True, exist_ok=True)
        destino = destino_diretorio / fonte.name
        dados[["din_referenciabrasilia", "val_cargammgd"]].to_parquet(destino, index=False)
        print(f"{fonte.name}: data/hora + val_cargammgd -> {destino}")
        destinos.append(destino)
    return destinos


def processar_diretorio(entrada: Path, saida: Path) -> list[Path]:
    """Transforma todos os Parquets diretamente dentro do diretório de entrada."""
    fontes = sorted(entrada.glob("*.parquet"))
    if not fontes:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em {entrada}.")
    saida.mkdir(parents=True, exist_ok=True)
    destinos: list[Path] = []
    for fonte in fontes:
        resultado = transformar_carga(pd.read_parquet(fonte))
        resultado.insert(1, "tipo_carga", tipo_carga(fonte))
        destino = saida / fonte.name
        resultado.to_parquet(destino, index=False)
        print(f"{fonte.name}: {len(resultado):,} horas -> {destino}")
        destinos.append(destino)
    if SUM_BAOE_AND_BASE:
        destinos.extend(consolidar_baoe_e_base(saida))
    if SAVE_MMGD_ONLY:
        destinos.extend(salvar_base_mmgd_horaria(saida))
    return destinos


def main() -> None:
    """Executa a transformação usando os diretórios configurados acima."""
    processar_diretorio(INPUT_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
