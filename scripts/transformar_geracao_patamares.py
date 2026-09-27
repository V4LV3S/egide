"""Consolida geração individual semihorária em intervalos horários."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


AMOSTRAS_POR_HORA = 2
TIME_COLUMN = "time"
PLANT_COLUMNS = ("nom_usina", "ceg", "id_estado")

# Configuração editável para execução direta pela IDE.
INPUT_DIRECTORY = Path("data/processed/geracao_indv")
OUTPUT_DIRECTORY = Path("ml/data/geracao_horaria_indv")


def transformar_geracao(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcula média de cada par de patamares semihorários por usina.

    ``time`` já representa horário local. Como cada marca é o fim do
    intervalo, a hora ``D 01:00`` agrupa ``D 00:30`` e ``D 01:00``.
    """
    obrigatorias = {TIME_COLUMN, *PLANT_COLUMNS}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")

    valores = [coluna for coluna in dados if coluna.startswith("val_")]
    if not valores:
        raise ValueError("Não há colunas de valor iniciadas por 'val_'.")

    base = dados.copy()
    base[TIME_COLUMN] = pd.to_datetime(base[TIME_COLUMN], errors="raise")
    if base[TIME_COLUMN].isna().any():
        raise ValueError("A coluna 'time' possui instantes ausentes.")

    # Páginas ou extrações sobrepostas podem repetir um mesmo patamar.
    chaves_amostra = [*PLANT_COLUMNS, TIME_COLUMN]
    base = base.drop_duplicates(chaves_amostra, keep="last")
    base["fim_hora"] = base[TIME_COLUMN].dt.ceil("h")
    chaves_horarias = [*PLANT_COLUMNS, "fim_hora"]
    horario = (
        base.groupby(chaves_horarias, as_index=False, dropna=False)
        .agg(
            **{coluna: (coluna, "mean") for coluna in valores},
            n_amostras=(valores[0], "size"),
        )
        .rename(columns={"fim_hora": TIME_COLUMN})
        .sort_values(["nom_usina", TIME_COLUMN], ignore_index=True)
    )
    horario["hora_completa"] = horario["n_amostras"].eq(AMOSTRAS_POR_HORA)
    return horario[
        [TIME_COLUMN, *PLANT_COLUMNS, "n_amostras", "hora_completa", *valores]
    ]


def salvar_geracao_horaria(
    dados_horarios: pd.DataFrame,
    destino: Path,
) -> Path:
    """Salva somente horas completas, sem colunas auxiliares de qualidade."""
    valores = [coluna for coluna in dados_horarios if coluna.startswith("val_")]
    colunas = [TIME_COLUMN, *PLANT_COLUMNS, *valores]
    horas_completas = dados_horarios.loc[dados_horarios["hora_completa"]]
    destino.parent.mkdir(parents=True, exist_ok=True)
    horas_completas[colunas].to_parquet(destino, index=False)
    descartadas = len(dados_horarios) - len(horas_completas)
    print(
        f"{destino.name}: {len(horas_completas)} horas completas; "
        f"{descartadas} incompletas descartadas -> {destino}"
    )
    return destino


def processar_diretorio(entrada: Path, saida: Path) -> list[Path]:
    """Transforma todos os Parquets individuais e preserva seus nomes."""
    fontes = sorted(entrada.glob("*.parquet"))
    if not fontes:
        raise FileNotFoundError(f"Nenhum Parquet de geração encontrado em {entrada}.")

    destinos: list[Path] = []
    for fonte in fontes:
        horario = transformar_geracao(pd.read_parquet(fonte))
        destinos.append(salvar_geracao_horaria(horario, saida / fonte.name))
    return destinos


def main() -> None:
    """Executa transformação usando diretórios configurados no módulo."""
    processar_diretorio(INPUT_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
