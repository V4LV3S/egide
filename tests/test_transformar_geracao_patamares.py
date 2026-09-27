from pathlib import Path

import pandas as pd

from scripts.transformar_geracao_patamares import (
    processar_diretorio,
    salvar_geracao_horaria,
    transformar_geracao,
)


def dados_individuais() -> pd.DataFrame:
    """Cria dois pares semihorários de uma usina."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2025-01-01 00:30:00",
                    "2025-01-01 01:00:00",
                    "2025-01-01 01:30:00",
                    "2025-01-01 02:00:00",
                ]
            ),
            "nom_usina": ["USINA TESTE"] * 4,
            "ceg": ["CEG123"] * 4,
            "id_estado": ["BA"] * 4,
            "val_geracaoreferencia": [10.0, 20.0, 30.0, 50.0],
        }
    )


def test_transformar_geracao_agrupa_patamares_por_media() -> None:
    resultado = transformar_geracao(dados_individuais())

    assert resultado["time"].tolist() == list(
        pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 02:00:00"])
    )
    assert resultado["val_geracaoreferencia"].tolist() == [15.0, 40.0]
    assert resultado["n_amostras"].tolist() == [2, 2]
    assert resultado["hora_completa"].all()


def test_transformar_geracao_remove_amostra_duplicada() -> None:
    dados = dados_individuais()
    dados = pd.concat([dados, dados.iloc[[0]]], ignore_index=True)

    resultado = transformar_geracao(dados)

    assert resultado.loc[0, "val_geracaoreferencia"] == 15.0
    assert resultado.loc[0, "n_amostras"] == 2


def test_processar_diretorio_preserva_schema_individual(tmp_path: Path) -> None:
    entrada = tmp_path / "entrada"
    saida = tmp_path / "saida"
    entrada.mkdir()
    dados_individuais().to_parquet(entrada / "usina_teste.parquet", index=False)

    destinos = processar_diretorio(entrada, saida)
    resultado = pd.read_parquet(destinos[0])

    assert destinos == [saida / "usina_teste.parquet"]
    assert resultado.columns.tolist() == [
        "time",
        "nom_usina",
        "ceg",
        "id_estado",
        "val_geracaoreferencia",
    ]
    assert resultado["val_geracaoreferencia"].tolist() == [15.0, 40.0]


def test_salvar_geracao_horaria_descarta_horas_incompletas(tmp_path: Path) -> None:
    dados = dados_individuais()
    ultima_linha = dados.iloc[[-1]].copy()
    ultima_linha["time"] = pd.Timestamp("2025-01-01 02:30:00")
    dados = pd.concat([dados, ultima_linha], ignore_index=True)
    horario = transformar_geracao(dados)

    destino = salvar_geracao_horaria(horario, tmp_path / "usina.parquet")
    resultado = pd.read_parquet(destino)

    assert resultado["time"].max() == pd.Timestamp("2025-01-01 02:00:00")
    assert len(resultado) == 2
