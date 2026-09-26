import pandas as pd

from scripts.transformar_carga_patamares_ons import (
    salvar_bases_mmgd_horaria,
    somar_baoe_e_base,
    transformar_carga,
)


def test_primeira_hora_do_dia_agrega_2330_e_0000_locais() -> None:
    dados = pd.DataFrame({
        "cod_areacarga": ["NE", "NE"],
        "din_referenciautc": ["2025-01-01T02:30:00Z", "2025-01-01T03:00:00Z"],
        "val_cargaglobal": [100.0, 120.0],
    })

    linha = transformar_carga(dados).iloc[0]

    assert str(linha["din_referenciabrasilia"]) == "2025-01-01 00:00:00-03:00"
    assert linha["val_cargaglobal"] == 110.0
    assert linha["n_amostras"] == 2
    assert linha["hora_completa"]


def test_hora_seguinte_agrega_0030_e_0100_locais() -> None:
    dados = pd.DataFrame({
        "cod_areacarga": ["NE", "NE"],
        "din_referenciautc": ["2025-01-01T03:30:00Z", "2025-01-01T04:00:00Z"],
        "val_cargaglobal": [100.0, 120.0],
    })

    linha = transformar_carga(dados).iloc[0]

    assert str(linha["din_referenciabrasilia"]) == "2025-01-01 01:00:00-03:00"
    assert linha["val_cargaglobal"] == 110.0


def test_timestamps_duplicados_nao_duplicam_a_media() -> None:
    dados = pd.DataFrame({
        "cod_areacarga": ["NE", "NE", "NE"],
        "din_referenciautc": [
            "2025-01-01T03:30:00Z", "2025-01-01T03:30:00Z", "2025-01-01T04:00:00Z",
        ],
        "val_cargaglobal": [100.0, 100.0, 120.0],
    })

    linha = transformar_carga(dados).iloc[0]

    assert linha["n_amostras"] == 2
    assert linha["hora_completa"]


def test_hora_incompleta_e_sinalizada() -> None:
    dados = pd.DataFrame({
        "cod_areacarga": ["NE"],
        "din_referenciautc": ["2025-01-01T03:30:00Z"],
        "val_cargaglobal": [100.0],
    })

    linha = transformar_carga(dados).iloc[0]

    assert linha["n_amostras"] == 1
    assert not linha["hora_completa"]


def test_somar_baoe_e_base_soma_apenas_medidas() -> None:
    horarios = {
        "din_referenciautc": pd.to_datetime(["2025-01-01T03:00:00Z"], utc=True),
        "din_referenciabrasilia": pd.to_datetime(["2025-01-01T00:00:00-03:00"]),
        "n_amostras": [2],
        "hora_completa": [True],
    }
    baoe = pd.DataFrame({"cod_areacarga": ["BAOE"], **horarios, "val_cargaglobal": [100.0], "val_consistencia": [1.0]})
    base = pd.DataFrame({"cod_areacarga": ["BASE"], **horarios, "val_cargaglobal": [250.0], "val_consistencia": [2.0]})

    resultado = somar_baoe_e_base(baoe, base).iloc[0]

    assert resultado["cod_areacarga"] == "BA_SE"
    assert resultado["val_cargaglobal"] == 350.0
    assert resultado["val_consistencia"] == 3.0
    assert resultado["n_amostras"] == 2
    assert resultado["hora_completa"]


def test_salvar_bases_mmgd_horaria_grava_uma_unica_saida_final(tmp_path) -> None:
    dados = pd.DataFrame({
        "din_referenciabrasilia": pd.to_datetime(["2025-01-01T00:00:00-03:00"]),
        "val_cargammgd": [12.5],
        "val_cargaglobal": [300.0],
    })

    destinos = salvar_bases_mmgd_horaria(
        {"BA_SE_cargaverificada.parquet": dados}, tmp_path
    )
    resultado = pd.read_parquet(destinos[0])

    assert destinos == [tmp_path / "BA_SE_cargaverificada.parquet"]
    assert list(resultado.columns) == ["din_referenciabrasilia", "val_cargammgd"]
    assert resultado.iloc[0]["val_cargammgd"] == 12.5
