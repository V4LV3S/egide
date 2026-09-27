from pathlib import Path

import pytest

from scripts.download_cadastro_aneel import download_aneel_registry


class FakeResponse:
    """Resposta HTTP mínima para testar download sem rede."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield from self.chunks


class FakeSession:
    """Sessão que devolve resposta configurada."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.response = FakeResponse(chunks)

    def get(self, url: str, **kwargs) -> FakeResponse:
        assert url.startswith("https://dadosabertos.aneel.gov.br/")
        assert kwargs["stream"] is True
        return self.response


def test_download_aneel_registry_writes_csv_atomically(tmp_path: Path) -> None:
    output_path = tmp_path / "cadastro" / "siga.csv"
    session = FakeSession(
        [
            b"DatGeracaoConjuntoDados;NomEmpreendimento;CodCEG\n",
            b"2026-09-01;USINA TESTE;CEG123\n",
        ]
    )

    result = download_aneel_registry(output_path, session=session)  # type: ignore[arg-type]

    assert result == output_path
    assert output_path.read_bytes().endswith(b"USINA TESTE;CEG123\n")
    assert not output_path.with_suffix(".csv.part").exists()


def test_download_aneel_registry_rejects_unexpected_response(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "cadastro" / "siga.csv"
    session = FakeSession([b"<html>erro</html>"])

    with pytest.raises(ValueError, match="cabeçalho esperado"):
        download_aneel_registry(output_path, session=session)  # type: ignore[arg-type]

    assert not output_path.exists()
    assert not output_path.with_suffix(".csv.part").exists()
