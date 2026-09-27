"""Baixa o cadastro de empreendimentos de geração do SIGA/ANEEL.

Execute na raiz do projeto:

    uv run python scripts/download_cadastro_aneel.py
"""
from __future__ import annotations

from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "cadastro" / "siga-empreendimentos-geracao.csv"
DOWNLOAD_URL = (
    "https://dadosabertos.aneel.gov.br/dataset/"
    "6d90b77c-c5f5-4d81-bdec-7bc619494bb9/resource/"
    "11ec447d-698d-4ab8-977f-b424d5deee6a/download/"
    "siga-empreendimentos-geracao.csv"
)
EXPECTED_HEADER = b"NomEmpreendimento"
CHUNK_SIZE = 1024 * 1024


def create_session() -> requests.Session:
    """Cria sessão HTTP com novas tentativas para falhas transitórias."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_aneel_registry(
    output_path: Path = OUTPUT_PATH,
    *,
    session: requests.Session | None = None,
) -> Path:
    """Baixa e valida o CSV, substituindo a saída somente após sucesso."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
    http_session = session or create_session()
    owns_session = session is None

    try:
        with http_session.get(
            DOWNLOAD_URL,
            stream=True,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()
            first_chunk = True
            bytes_written = 0
            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    if first_chunk:
                        if EXPECTED_HEADER not in chunk:
                            raise ValueError(
                                "Resposta da ANEEL não contém o cabeçalho esperado."
                            )
                        first_chunk = False
                    output_file.write(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                raise ValueError("A ANEEL retornou um arquivo vazio.")
            temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_session:
            http_session.close()

    return output_path


def main() -> None:
    """Baixa o cadastro para o diretório configurado."""
    output_path = download_aneel_registry()
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(f"Cadastro ANEEL salvo em {output_path} ({size_mib:.1f} MiB).")


if __name__ == "__main__":
    main()
