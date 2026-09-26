from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = Path("data/raw")
BASE_URL = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """Descrição declarativa de um dataset ONS. Para adicionar um novo, basta instanciar."""
    name: str                      # pasta de saída
    slug: str                      # caminho no bucket S3
    pattern: str                   # ex.: "ARQ_{year}_{month:02d}.csv" ou "ARQ_{year}.csv"
    start: tuple[int, int]         # (ano, mês)
    end: tuple[int, int]           # (ano, mês)
    monthly: bool = False

    def periods(self) -> Iterator[tuple[int, int]]:
        (sy, sm), (ey, em) = self.start, self.end
        for year in range(sy, ey + 1):
            if not self.monthly:
                yield year, 1
                continue
            for month in range(sm if year == sy else 1, (em if year == ey else 12) + 1):
                yield year, month

    def files(self) -> Iterator[tuple[str, Path]]:
        for year, month in self.periods():
            filename = self.pattern.format(year=year, month=month)
            yield f"{BASE_URL}{self.slug}/{filename}", OUTPUT_DIR / self.name / str(year) / filename


DATASETS = (
    Dataset("constrained_off_eolico", "restricao_coff_eolica_tm",
            "RESTRICAO_COFF_EOLICA_{year}_{month:02d}.csv", (2021, 10), (2026, 9), monthly=True),
    Dataset("curva_carga", "curva-carga-ho", "CURVA_CARGA_{year}.csv", (2021, 1), (2026, 12)),
    Dataset("intercambio_subsistemas", "intercambio_nacional_ho",
            "INTERCAMBIO_NACIONAL_{year}.csv", (2021, 1), (2026, 12)),
)


def create_session() -> requests.Session:
    """Sessão HTTP com retry para erros transitórios."""
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(session: requests.Session, url: str, path: Path) -> bool:
    """Baixa o arquivo bruto. Retorna False se não existir ou falhar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.info("Já existe: %s", path)
        return True
    logger.info("Baixando: %s", url)
    try:
        with session.get(url, stream=True, timeout=(30, 300)) as resp:
            if resp.status_code == 404:
                logger.warning("Não encontrado: %s", url)
                return False
            resp.raise_for_status()
            with path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        logger.info("Salvo: %s", path)
        return True
    except requests.RequestException as exc:
        logger.error("Erro ao baixar %s: %s", url, exc)
        path.unlink(missing_ok=True)
        return False


def extract(session: requests.Session, dataset: Dataset) -> tuple[int, int]:
    """Extrai todos os arquivos de um dataset. Retorna (ok, falhas)."""
    results = [download_file(session, url, path) for url, path in dataset.files()]
    ok = sum(results)
    return ok, len(results) - ok


def main(datasets: tuple[Dataset, ...] = DATASETS) -> None:
    session = create_session()
    for ds in datasets:
        ok, failed = extract(session, ds)
        logger.info("%s: %d ok, %d falhas/ausentes", ds.name, ok, failed)
    logger.info("Extração finalizada.")


if __name__ == "__main__":
    main()
