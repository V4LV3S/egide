"""Converte NetCDFs meteorológicos recortados em Parquet sem transformar dados.

Cada Parquet é uma representação tabular direta do respectivo NetCDF: há uma
linha por combinação das dimensões (normalmente time, latitude e longitude),
e as coordenadas e variáveis do Dataset tornam-se colunas. Não há filtragem,
ordenação, agregação, imputação, normalização ou alteração de unidades.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

# CONFIGURAÇÃO EDITÁVEL PARA EXECUÇÃO DIRETA PELA IDE.
INPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro-recortado"
OUTPUT_DIRECTORY = ROOT / "ml" / "data" / "meteoro-raw"
REGIONS_TO_CONVERT: tuple[str, ...] | None = None
OVERWRITE_EXISTING_OUTPUTS = False


def selected_regions(input_directory: Path) -> list[Path]:
    """Retorna as regiões configuradas e valida os diretórios de origem."""
    if REGIONS_TO_CONVERT is None:
        regions = sorted(path for path in input_directory.iterdir() if path.is_dir())
    else:
        regions = [input_directory / region for region in REGIONS_TO_CONVERT]

    missing = [str(path) for path in regions if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Regiões não encontradas: {', '.join(missing)}")
    return regions


def netcdf_to_dataframe(source: Path) -> pd.DataFrame:
    """Lê um NetCDF e materializa sua representação tabular sem alterações."""
    with xr.open_dataset(source) as dataset:
        if not dataset.data_vars:
            raise ValueError(f"{source} não contém variáveis de dados.")
        return dataset.to_dataframe().reset_index()


def convert_file(source: Path, destination: Path) -> bool:
    """Converte um NetCDF para Parquet e retorna se uma saída foi gravada."""
    if destination.exists() and not OVERWRITE_EXISTING_OUTPUTS:
        print(f"Ignorando saída existente: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    netcdf_to_dataframe(source).to_parquet(destination, index=False)
    print(f"{source} -> {destination}")
    return True


def convert_region(region_directory: Path, output_directory: Path) -> list[Path]:
    """Converte todos os NetCDFs de uma região para ``ml/data/meteoro-raw/<região>``."""
    sources = sorted(region_directory.glob("*.nc"))
    if not sources:
        raise FileNotFoundError(f"Nenhum NetCDF encontrado em {region_directory}.")

    destinations: list[Path] = []
    for source in sources:
        destination = output_directory / region_directory.name / f"{source.stem}.parquet"
        if convert_file(source, destination):
            destinations.append(destination)
    return destinations


def convert_all_regions(input_directory: Path, output_directory: Path) -> list[Path]:
    """Converte os NetCDFs de todas as regiões selecionadas."""
    destinations: list[Path] = []
    for region_directory in selected_regions(input_directory):
        destinations.extend(convert_region(region_directory, output_directory))
    return destinations


def main() -> None:
    """Executa a conversão com a configuração editável deste módulo."""
    convert_all_regions(INPUT_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
