"""Download hourly ERA5 and ERA5-Land data with the CDS API.

Edit the configuration values below and run the script.

Set ``DATASETS_TO_DOWNLOAD`` to choose which sources will be downloaded.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cdsapi


# Keep the variables from each CDS dataset in separate lists.
ERA5_VARIABLES = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_cloud_cover",
]

ERA5_LAND_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_pressure",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]

# Keep both values to download both datasets, or remove one of them.
DATASETS_TO_DOWNLOAD = ["era5-land"]
MAX_WORKERS = 4

# Inclusive range of months to download.
START_DATE = datetime(2023, 10, 1)
END_DATE = datetime(2026, 8, 31)
HOURS = [f"{hour:02d}:00" for hour in range(24)]

# Geographic subset: [north, west, south, east].
AREA = [6, -74, -34, -34]

OUTPUT_DIR = Path("data/raw")


@dataclass(frozen=True)
class DatasetConfig:
    cds_name: str
    variables: list[str]
    output_dir: Path
    file_prefix: str


DATASETS = {
    "era5": DatasetConfig(
        cds_name="reanalysis-era5-single-levels",
        variables=ERA5_VARIABLES,
        output_dir=OUTPUT_DIR / "era5",
        file_prefix="era5_hourly",
    ),
    "era5-land": DatasetConfig(
        cds_name="reanalysis-era5-land",
        variables=ERA5_LAND_VARIABLES,
        output_dir=OUTPUT_DIR / "era5-land",
        file_prefix="era5_land_hourly",
    ),
}



def iter_months(start_date: datetime, end_date: datetime):
    if start_date > end_date:
        raise ValueError("START_DATE must be before or equal to END_DATE")

    current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current <= last:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def download_month(config: DatasetConfig, reference_date: datetime) -> str:
    variables = [variable.strip() for variable in config.variables if variable.strip()]
    days_in_month = monthrange(reference_date.year, reference_date.month)[1]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [str(reference_date.year)],
        "month": [f"{reference_date.month:02d}"],
        "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
        "time": HOURS,
        "area": AREA,
        "data_format": "grib",
        "download_format": "unarchived",
    }
    output_file = (
        config.output_dir / f"{config.file_prefix}_{reference_date:%Y_%m}.grib"
    )

    if output_file.exists():
        return f"Skipping existing file: {output_file}"

    client = cdsapi.Client()
    client.retrieve(config.cds_name, request, str(output_file))
    return f"Downloaded {config.cds_name} to {output_file}"


def main() -> None:
    tasks = []

    for dataset_name in DATASETS_TO_DOWNLOAD:
        if dataset_name not in DATASETS:
            raise Exception(
                f"Unknown dataset {dataset_name!r}. Choose from: {', '.join(DATASETS)}"
            )

        config = DATASETS[dataset_name]
        if not any(variable.strip() for variable in config.variables):
            print(f"Skipping {config.cds_name}: no variables configured")
            continue

        tasks.extend(
            (config, reference_date)
            for reference_date in iter_months(START_DATE, END_DATE)
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_month, config, reference_date)
            for config, reference_date in tasks
        ]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
