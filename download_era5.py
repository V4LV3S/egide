"""Download ERA5-Land data with the Copernicus CDS API.

Edit the lists and values below, then run:

    uv run python hackathon_content/ERA5/download_era5.py
"""

from pathlib import Path

import cdsapi


# Choose the ERA5 variables here.
VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

# Choose years and months here.
YEARS = ["2025"]
MONTHS = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]

# Choose the area here: [north, west, south, east].
# This default is the same Brazil extent used in the notebook.
AREA = [6, -74, -34, -34]

# Choose the output file here.
OUTPUT_FILE = Path("hackathon_content/ERA5/era5_land_2025.nc")


dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": VARIABLES,
    "year": YEARS,
    "month": MONTHS,
    "time": ["00:00"],
    "area": AREA,
    "data_format": "netcdf",
    "download_format": "unarchived",
}


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()
client.retrieve(dataset, request, str(OUTPUT_FILE))

print(f"Downloaded {dataset} to {OUTPUT_FILE}")
