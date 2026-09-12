# Egide

Egide is an open-source project scaffold.

## Status

Initial project structure. Replace this section with the project's purpose, goals, and setup steps as the implementation takes shape.

## Getting Started

Clone the repository:

```sh
git clone https://github.com/YOUR-USERNAME/egide.git
cd egide
```

Add project-specific setup instructions here.

## ERA5 Download

The script at `hackathon_content/ERA5/download_era5.py` downloads ERA5-Land
monthly means with the Copernicus CDS API. Before running it, create a CDS
account, accept the dataset terms, and configure your credentials in
`~/.cdsapirc`.

Edit the lists at the top of the script:

```python
VARIABLES = ["2m_temperature", "total_precipitation"]
YEARS = ["2025"]
MONTHS = ["01", "02", "03"]
AREA = [6, -74, -34, -34]  # north, west, south, east
```

Then run:

```sh
uv run python hackathon_content/ERA5/download_era5.py
```

## ONS Load Download

The script `download_ons_carga.py` downloads verified and scheduled load data
from the ONS API. Its defaults request the `NE` subsystem from `01/01/2023`
through the current date in Brasilia time.

Edit `ENDPOINTS`, `LOAD_AREA`, `START_DATE_BR`, or `END_DATE_BR` at the top of
the script, then run:

```sh
uv run python download_ons_carga.py
```

The results are saved as `processed_data/ONS/cargaverificada_NE.parquet` and
`processed_data/ONS/cargaprogramada_NE.parquet`. The original UTC timestamp is
kept in `din_referenciautc`, with its Brasilia-time equivalent in
`din_referenciabr`.

## Access

This repository is public for viewing. Issue and request posting is intended for selected collaborators only.

## License

This project is licensed under the [MIT License](LICENSE).
