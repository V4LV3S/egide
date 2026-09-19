from glob import glob
from pathlib import Path

from eccodes import codes_get, codes_grib_new_from_file, codes_release, codes_write

basepath = Path(__file__).resolve().parents[1] 
era5land = glob(f"{basepath}/data/raw/era5-land/*.grib", recursive=True)

era5land = sorted(era5land)


def split_grib_by_variable(input_file: Path, output_dir: Path, date: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}

    try:
        with input_file.open("rb") as source:
            while handle := codes_grib_new_from_file(source):
                try:
                    short_name = codes_get(handle, "shortName")
                    output_file = output_files.get(short_name)
                    if output_file is None:
                        output_path = output_dir / f"{short_name}_{date}.grib"
                        output_file = output_path.open("wb")
                        output_files[short_name] = output_file
                    codes_write(handle, output_file)
                finally:
                    codes_release(handle)
    finally:
        for output_file in output_files.values():
            output_file.close()


for file in era5land:
    file_name = Path(file).stem
    date = file_name.split("hourly_")[-1]
    split_grib_by_variable(
        Path(file), basepath / "data/processed/meteoro", date
    )


