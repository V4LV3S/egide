"""Separa os GRIBs mensais do ERA5 em um arquivo por variável disponível."""

from pathlib import Path

from eccodes import codes_get, codes_grib_new_from_file, codes_release, codes_write


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECTORY = ROOT / "data" / "raw" / "era5"
OUTPUT_DIRECTORY = ROOT / "data" / "processed" / "meteoro"


def split_grib_by_variable(input_file: Path, output_dir: Path, date: str) -> None:
    """Grava cada variável existente no GRIB em seu próprio arquivo mensal."""
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


def main() -> None:
    for input_file in sorted(INPUT_DIRECTORY.glob("*.grib")):
        split_grib_by_variable(
            input_file,
            OUTPUT_DIRECTORY,
            input_file.stem.rsplit("hourly_", maxsplit=1)[-1],
        )


if __name__ == "__main__":
    main()
