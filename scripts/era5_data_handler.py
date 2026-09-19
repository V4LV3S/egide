from glob import glob
from pathlib import Path

from eccodes import (
    codes_clone,
    codes_get,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
    codes_set,
    codes_set_values,
    codes_write,
)

basepath = Path(__file__).resolve().parents[1] 
era5 = glob(f"{basepath}/data/raw/era5/*.grib", recursive=True)

era5 = sorted(era5)


def split_grib_by_variable(input_file: Path, output_dir: Path, date: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    wind_components = {}

    def output_for(variable: str):
        output_file = output_files.get(variable)
        if output_file is None:
            output_path = output_dir / f"{variable}_{date}.grib"
            output_file = output_path.open("wb")
            output_files[variable] = output_file
        return output_file

    try:
        with input_file.open("rb") as source:
            while handle := codes_grib_new_from_file(source):
                try:
                    short_name = codes_get(handle, "shortName")
                    codes_write(handle, output_for(short_name))

                    if short_name in {"100u", "100v"}:
                        message_key = (
                            codes_get(handle, "dataDate"),
                            codes_get(handle, "dataTime"),
                            codes_get(handle, "step"),
                        )
                        previous = wind_components.pop(message_key, None)
                        if previous is None:
                            wind_components[message_key] = (
                                short_name,
                                codes_get_values(handle),
                                codes_clone(handle),
                            )
                        elif previous[0] != short_name:
                            _, previous_values, wind_template = previous
                            wind_speed = (previous_values**2 + codes_get_values(handle)**2) ** 0.5
                            codes_set(wind_template, "paramId", 228249)
                            codes_set_values(wind_template, wind_speed)
                            codes_write(wind_template, output_for("vent100"))
                            codes_release(wind_template)
                        else:
                            codes_release(previous[2])
                            wind_components[message_key] = (
                                short_name,
                                codes_get_values(handle),
                                codes_clone(handle),
                            )
                finally:
                    codes_release(handle)
    finally:
        for _, _, wind_template in wind_components.values():
            codes_release(wind_template)
        for output_file in output_files.values():
            output_file.close()


for file in era5:
    file_name = Path(file).stem
    date = file_name.split("hourly_")[-1]
    split_grib_by_variable(
        Path(file), basepath / "data/processed/meteoro", date
    )