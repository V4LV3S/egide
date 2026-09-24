import numpy as np
from datetime import datetime

import pytest

from scripts.transformar_precipitacao_era5 import (
    interval_precipitation_m_to_mm,
    validate_source_unit,
)


def test_precipitation_conversion_uses_00_as_closure_and_01_as_reset() -> None:
    timestamps = [
        datetime(2024, 1, 1, 23),
        datetime(2024, 1, 2, 0),
        datetime(2024, 1, 2, 1),
        datetime(2024, 1, 2, 2),
    ]
    accumulated = np.array([[0.100], [0.120], [0.010], [0.025]])

    result = interval_precipitation_m_to_mm(accumulated, timestamps)

    np.testing.assert_allclose(result[1:], [[20.0], [10.0], [15.0]])
    assert np.isnan(result[0, 0])


def test_precipitation_conversion_clips_negative_numerical_differences() -> None:
    timestamps = [datetime(2024, 1, 1, 23), datetime(2024, 1, 2, 0)]
    accumulated = np.array([[0.120], [0.119]])

    result = interval_precipitation_m_to_mm(accumulated, timestamps)

    assert result[1, 0] == 0.0


def test_precipitation_conversion_rejects_an_already_converted_unit() -> None:
    precipitation = type("Precipitation", (), {"units": "mm"})()

    with pytest.raises(ValueError, match="units='m'"):
        validate_source_unit(precipitation)
