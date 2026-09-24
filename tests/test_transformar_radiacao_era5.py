from datetime import datetime

import numpy as np

from scripts.transformar_radiacao_era5 import (
    interval_duration_seconds,
    interval_energy,
)


def test_interval_energy_removes_daily_accumulation_and_keeps_sign() -> None:
    timestamps = [
        datetime(2024, 1, 1, 23),
        datetime(2024, 1, 2, 0),
        datetime(2024, 1, 2, 1),
        datetime(2024, 1, 2, 2),
    ]
    accumulated = np.array([[-100.0], [-120.0], [-10.0], [-25.0]])

    result = interval_energy(accumulated, timestamps)

    np.testing.assert_allclose(result[1:], [[-20.0], [-10.0], [-15.0]])
    assert np.isnan(result[0, 0])


def test_interval_energy_auxiliary_cycle_treats_midnight_as_previous_day() -> None:
    timestamps = [datetime(2024, 1, 1, 23), datetime(2024, 1, 2, 0)]
    accumulated = np.array([[100.0], [120.0]])

    result = interval_energy(accumulated, timestamps)

    np.testing.assert_allclose(result[1], [20.0])


def test_interval_duration_uses_midnight_at_reset_and_actual_gaps() -> None:
    timestamps = [
        datetime(2024, 1, 2, 1),
        datetime(2024, 1, 2, 2),
        datetime(2024, 1, 2, 5),
    ]

    durations = interval_duration_seconds(timestamps)

    np.testing.assert_array_equal(durations, [3600.0, 3600.0, 10800.0])


def test_interval_energy_keeps_a_masked_previous_accumulation_as_missing() -> None:
    timestamps = [datetime(2024, 1, 1, 23), datetime(2024, 1, 2, 0)]
    accumulated = np.ma.masked_array(
        [[0.0], [120.0]], mask=[[True], [False]]
    )

    result = interval_energy(np.ma.filled(accumulated, np.nan), timestamps)

    assert np.isnan(result).all()
