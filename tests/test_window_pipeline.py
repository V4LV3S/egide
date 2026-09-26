import numpy as np
import pandas as pd

from ml.scripts.window_pipeline import (
    create_train_validation_test_windows,
    default_past_columns,
)


def make_dataframe() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=150, freq="h")
    split = np.repeat(["train", "validation", "test"], 50)
    return pd.DataFrame(
        {
            "time": timestamps,
            "split": split,
            "val_cargammgd": np.arange(150, dtype=float),
            "t2m_pca_0": np.arange(150, dtype=float) + 1000,
            "hour_sin": np.arange(150, dtype=float) + 2000,
            "hour_cos": np.arange(150, dtype=float) + 3000,
        }
    )


def test_windows_keep_past_calendar_and_target_aligned() -> None:
    data = make_dataframe()
    result = create_train_validation_test_windows(
        data,
        past_columns=["val_cargammgd", "t2m_pca_0"],
        calendar_columns=["hour_sin", "hour_cos"],
    )
    x_past_train, x_calendar_train, y_train = result[:3]

    assert x_past_train.shape == (3, 24, 2)
    assert x_calendar_train.shape == (3, 24, 2)
    assert y_train.shape == (3, 24)
    np.testing.assert_array_equal(x_past_train[0, :, 0], np.arange(24))
    np.testing.assert_array_equal(x_calendar_train[0, :, 0], np.arange(2024, 2048))
    np.testing.assert_array_equal(y_train[0], np.arange(24, 48))


def test_windows_never_cross_split_boundaries() -> None:
    data = make_dataframe()
    result = create_train_validation_test_windows(
        data,
        past_columns=["val_cargammgd", "t2m_pca_0"],
        calendar_columns=["hour_sin", "hour_cos"],
    )

    for x_past, _, y in (result[:3], result[3:6], result[6:]):
        assert len(x_past) == 3
        assert y[-1, -1] in {49.0, 99.0, 149.0}


def test_windows_skip_ranges_with_missing_hours() -> None:
    data = make_dataframe().drop(index=30).reset_index(drop=True)
    result = create_train_validation_test_windows(
        data,
        past_columns=["val_cargammgd", "t2m_pca_0"],
        calendar_columns=["hour_sin", "hour_cos"],
    )

    assert result[0].shape[0] == 0
    assert result[3].shape[0] == 3
    assert result[6].shape[0] == 3


def test_default_past_columns_selects_target_and_pcas() -> None:
    data = make_dataframe()

    assert default_past_columns(data) == ["val_cargammgd", "t2m_pca_0"]
