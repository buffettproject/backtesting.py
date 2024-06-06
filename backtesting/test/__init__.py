"""Data and utilities for testing."""

import pandas as pd


def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(
        join(dirname(__file__), filename),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )


GOOG = _read_file("GOOG.csv")
"""DataFrame of daily NASDAQ:GOOG (Google/Alphabet) stock price data from 2004 to 2013."""

EURUSD = _read_file("EURUSD.csv")
"""DataFrame of hourly EUR/USD forex data from April 2017 to February 2018."""

BTCUSDT = _read_file("BTCUSDT.csv")
"""DataFrame of minutely spot market binance BTC/USDT data from 2017-08-17 13:00:00 to 2021-08-01 09:00:00."""

BTCUSDT_1H = _read_file("BTCUSDT_1H.csv")
"""DataFrame of hourly spot market binance BTC/USDT data from 2017-08-17 13:00:00 to 2021-08-01 09:00:00."""

ETHUSDT = _read_file("ETHUSDT.csv")
"""DataFrame of hourly spot market binance BTC/USDT data from 2017-08-17 13:00:00 to 2021-08-01 09:00:00."""


def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()


def RSI(array, n):
    """Relative strength index"""
    # Approximate; good enough
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
    return 100 - 100 / (1 + rs)


def MAX(arr: pd.Series, n: int) -> pd.Series:
    return pd.Series(arr).rolling(n).max()


def MIN(arr: pd.Series, n: int) -> pd.Series:
    return pd.Series(arr).rolling(n).min()


def IS_LOCAL_MAX(arr: pd.Series, n: int) -> pd.Series:
    return (
        pd.Series(arr).rooling(2 * n + 1).max().shift(-n)
        == pd.Series(arr).rolling(1).max()
    )


def IS_LOCAL_MIN(arr: pd.Series, n: int) -> pd.Series:
    return (
        pd.Series(arr).rooling(2 * n + 1).min().shift(-n)
        == pd.Series(arr).rolling(1).min()
    )
