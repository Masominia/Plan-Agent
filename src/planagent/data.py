"""Build the analytical SQLite database the agent queries.

Tables (all quarterly, ``quarter`` as 'YYYYQn'):

* ``macro_quarterly``: real GDP growth, CPI inflation, 10Y yield, S&P 500
  return, VIX quarterly maximum, house-price growth, WTI return, EURUSD,
  1990Q2-2024Q4;
* ``sp500_fundamentals``: S&P 500 price and trailing-12-month earnings and
  dividends per index share (R. Shiller), 1960Q1-2023Q2;
* ``recessions``: NBER business-cycle peaks and troughs since 1960;
* ``series_catalog``: description, unit and source of every column.

The database is rebuilt from the CSV snapshot in ``data/raw`` (never edited by
hand), so every number the agent reports traces back to a public file.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "planagent.db"

NBER = [
    ("1960Q2", "1961Q1"),
    ("1969Q4", "1970Q4"),
    ("1973Q4", "1975Q1"),
    ("1980Q1", "1980Q3"),
    ("1981Q3", "1982Q4"),
    ("1990Q3", "1991Q1"),
    ("2001Q1", "2001Q4"),
    ("2007Q4", "2009Q2"),
    ("2019Q4", "2020Q2"),
]

CATALOG = [
    ("macro_quarterly", "gdp_growth", "Real GDP growth, annualised (400 x dln)", "% per year", "BEA via datahub.io"),
    (
        "macro_quarterly",
        "cpi_inflation",
        "CPI-U inflation, annualised, seasonal means removed",
        "% per year",
        "BLS via datahub.io",
    ),
    ("macro_quarterly", "ust10y", "10-year Treasury yield, quarterly average", "%", "Federal Reserve via datahub.io"),
    (
        "macro_quarterly",
        "sp500_return",
        "S&P 500 price return (100 x dln, monthly-average prices)",
        "% per quarter",
        "R. Shiller via datahub.io",
    ),
    ("macro_quarterly", "vix_max", "VIX, maximum daily close in the quarter", "index points", "CBOE via datahub.io"),
    (
        "macro_quarterly",
        "house_price_growth",
        "Case-Shiller national HPI growth, annualised",
        "% per year",
        "S&P CoreLogic via datahub.io",
    ),
    (
        "macro_quarterly",
        "wti_return",
        "WTI spot return (100 x dln of quarterly average)",
        "% per quarter",
        "EIA via datahub.io",
    ),
    (
        "macro_quarterly",
        "eurusd",
        "EURUSD, last daily rate in the quarter",
        "USD per EUR",
        "Federal Reserve H.10 via datahub.io",
    ),
    ("sp500_fundamentals", "price", "S&P 500 index, quarter-end month average", "index points", "R. Shiller"),
    ("sp500_fundamentals", "eps_ttm", "Trailing-12-month reported earnings per index share", "USD", "R. Shiller"),
    ("sp500_fundamentals", "dps_ttm", "Trailing-12-month dividends per index share", "USD", "R. Shiller"),
    ("sp500_fundamentals", "eps_growth", "Quarterly log growth of eps_ttm (100 x dln)", "%", "derived"),
    ("sp500_fundamentals", "dps_growth", "Quarterly log growth of dps_ttm (100 x dln)", "%", "derived"),
]


def _read(name, date_col):
    df = pd.read_csv(RAW / name)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def _q(s: pd.Series, how: str) -> pd.Series:
    r = s.resample("QS")
    out = {"mean": r.mean, "last": r.last, "max": r.max}[how]()
    out.index = out.index.to_period("Q")
    return out


def build_frames() -> dict[str, pd.DataFrame]:
    gdp = _read("gdp_quarterly.csv", "date")["level-chained"]
    gdp.index = gdp.index.to_period("Q")
    cpi = _q(_read("cpi_monthly.csv", "Date")["Index"], "mean")
    infl = 400 * np.log(cpi).diff()
    infl = infl - infl.groupby(infl.index.quarter).transform("mean") + infl.mean()
    sh = _read("sp500_monthly.csv", "Date")
    hpi = _read("hpi_monthly.csv", "Date")["National-US"]  # the seasonally adjusted column (labels swapped)
    wti = _read("wti_daily.csv", "Date")["Price"]
    fx = 1.0 / _read("eurusd_daily.csv", "date")["eur_per_usd"]
    macro = pd.DataFrame(
        {
            "gdp_growth": 400 * np.log(gdp).diff(),
            "cpi_inflation": infl,
            "ust10y": _q(_read("ust10y_monthly.csv", "Date")["Rate"], "mean"),
            "sp500_return": 100 * np.log(_q(sh["SP500"], "last")).diff(),
            "vix_max": _q(_read("vix_daily.csv", "DATE")["CLOSE"], "max"),
            "house_price_growth": 400 * np.log(_q(hpi, "mean")).diff(),
            "wti_return": 100 * np.log(_q(wti[wti > 0], "mean")).diff(),
            "eurusd": _q(fx.dropna(), "last"),
        }
    ).loc["1990Q2":"2024Q4"]
    q = sh[sh.index.month.isin([3, 6, 9, 12])]
    fund = pd.DataFrame(
        {
            "price": _q(q["SP500"], "last"),
            "eps_ttm": _q(q["Earnings"].where(q["Earnings"] > 0), "last"),
            "dps_ttm": _q(q["Dividend"].where(q["Dividend"] > 0), "last"),
        }
    )
    fund["eps_growth"] = 100 * np.log(fund["eps_ttm"]).diff()
    fund["dps_growth"] = 100 * np.log(fund["dps_ttm"]).diff()
    fund = fund.loc["1960Q1":"2023Q2"]
    rec = pd.DataFrame(NBER, columns=["peak", "trough"])
    cat = pd.DataFrame(CATALOG, columns=["table_name", "column_name", "description", "unit", "source"])
    return {
        "macro_quarterly": macro.round(6),
        "sp500_fundamentals": fund.round(6),
        "recessions": rec,
        "series_catalog": cat,
    }


def build_db(path: Path = DB) -> Path:
    frames = build_frames()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        for name, df in frames.items():
            out = df.copy()
            if isinstance(out.index, pd.PeriodIndex):
                out.index = out.index.astype(str)
                out = out.reset_index(names="quarter")
            out.to_sql(name, con, index=False)
    return path


def ensure_db(path: Path = DB) -> Path:
    return path if path.exists() else build_db(path)


def macro_frame(path: Path = DB) -> pd.DataFrame:
    with sqlite3.connect(ensure_db(path)) as con:
        df = pd.read_sql("SELECT * FROM macro_quarterly ORDER BY quarter", con)
    df.index = pd.PeriodIndex(df.pop("quarter"), freq="Q")
    return df


def fundamentals_frame(path: Path = DB) -> pd.DataFrame:
    with sqlite3.connect(ensure_db(path)) as con:
        df = pd.read_sql("SELECT * FROM sp500_fundamentals ORDER BY quarter", con)
    df.index = pd.PeriodIndex(df.pop("quarter"), freq="Q")
    return df
