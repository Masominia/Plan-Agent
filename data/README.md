# Data

`data/raw` is a snapshot of public series from the datahub.io core datasets on GitHub; the
SQLite database `data/planagent.db` is rebuilt from it with `python -m planagent build-db` (or automatically on
first use) and is not committed.

| Raw file | Source |
|---|---|
| `gdp_quarterly.csv` | BEA real GDP (`datasets/gdp-us`) |
| `cpi_monthly.csv` | BLS CPI-U, NSA (`datasets/cpi-us`) |
| `ust10y_monthly.csv` | Federal Reserve 10-year Treasury yield (`datasets/bond-yields-us-10y`) |
| `sp500_monthly.csv` | R. Shiller S&P 500 price, dividends, earnings (`datasets/s-and-p-500`) |
| `vix_daily.csv` | CBOE VIX (`datasets/finance-vix`) |
| `hpi_monthly.csv` | S&P CoreLogic Case-Shiller national index (`datasets/house-prices-us`) |
| `wti_daily.csv` | EIA WTI spot (`datasets/oil-prices`) |
| `eurusd_daily.csv` | Federal Reserve H.10 EUR per USD (`datasets/exchange-rates`) |

Known issues handled in `src/planagent/data.py`: the two Case-Shiller columns are labelled the wrong way round
(the seasonally adjusted one is used), CPI is seasonally adjusted by removing quarter-of-year means, GDP growth is
recomputed from the chained level, non-positive WTI prints are dropped, and EURUSD starts in 1999.

The knowledge base in `kb/` is written for this project; the Federal Reserve reference points in
`kb/fed_2026_severely_adverse.md` cite their public sources.
