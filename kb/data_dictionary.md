# Data dictionary

## Tables
macro_quarterly holds quarterly macro-financial variables from 1990Q2 to 2024Q4: gdp_growth (real GDP growth,
percent annualised), cpi_inflation (percent annualised, seasonal quarter means removed because the CPI index is
not seasonally adjusted), ust10y (quarterly average 10-year Treasury yield, percent), sp500_return (quarterly log
return in percent), vix_max (maximum daily VIX close in the quarter, the Federal Reserve's convention),
house_price_growth (Case-Shiller national index, percent annualised), wti_return (quarterly log return of the
average WTI price) and eurusd (US dollars per euro at the end of the quarter, from 1999). sp500_fundamentals
holds the S&P 500 price and trailing-twelve-month earnings and dividends per index share from R. Shiller, from
1960Q1 to 2023Q2, with quarterly log growth rates. recessions lists NBER business-cycle peaks and troughs.
series_catalog documents every column.

## Conventions
Quarters are stored as text 'YYYYQn', which sorts chronologically. Growth rates are 100 times the change in the
natural log for quarterly rates and 400 times for annualised rates, so cumulative changes are sums.

## Caveats
Shiller's monthly earnings are interpolated from quarterly reports, and reported earnings include write-downs:
in 2008Q4 trailing earnings fell by more than 100 log points. Macro data are latest vintage, not real-time.
The house price snapshot has its two columns labelled the wrong way round; the database uses the seasonally
adjusted one.
