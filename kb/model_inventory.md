# Model inventory

## PA-001 Scenario expansion VAR
Purpose: expand scenario anchors into full macro paths. Specification: VAR(1) on real GDP growth, CPI
inflation, 10-year Treasury yield, S&P 500 return and the quarterly maximum of the VIX, estimated by least
squares on 1990Q2-2024Q4, with 2020Q2 and 2020Q3 excluded from estimation. Conditional forecasts use Gaussian
conditioning (Waggoner and Zha). Output: nine-quarter paths and a chi-square plausibility p-value. Owner:
Planning and Analysis Strats. Status: illustrative, not validated for production.

## PA-002 Earnings projection model
Purpose: project S&P 500 trailing-twelve-month earnings per share under a scenario. Specification: ARX(2) in
quarterly log growth with GDP growth, inflation, change in the 10-year yield and equity return, estimated by
Huber M-estimation on 1990Q2-2023Q2 to limit the influence of the 2008-09 write-down quarters. Uncertainty:
stationary block bootstrap of residuals with mean block length four quarters, 5,000 paths by default.
Known limitation: the 1990-2023 sample is short and dominated by three recessions; equity sensitivity is
higher than on longer samples.

## PA-003 Dividend projection model
Purpose: project S&P 500 dividends per share. Specification: Lintner (1956) partial adjustment: quarterly
dividend growth depends on its own two lags, four lags of earnings growth and the lagged log payout ratio, so
dividends mean-revert to a target payout. Simulated on top of each PA-002 earnings path, with residuals drawn
from the same historical quarters as the earnings shocks. Known behaviour: dividends are sticky and fall much
less than earnings in stress scenarios.

## Validation and monitoring
Each model has automated tests and run-time checks. Monitoring covers parameter stability, forecast error
drift and input drift. Any change to specification, sample or data source requires a new model version and an
entry in the change log.
