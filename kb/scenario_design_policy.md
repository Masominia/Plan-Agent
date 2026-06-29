# Scenario design policy

## Purpose and scope
Planning scenarios translate a narrative about the economy into consistent paths for every variable the
planning models need, over a nine-quarter horizon, the same horizon as the Federal Reserve's supervisory stress
test. Scenarios are used for budget planning, firm-wide scenario analysis and regulatory stress testing.

## Scenario expansion
A scenario owner specifies a small number of anchor variables, usually the real GDP growth path and,
optionally, the cumulative equity decline over the first three quarters. All remaining variables are filled in
by the scenario expansion model (model PA-001, a VAR estimated on 1990-2024 data) as the most likely path
consistent with the anchors. Hand-written paths for non-anchor variables are not allowed, because they are
frequently internally inconsistent, for example a deep equity decline with a calm volatility index.

## Plausibility and severity
Every model-based scenario carries a plausibility statistic: the chi-square distance of the shocks needed to
produce the anchors, and its p-value. A baseline should have a p-value well above 0.1. Adverse scenarios
typically have p-values between 0.01 and 0.3. Severely adverse scenarios are deliberately extreme; p-values far
below 0.001 are expected and must be documented, not avoided. Severity is benchmarked against the Federal
Reserve's published severely adverse scenario and against historical episodes such as 2008-09 and 2020.

## Governance and approval
Scenario definitions, anchor values, model versions and the resulting paths are recorded in the audit log of
each run. Scenarios used for regulatory submissions require review by Model Risk Management and sign-off by the
scenario committee. Analysts must not alter model outputs by hand; overlays are documented separately with
their rationale.

## Limitations
The expansion model is linear and Gaussian: it understates how strongly variables co-move in crises and has no
regime switching. Results for severe scenarios should be read as central paths, with tail risk assessed through
the line-item Monte Carlo and historical replays.
