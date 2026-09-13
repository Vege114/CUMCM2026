# One bounded holiday-calendar forecast diagnostic

The State Council General Office published the 2025 holiday/makeup-workday
schedule on 2024-11-12, in 国办发明电〔2024〕12号. Its dates were available before
all evaluated load observations. The official gov.cn traditional-Chinese
mirror was checked; the simplified endpoint returned HTTP 403.
[Official notice](https://big5.www.gov.cn/gate/big5/www.gov.cn/zhengce/zhengceku/202411/content_6986383.htm).

The community's region and actual observance of this calendar are unknown.
This is a hypothetical exogenous feature, not an established local mechanism.
The full calendar has 28 holiday dates and 5 makeup-workday dates. The formal
February–December interval contains 23 holidays and 4 makeup workdays.

Baseline ridge28 errors on holidays were concentrated: net RMSE 481.98 kW
versus 376.95 on ordinary weekdays; mean absolute daily integrated net error
5,316.75 versus 2,873.73 kWh. All four makeup days had negative mean net error,
averaging -141.42 kW. However, PV error contributes substantially, and Labour
Day versus National Day errors have opposing signs. Calendar association is
not evidence of a single causal demand mechanism.

Exactly one model was specified and tested. For each special issue date, a
ridge model uses only prior complete formal days' ridge28 **load** residuals,
averaged into four six-hour blocks. Its nine features are seven weekday
indicators plus holiday and makeup-workday increments. Training uses a
56-day half-life, ridge penalty equivalent to four daily observations,
at least 14 prior days, and a fixed +/-500 kW correction cap. Only the fitted
calendar increment is applied. Ordinary dates and every PV forecast value
remain unchanged. Correction signs are learned from prior labels, never the
current date's load or error. Early-February special dates remain unchanged
because fewer than 14 formal historical days exist.

| Metric | Ridge28 | Calendar increment |
|---|---:|---:|
| Full-year net RMSE, kW | 387.683 | 387.176 |
| Special-day net RMSE, kW | 462.497 | 457.209 |
| Special-day high-tariff net RMSE, kW | 444.566 | 434.865 |
| Special-day load RMSE, kW | 250.427 | 241.751 |
| Special-day mean absolute integrated net error, kWh/day | 5,031.92 | 4,873.62 |

It passed the predeclared point gate (lower overall, special-date and
high-tariff special-date net RMSE), allowing exactly one planning link.
Fresh tree28 residual supports were rebuilt from its own prequential
forecasts, with LP quantile .8 and state buffer 500 unchanged.

| Full 334-day result | Ridge28 control | Calendar increment |
|---|---:|---:|
| Total cost, yuan | 13,393,480.15 | 13,377,421.38 |
| Planned cost, yuan | 12,679,818.49 | 12,679,983.96 |
| Emergency cost, yuan | 713,661.65 | 697,437.42 |
| Non-idle direction reversals | 6,681 | 6,723 |
| Active ten-minute slots | 39,594 | 39,645 |
| Final SOC, kWh | 1,376.422315 | 1,376.422315 |

The one planning link saves 16,058.76 yuan (0.120%) with identical final SOC,
but adds 42 reversals. This is a small useful development effect, not the
required 8% saving and lower-reversal result. No additional calendar models,
sign choices or planning parameters were searched.

`forecasts.npz` contains origins, ridge base values, signed deltas, resulting
forecasts and post-hoc residual labels. `audit.json` preserves every fit's
historical origins and coefficients. `metrics.csv`, `daily_metrics.csv` and
`special_day_metrics.csv` support the prediction comparisons. The
`dispatch_linkage` directory contains the full plan/replay, newly built
supports, paired daily/monthly deltas and independent verification.

Current/future actual values and future forecast values were perturbed at
April 4, September 28 and October 3; issue-time outputs remained exactly the
same. All source, physics, billing, continuous SOC and mutual-exclusion
checks pass for the 334-day planning link. The evaluation year has already
been examined in development and is not an independent test set.

Reproduce in two steps:
`.venv/bin/python -m experiments.exp008.holiday_forecast`, then
`.venv/bin/python -m experiments.exp008.holiday_dispatch`.
Both refuse to overwrite existing completed experiment output.
