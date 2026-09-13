# Q4-2 price forecast linked to own-issued load/PV forecasts

This is one predeclared forecast diagnostic on all 334 development days. No battery optimization or realized purchase-cost comparison was run, and this result is not evidence that Q2's 8% target was reached. The shared `Forecasts` class, HGB forecasts and completed Q4 dispatch archive were preserved.

The original Q4 midnight price forecast uses weighted ridge regression over the previous 28 complete days, with a 14-day half-life. Its eight columns are intercept, the known attachment-1 tariff, previous-day and previous-week observed prices, and daily/half-daily sine/cosine features. This candidate appends exactly two columns: its own midnight-issued HGB load and PV forecasts, each divided by 1,000 kW. Both new coefficients have zero prior mean and ridge penalty 5; all original columns, penalties, priors, label weights and positive-price clipping remain unchanged. No alternative weights or models were searched in this diagnostic.

Every historical regression row receives the load/PV forecast that was available at that row's own midnight. February–December use the frozen HGB → Ridge28 → half-memory issued archive. January is explicitly the existing periodic cold start, not a retroactively applied HGB model. Current-day actual load, PV or prices are not features or training labels.

| Full 334-day forecast metric | Original price model | Linked HGB price model |
|---|---:|---:|
| Price RMSE (yuan/kWh) | 0.061947789 | 0.056139984 |
| Price MAE (yuan/kWh) | 0.046318007 | 0.042157169 |
| RMSE where realized price ≥1 yuan/kWh | 0.069592330 | 0.063409604 |
| MAE where realized price ≥1 yuan/kWh | 0.050580840 | 0.046342963 |
| Realized-price-weighted MAE | 0.047931190 | 0.043935698 |
| Fixed-purchase absolute price-exposure proxy (yuan) | 811,363.507193 | 752,841.050692 |

Price RMSE decreases about 9.38%. All four recorded forecast gates improve: full RMSE, full MAE, high-price RMSE and fixed-purchase absolute exposure. The high-price subset contains 12,614 intervals; it is defined from realized prices only for scoring.

The last row fixes the existing completed Q4-2 HGB policy's original purchased kWh and computes `sum(q * abs(predicted_price−realized_price))`. Its 58,522.46 yuan decrease is a forecast exposure diagnostic, **not an actual electricity-bill saving**. Purchases would change under optimization, and actual bills always use realized prices. The additional price-weighted MAE uses realized price only as an evaluation weight. The entire 2025 evaluation year has already been repeatedly examined during development, so these results are not an untouched test or an independent final selection.

## Verification

`protocol.json` and the exact source snapshot were written before forecasting/scoring. Hashes freeze the original price model, HGB inputs/provenance and fixed purchase archive. All hashes remain unchanged. For all 334 days, the original eight design columns and the issued load/PV outputs equal the previous implementation exactly; price labels and historical feature issue times strictly precede the current midnight.

Twenty-four checks independently mutate future actual prices, future actual load/PV, future forecast-store rows, or all of these, on days 31, 32, 60, 151, 243 and 364. Current price predictions and fitted coefficients remain unchanged. The asymmetric store perturbation changes future net forecasts; day 364 has no later store rows and its corresponding test is explicitly marked vacuous.

The separate independent audit rebuilds all 364 available historical/current load/PV feature days directly from the frozen HGB store or January raw-data lags. It then reconstructs all 334 price regression designs from raw source prices and solves each ridge fit as an augmented rectangular least-squares system using SVD, instead of the production normal-equation solve. Maximum feature error is zero, coefficient error 7.25e−13 and prediction error 1.71e−13 yuan/kWh. The scoring price array equals the original attachment-4 CSV exactly.

Five additional nonvacuity checks perturb the **known current** issued load forecast by 1,000 kW. Price forecasts respond by the expected fitted coefficient, confirming that the new known inputs actually enter the model. These sensitivity probes are not future-information tests or alternative selected policies.

Reusable implementation: `experiments.exp008.hgb_linked_price_forecast.LinkedPriceForecasts`. Detailed evidence: `price_predictions.npz`, `prediction_audit.json`, `daily_scores.csv`, `causality_audit.json`, `independent_audit.json` and their source snapshots. A subsequent dispatch experiment must preserve the same price information timing and settle actual fees against the realized attachment-4 tariff.

Metadata erratum: the inherited generic `load_method` string still says `common_midnight_cnn_remaining_trajectory`, although `base_forecast`, `selected_model_id`, source hashes and every forecast value correctly identify the HGB pipeline. `metadata_erratum.json` records this existing label residue; `source_corrected_prediction_audit.json` changes only that display field for all 334 rows. The original signed audit, code and all numerical forecasts remain unchanged.
