# Q4-2: controlled transfer of the HGB-linked price forecast

This completed 334-day development experiment changes only the causal price forecasting method relative to `q4_absolute_hgb_physical`. The load/PV forecast remains the identical absolute-HGB → Ridge28 → half-memory issued pipeline. The new price model adds its own issued load and PV forecasts to the existing price Ridge28 regression; historical regressors use their own midnight issues. The earlier forecast-only diagnostic improved price RMSE by about 9.38% before this full dispatch run was predeclared.

| Actual 334-day result | Previous HGB / original price model | HGB / linked price model | Change |
|---|---:|---:|---:|
| Total fee (yuan) | 14,018,022.610012 | 13,994,531.257937 | −23,491.352076 (−0.16758%) |
| Original planned purchase fee (yuan) | 13,406,495.981770 | 13,400,456.682553 | −6,039.299217 |
| Emergency purchase fee (yuan) | 611,526.628243 | 594,074.575384 | −17,452.052859 |
| Non-idle direction reversals | 1,793 | 1,763 | −30 |
| Active 10-minute intervals | 35,231 | 35,187 | −44 |
| AC throughput (kWh) | 11,259,341.875293 | 11,179,601.983458 | −79,739.891835 |
| Equivalent full cycles | 468.489434 | 465.171544 | −3.317891 |
| Charge + discharge episodes | 3,449 | 3,452 | +3 |
| Power total variation (kW) | 31,060,640.076020 | 28,859,635.103811 | −2,201,004.972209 |
| Final SOC (kWh) | 1,200 | 1,200 | 0 |

Both groups have zero simultaneous charge/discharge and satisfy the same physical constraints. Charge episodes fall by 57, while discharge episodes rise by 60, so combined episodes rise by three despite fewer direction reversals and active intervals. These measures are operational proxies, not battery-life estimates. Reported fees contain actual purchases and emergency charges only, with no terminal-value or wear-penalty deduction. The price-only forecast-exposure improvement of 58,522.46 yuan was not treated as actual savings; actual savings from this completed dispatch comparison are 23,491.35 yuan.

`three_layer_comparison.csv` and `monthly_three_layer_comparison.csv` also retain the preceding Ridge28 load/PV Q4 policy with total fee 14,272,632.796947 yuan. The immediate previous HGB group above is the controlled comparison that isolates the price-model change; the older comparison changes both load/PV and price methods.

## Fixed execution and information rules

The run uses the same continuous initial SOC of 1,390.382746315672 kWh from the Q4-2 warmup, the same previous non-idle mode and actual power, 12,000 kWh battery, SOC range 1,200–10,800 kWh, 5,000 kW limits and square-root-of-0.9 one-way efficiencies. Day-ahead purchase is locked before actual demand is read. A three-scenario physical MIP initializes purchases and a shared hourly charge/discharge mask; the unchanged 28-history greedy refinement then optimizes purchases under that mask. The MIP has a five-second budget and target relative gap 0.2%; L-BFGS-B has at most 120 iterations. Wear, variation, terminal credit and deadband settings match the preceding HGB run.

Optimization uses the midnight causal price forecast. Actual attachment-4 prices enter only after purchases and physical actions are fixed, in `sum(actual_price * (original_purchase + 5 * emergency))`. All 28 historical price-error paths are recomputed using the new method at each historical issue, retained in each planning archive, and independently verified; the current planner uses the mean price trajectory, not a stochastic price-scenario objective. The load/PV paths remain byte-for-byte equal to the preceding HGB run.

The first three days were a computational/physical feasibility gate only. Their cost and counts did not select whether to continue. The same process continued through all 334 days with continuous SOC/mode; every first-three-day array is exactly the corresponding prefix of the completed archive. There was no partial-cost selection or parameter sweep.

## Verification and scope

The protocol freezes 36 local source/dependency files, the warmup, raw-data hashes, full HGB issued values and origins, and twelve forecast/model evidence artifacts. The dependency capture includes loaded local modules and statically discoverable local imports. Final hashes remain unchanged. `controlled_input_audit.json` confirms identical warmup, raw inputs, HGB arrays and the AST of all plan/refine/execute calls; configuration differences are limited to the price method and its descriptions.

`full334/independent_audit.json` verifies every day against raw source demand and realized tariffs, SOC/balance/physical bounds, original purchases and the exact actual executor. It checks every same-issue HGB/net path and own historical price-error path, plus five predetermined purchase-refinement reconstructions. Fresh future-actual mutation checks pass, complementing the predeclared price-model audit's asymmetric future-data and future-store tests. Full evaluation contains 48,096 ten-minute intervals with continuous day boundaries and zero simultaneous flows.

Every MIP day has a feasible incumbent; maximum gap is 1.70485%, mean gap 0.204862%. The local greedy refinement reports success on 35 of 334 days, with the remaining runs preserving the fixed 120-iteration cap and their solver messages. These are numerical approximations, not a continuous global-optimality or nonanticipative scenario-recourse certificate.

`metadata_erratum.json` explains an inherited generic CNN `load_method` label even though the base model, selected ID and numerical arrays identify HGB correctly. `full334/source_corrected_audit.json` changes only that display label; raw signed audit records, code and all numerical outputs are preserved.

This result improves Q4-2. It does not satisfy, weaken or re-evaluate the separate Q2 requirement of at least 8% actual-cost reduction relative to exp006. No final experiment report or final model selection is made here.
