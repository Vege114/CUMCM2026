# Causal residual-window ablation

The three permitted alternatives were evaluated without changing ridge28
forecasts, LP quantile `.8`, or the state-buffer-500 executor.  The original
tree28 q=.8 ridge28 archive lacked the state buffer, so a strictly matching
control was replayed.  No shared forecast, tree, planner or controller source
was changed.  These are development experiments on 2025, not an independent
test or a claim to meet the overall exp008 target.

| Support | First 30 days bill, yuan | Emergency bill | LP threshold coverage |
|---|---:|---:|---:|
| tree28 control | 1,256,142.27 | 33,619.14 | 79.12% |
| tree7 | 1,253,072.60 | 42,040.61 | 76.48% |
| tree14 | 1,249,962.09 | 35,043.50 | 77.92% |
| per_slot14 | 1,248,154.58 | 40,116.94 | 73.47% |

The pilot interval was fixed before running: February 1 through March 2.
The extension gate required at least 0.5% real-bill savings after charging
any lower final SOC at the conservative upper bound `sqrt(.9)*5*p_max`.
Only per_slot14 passed, with 7,046.02 yuan adjusted savings (0.56%).
Tree7 and tree14 were not extended or retuned.  Early fallback matters:
6,041.48 of tree14's 6,180.17 yuan pilot savings arose in the first two days,
when the explicitly labelled periodic-baseline residual fallback is used.
The pilot cannot establish that seasonal transitions improved.

| Support | 334-day bill, yuan | Emergency bill | Direction reversals |
|---|---:|---:|---:|
| tree28 control | 13,393,480.15 | 713,661.65 | 6,681 |
| per_slot14 | 13,461,551.02 | 793,981.04 | 7,537 |

Per_slot14 fails on the full year: plan fees fall 12,248.51 yuan but emergency
fees rise 80,319.39, for a net increase of 68,070.88 yuan.  September alone
adds 38,718.45 yuan.  Its nominal 88.89% central support range covers only
77.84% of actual net demand, versus 86.43% for tree28.  Tariff-weighted pinball
loss also worsens from 13.2555 to 13.6598.  Narrower same-slot empirical
supports are not better calibrated and do not solve seasonal lag here.

The matching tree28 control is a useful result in its own right, but its
13.39348 million yuan bill and 6,681 reversals still miss the exp006 target.
No further window or conditioning configurations were explored.

`comparison.csv` contains the compact paired results; `annual_monthly_delta.csv`
contains the full-year paired monthly bill differences. Each case directory
contains its dispatch, issued supports, daily metrics, training origins and
independent physics/billing verification. `implementation_verification.json`
records exact equality to the original tree28 implementation at seven issue
dates and future-actual/future-forecast mutation checks for all alternatives.

All nine support atoms and interpolation conventions were preserved. In
particular, the LP calls NumPy quantile `.8` over nine support atoms whose
underlying historical levels are `(j+.5)/9`; interpolating those levels gives
0.766667. The displayed coverage is the actual decision threshold coverage,
not an assertion of a calibrated exact 80% historical inverse CDF.

Reproduction: `.venv/bin/python -m experiments.exp008.risk_window --extend-if-improved`.
Completed output is reused only under the exact recorded protocol and source
hashes. Annual extension is restricted to alternatives that passed the fixed
pilot gate.
