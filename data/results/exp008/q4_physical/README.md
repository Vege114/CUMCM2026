# Q4-2 fixed physical common-mode transfer

One configuration was transferred from the verified Q2 physical initializer:
ridge28 midnight load/PV, causal midnight prices, three historical paths,
hourly common charge/discharge modes, switch weight 50, five-second MIP
initialization, followed by greedy-execution purchase refinement on all 28
historical paths for at most 120 L-BFGS-B iterations. No holiday feature,
future actual price, parameter sweep or shared-source edit was introduced.

The 30-day pilot passed its predeclared true-fee and reversal gate. Annual
execution then continued from that exact prefix with unchanged parameters.

| Metric | Migrated old dispatch | Physical common modes | Change |
|---|---:|---:|---:|
| 30-day true cost, yuan | 1,328,916.27 | 1,327,128.84 | -1,787.42 |
| 30-day reversals | 774 | 167 | -607 |
| 334-day true cost, yuan | 14,373,895.92 | 14,272,632.80 | -101,263.12 (-0.7045%) |
| Annual planned cost, yuan | 13,460,884.32 | 13,543,521.06 | +82,636.74 |
| Annual emergency cost, yuan | 913,011.60 | 729,111.73 | -183,899.86 |
| Annual non-idle reversals | 8,717 | 1,833 | -6,884 |
| Annual active ten-minute slots | 39,853 | 34,842 | -5,011 |
| Annual throughput, kWh | 13,640,457.61 | 11,208,918.89 | -2,431,538.72 |
| Simultaneous charge/discharge slots | 0 | 0 | 0 |
| Final SOC, kWh | 1,297.642928 | 1,200.000000 | -97.642928 |

Both cases start from the same Q4-2 January warmup state, 1390.382746315672
kWh, and carry actual SOC across all 334 days. The annual fee benefit remains
positive if the lower final inventory is valued at the highest observed
annual emergency tariff; this sensitivity is recorded separately and is
not subtracted from the reported true bill. The actual operating-intensity
improvement is substantial, while the fee improvement is small. These
intensity metrics are not a calibrated battery lifetime or ageing estimate.

Fees do not improve every month. July saves 103,994.00 yuan; April, June,
August and September increase cost. `comparison_monthly.csv` preserves
those differences and assigns each chronological reversal to its actual
month, using the same 1e-6 kWh non-idle threshold as the annual verifier.

All 334 actual replays pass independent raw-source, exact fee, physical,
continuous-SOC and mutual-exclusion checks. Every initialization passes
scenario physical checks; no meaningful discharge into scenario surplus
was found (maximum numerical residue 2.54e-9 kWh). Future actual load/PV/price
perturbations at three issue dates leave all optimizer inputs unchanged.
The pilot and full-year first 30 days match exactly in every archived field.
Every dependency snapshot matches its recorded source hash.

All 334 MIP initializations have feasible incumbents. Two hit the time limit;
the maximum reported relative gap is 0.005570 and the mean is 0.001695.
Only 26 of 334 refinements report normal solver convergence; the other
results retain their recorded local-solver termination statuses. No global
optimality or nonanticipative scenario-recourse certificate is claimed.

The comparator is `latest_forecast_old_dispatch/4-2`, which retains the
original CNN output. Thus the comparison measures the combined ridge28
forecast and physical planning transfer, not a pure mode-planning ablation.
The overall exp008 goal still requires its separately fixed Q2 cost/count
test; this experiment establishes the Q4-2 transfer result only.

Use `full334/dispatch_4-2.npz` and `full334/daily.csv` for the completed
candidate. `comparison.csv`, `comparison_daily.csv` and
`comparison_monthly.csv` contain the paired results. `verification_summary.json`
and stage verification files retain the independent checks. Method and
information boundaries are described in `method_notes.md`; exact source
versions and all daily scenario/purchase/execution records are retained.

Reproduce or resume under the recorded dependency versions with:
`.venv/bin/python -m experiments.exp008.run_q4_physical --extend-if-improved`.
Changing source or configuration requires a new output directory.
