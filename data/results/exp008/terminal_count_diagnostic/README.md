# Actual operation counts and midnight SOC diagnostic

Read-only development analysis of exp006/primary and exp008's completed `direct_hgb_memory_hold1_cap8_kappa0_334days`. The protocol freezes both dispatch archives and the historical metric/model source files. No purchases, forecasts or dispatch archives were changed. This is not the final report and does not change the 8% fee target.

## Count definition remains historical, but the wording must remain specific

The exp006 report at `reports/experiments/exp006/report.md:105` explicitly defines direction reversals by concatenating all 334 days, deleting idle signs and counting adjacent nonzero sign changes. Its verifier at `experiments/problem2/tree_planning/verify.py:21` implements that definition. Evaluation-period midnight transitions count; the January warmup transition does not. The exp008 primary direction-reversal metric therefore retains the exp006 definition.

An independent scalar chronological loop also counts contiguous charge/discharge episodes. An idle slot ends such an episode, whereas it does not reset the last non-idle direction used by the historical reversal metric. The first observed episode is counted, treating the evaluation boundary as idle for this separate metric.

| Metric | exp006 | hold1 cap8 | Change |
|---|---:|---:|---:|
| Non-idle direction reversals | 2,729 | 2,521 | −208, −7.62% |
| Charge episodes | 2,098 | 1,677 | −421, −20.07% |
| Discharge episodes | 1,693 | 2,211 | +518, +30.60% |
| Charge + discharge episodes | 3,791 | 3,888 | +97, +2.56% |
| Active 10-minute intervals | 23,028 | 37,217 | +14,189, +61.62% |
| AC throughput (kWh) | 11,480,847.039824 | 11,337,080.993656 | −1.25% |
| Simultaneous-flow intervals | 0 | 0 | 0 |

The supported statement is **fewer direction reversals and slightly less throughput**. An unqualified claim that all charge/discharge operation counts decreased is false for this candidate. More contiguous discharge episodes and more active intervals must remain visible when discussing the user's request for fewer battery actions. These proxies do not establish longer battery lifetime. `count_comparison.csv` preserves exact values and deltas.

A supplemental read-only check of three other completed candidates is saved with each archive's hash in `candidate_episode_comparison.json`. Fixed-hour HGB has 3,442 charge/discharge episodes, hold3 with switching penalty 50 has 3,313, and hold3 cap8 has 3,900. Thus the first two have fewer combined episodes than exp006's 3,791 as well as fewer non-idle reversals; both cap8 variants have slightly more combined episodes. All these variants still have more active 10-minute intervals than exp006. This comparison selects no new policy and does not change the fee threshold.

## The remaining fee gap is not concentrated in depleted midnights

Across the 333 internal evaluation-period day boundaries, actual SOC has mean 2,704.60 kWh, median 2,335.97 kWh and interquartile range 1,769.79–3,226.59 kWh. Eleven boundaries are at the 1,200 kWh lower bound; fifteen are within 50 kWh of it. The final December 31 SOC is excluded from this assessment because the evaluation ends there and terminal credit is correctly zero.

Next-day 00:00–06:00 emergency expense is 19,063.81 yuan: 3.33% of the full 334-day emergency bill and 6.84% of the 278,731.12 yuan remaining gap to the 8% target. Of that next-morning expense, 2,847.75 yuan follows a boundary within 50 kWh of the lower bound. Expanding the morning window to 09:00 gives 42,449.20 yuan. This association does not prove that higher previous-day SOC could eliminate those costs, because discharge masks and power limits can prevent using inventory.

`emergency_constraint_accounting.json` separately partitions the existing emergency flow in an explicit order: first slots whose mode forbids discharge, then demand exceeding rated discharge power where discharge is allowed, then unmet demand caused by current available SOC. The components sum to the stored emergency arrays. Of the 19,063.81 yuan next-morning bill, 18,414.36 belongs to discharge-forbidden modes, 525.47 to rated power, and only 123.98 to available SOC. Over the full 334 days these components are 129,732.19, 7,631.67 and 435,643.19 yuan respectively. This is an accounting decomposition at fixed recorded states and modes, not attribution of savings achievable by changing one constraint.

The current closed-loop refinement credits 0.45 yuan per **battery SOC kWh**, while the MIP initializer uses minimum tariff divided by charge efficiency: 0.391384565 yuan/SOC-kWh. The overnight 00:00–06:00 median replacement cost is 0.457634282 yuan/SOC-kWh. These are planning credits, excluded from reported actual purchase expense. Both credits become zero on the final day.

## Conditional inventory probes

For each of those 333 next days, the diagnostic independently replays its existing purchases, mode mask and recorded starting SOC using the same current-observation-only physical execution law. Rebuilt charge, discharge, emergency and SOC match the stored arrays to under 1e−8 kWh. It then repeats that fixed-Q replay with a free extra 100 or 1,000 SOC-kWh at the day's start. No probe actions are exported.

| Free added SOC | Mean full-day emergency-fee value (yuan/SOC-kWh) | Mean 00–06 value | Days with nonzero full-day value |
|---|---:|---:|---:|
| 100 kWh | 0.438925228 | 0.003723055 | 31 / 333 |
| 1,000 kWh | 0.224558779 | 0.000372306 | 31 / 333 |

The 100 kWh average is close to the existing 0.45 planning credit, but highly heterogeneous: 302 days have zero emergency-fee response under their fixed purchase/mask. The extra inventory can also displace charging or be lost through later saturation. This result is not an estimate of an optimally replanned next-day value function.

The probes use future realized demand only for retrospective evaluation of a fixed causal executor. Each day's added starting SOC is an independent free intervention; the collection is not a feasible continuous annual strategy, the fee changes are not additive realizable savings, and the probes do not pay to acquire inventory. Next-day purchases remain fixed in this diagnostic, whereas a two-day planner could change them.

A causal two-day planning approach is worth at most a bounded initial diagnostic on preselected dates to see whether purchase adaptation adds value. These data do not justify assuming terminal underpricing is the dominant remaining problem or immediately running a new full annual strategy. A proper diagnostic must construct both days' forecasts from the first midnight's information, preserve paired historical paths and inherited mode state, and release only the first day's purchases before replanning next midnight.
