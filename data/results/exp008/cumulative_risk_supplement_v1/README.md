# Cumulative-risk planning: independent supplement v1

This is development evidence, not the final experiment report. No new purchase policy, annual dispatch, or numerical replacement was run for this supplement. The original fourteen signed source/output files remained unchanged, and the supplemental source snapshot matches its protocol hash. `supplementary_audit.json` contains the checks; `source_snapshot.py` is the exact implementation of those checks.

The two original 334-day results both fail the requested fee/count target:

| Method | Actual purchase fee (yuan) | Non-idle direction reversals | Final SOC (kWh) |
|---|---:|---:|---:|
| Marginal cumulative-prefix q0.8 | 13,706,239.587825 | 8,579 | 1,224.503590 |
| Sum of pointwise q0.8 margins | 13,190,724.676222 | 7,173 | 1,714.377346 |

Both use the same daily forecast, previous 28 complete joint issued errors, true empirical linear q0.8, and nominal 500 kWh state buffer. With `f` in AC kWh, prefix risk is `r_t=f_t+a_t−a_(t−1)`, where `a_t=q0.8(sum_(u≤t) error_u)` and `a_(-1)=0`. The pointwise control instead uses `r_t=f_t+q0.8(error_t)`. The prefix increments may be negative. Clipping them would invalidate the cumulative identity. There is no general ordering between a quantile of a sum and a sum of quantiles; the independent examples and all 334 identities are recorded in the audit. Maximum independently rebuilt risk-array error was 9.10e−13 kWh.

The LP has a feasible construction for every finite risk curve: positive risk can be purchased, negative risk spilled, and SOC held constant with zero battery flows. This statement uses the model's unbounded purchase/spill variables and is not a grid-capacity claim. Both realized archives pass source-demand, tariff, battery-physics, continuity and settlement checks. Their intended LP flows also pass independent SOC/power/energy checks and happen to have no simultaneous charge/discharge. A continuous LP without binary exclusivity does not acquire a universal exclusivity certificate from these observations.

## Why the additional future-mutation checks were necessary

The original test added 60,000 kW to both future forecast channels. Those mutations cancel in load-minus-PV and therefore were weak evidence for the net-input risk computation. The supplement preserves that original test and adds independent future-store-only, future-actual-only and combined tests on days 31, 32, 60, 151, 243 and 364. The new future-store mutation is `[+60000,+20000]` kW and changes future net demand by 40,000 kW. Current issued forecasts, the historical error window and both risk curves remain invariant.

Day 364 has no later issued store rows, so its store-only mutation is explicitly marked vacuous; it is not counted as evidence about nonexistent later forecasts. The separate actual-data mutation remains asymmetric. These are causal information-isolation checks, not an untouched annual test claim.

## Execution changes with the planned state

Although both variants have the same numerical buffer of 500 kWh, the actual discharge floor is `max(1200, intended_next_SOC−500)`. Changing the risk curve changes intended SOC and therefore also changes the feedback floor. The comparison cannot isolate the purchase vector alone.

| Floor diagnostic at the recorded actual SOC | Prefix | Point |
|---|---:|---:|
| Deficit slots where the floor withholds discharge | 2,214 | 599 |
| Conditional withheld demand (kWh) | 158,346.862634 | 45,965.946132 |
| Conditional gross emergency-fee proxy (yuan) | 799,604.496265 | 225,682.211552 |

These are one-step conditional quantities. Removing a floor changes future SOC, so the fee proxies are neither feasible annual savings nor additive causal benefits. The supplemental NPZ files preserve the floor, reconstructed planned spill and conditional quantities at every interval. Independently rebuilding the executed discharge from the recorded SOC, floor and deficit has zero error.

## Coverage and the fixed-purchase lower bounds

The marginal prefix curve covers 78.99% of realized cumulative time points, but only 32.63% of entire daily realized paths. Its historical whole-path coverage is 33.34%. Pointwise-risk accumulation covers 94.34% of cumulative time points and 64.67% of daily paths. None of these empirical, repeatedly inspected development-year observations is a joint chance guarantee. Neither curve explicitly resolves conversion losses and SOC saturation in the probabilistic coverage statement.

| Fixed-Q perfect-future diagnostic | Prefix | Point |
|---|---:|---:|
| Physical LP lower bound for actual annual fee (yuan) | 13,352,142.228979 | 13,094,574.535435 |
| Maximum possible execution-only reduction (yuan) | 354,097.358846 | 96,150.140787 |
| Lower bound above the 8% target (yuan) | 411,185.349879 | 153,617.656335 |

The bound optimistically gives battery execution all future realized demand, while fixing the original purchase vectors, initial SOC and hardware constraints. Primal and dual objectives agree. Even these optimistic bounds exceed the 8% target, so execution-only changes cannot make either fixed purchase archive achieve that target. Future-informed oracle actions were not exported or offered as a strategy. A subsequent useful attempt must alter purchase planning, not claim the floor-fee proxy as an available annual saving.
