# Fixed-purchase execution bounds for four preselected quantiles

The four sources were fixed in advance by name: `lp_q0.50_greedy`,
`lp_q0.60_greedy`, `lp_q0.70_greedy`, and `lp_q0.80_greedy`. They use the
same original CNN/tree forecast family and greedy execution. They were not
selected by their oracle outcomes. This is a diagnostic of existing frozen
purchase trajectories, not a comparison used to choose a deployable strategy.

| Purchase quantile | Frozen plan fee | Minimum emergency fee | Total lower bound | Above 8% reduction target |
|---|---:|---:|---:|---:|
| .50 | 11,724,395.00 | 2,355,951.66 | 14,080,346.66 | 1,139,389.78 |
| .60 | 12,022,685.02 | 1,578,844.27 | 13,601,529.29 | 660,572.41 |
| .70 | 12,344,171.35 | 978,005.04 | 13,322,176.39 | 381,219.51 |
| .80 | 12,714,614.83 | 553,690.39 | 13,268,305.21 | 327,348.34 |

All amounts are yuan over February 1–December 31, 334 days. The target is
12,940,956.879076244 yuan, fixed at 92% of the exp006 primary result.
Even perfect knowledge of future net demand cannot make any of these four
frozen purchase trajectories meet that threshold by changing battery
execution alone. This is not a proof that every causal purchase policy is
unable to meet it.

Reducing the quantile from .80 to .50 cuts plan fees by 990,219.82 yuan but
raises the *minimum* emergency fee by 1,802,261.27 yuan. The total lower bound
worsens by 812,041.45 yuan. Simply lowering the purchase quantile is therefore
not enough; further work must change the timing/shape of purchases and their
interaction with accumulated uncertainty and available storage.

The LP uses the entire actual annual net-demand path, fixes all historical
purchase quantities, and optimizes continuous battery execution. It starts
at 1421.7991105135516 kWh and carries SOC continuously, with 1200–10800 kWh
bounds, 5000 kW maximum charge/discharge and symmetric sqrt(.9) efficiencies.
The sign of net surplus from the fixed purchase structurally prohibits
emergency charging, discharge into surplus, and simultaneous charge/discharge.
Mode, switching, deadband and ramp restrictions are relaxed. Terminal SOC has
the same physical lower bound, with no credited residual inventory value.

The largest primal–dual gap is 8.38e-9 yuan. All four source archives and
oracle solutions pass independent raw-source, physical, SOC-continuity and
billing verification. Maximum SOC-equation error is 1.03e-12 kWh. The original
`data/results/exp008/fixed_purchase_oracle.json` was protected by an unchanged
SHA-256 hash before and after this run.

`comparison.csv` contains all scalar results. Each quantile directory holds
`bound.json`, `source_verification.json` and `oracle_verification.json`.
`constraint_diagnostic.csv` distinguishes the immediate discharge-power
emergency floor from the larger annual LP emergency result. At q=.80, the
immediate floor is only 69,057.68 yuan out of 553,690.39 yuan minimum emergency
fees; the rest arises from the full temporal inventory/charging constraints,
not solely the instantaneous discharge-power cap. This subtraction is a
diagnostic decomposition, not a separate causal attribution experiment.

No future-informed action trace was exported, no oracle actions were reused
by a causal controller, and no new quantiles were explored.

Reproduce with a fresh output directory:
`.venv/bin/python -m experiments.exp008.fixed_purchase_bounds --out PATH`.
The command refuses to overwrite an existing protocol.
