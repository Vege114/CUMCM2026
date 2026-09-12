"""Midnight-only policy observations and hidden historical execution rewards.

One episode constructs all 144 purchases before any real-day replay. Policy
observations depend on forecasts, tariffs and INTENDED SOC only. Historical
actuals are used after planning to score completed training episodes, never
fed back as policy observations. This is a partially observed planning task.
"""

from dataclasses import dataclass

import numpy as np

from experiments.problem2.tree_planning.model import (
    ETA,
    MAX_SOC,
    MIN_SOC,
    POWER_ENERGY,
    Config,
    execute_plan,
)
from experiments.problem2.tree_planning.risk import TreeResidualScenarios, periodic_baseline

STEPS = 144
BATTERY_LEVELS = np.linspace(-1, 1, 9)
PURCHASE_LEVELS = np.array([-1, -.5, 0, .5, 1])
ACTION_DIM = 45
OBS_DIM = 29


@dataclass(frozen=True)
class RewardConfig:
    throughput_yuan_per_kwh: float = .002
    reversal_yuan: float = .05
    ramp_yuan_per_kw: float = .0002
    deadband_kwh: float = 2.0
    reward_scale_yuan: float = 1000.0
    terminal_value: bool = True


DEFAULT_REWARD_CONFIG = RewardConfig()


class CausalDayCache:
    """Construct features as issued, and reveal labels only when complete."""

    def __init__(self, data, store):
        self.data, self.store = data, store
        self.risk = TreeResidualScenarios(
            store.origins, store.values, data.actual, data.fixed_price,
        )
        self._issued = {}

    def issued(self, day):
        if not 8 <= day < 365:
            raise ValueError("Supported issue days are 8..364")
        if day not in self._issued:
            if day >= 31:
                forecast = self.store.get(day * STEPS)
                supports, audit = self.risk.for_day(day)
            else:
                # January has no frozen exp004 forecasts. Recreate strictly
                # past-only periodic forecasts and error calibration explicitly.
                forecasts, residuals = [], []
                for past in range(1, day + 1):
                    yesterday = self.data.actual[(past-1)*STEPS:past*STEPS]
                    weekly = (self.data.actual[(past-7)*STEPS:(past-6)*STEPS]
                              if past >= 7 else None)
                    predicted = periodic_baseline(yesterday, weekly)
                    forecasts.append(predicted)
                    if past < day:
                        observed = self.data.actual[past*STEPS:(past+1)*STEPS]
                        residuals.append((observed[:, 0] - observed[:, 1]
                                          - predicted[:, 0] + predicted[:, 1]) / 6)
                forecast = forecasts[-1]
                errors = np.quantile(np.stack(residuals),
                                     (np.arange(9) + .5)/9, axis=0).T
                supports = (forecast[:, 0] - forecast[:, 1])[:, None]/6 + errors
                audit = {"day": day, "information_cutoff": day*STEPS,
                         "max_observed_index": day*STEPS-1,
                         "training_origins": list(range(STEPS, day*STEPS, STEPS)),
                         "residual_source": "january_periodic_baseline_per_slot",
                         "fallback": True}
            self._issued[day] = (forecast, supports, audit)
        return self._issued[day]

    def training_days(self, cutoff_day, lookback=90):
        days = list(range(max(8, cutoff_day-lookback), cutoff_day))
        if not days or not 9 <= cutoff_day <= 365 or lookback < 1:
            raise ValueError("No complete causal training days")
        examples = []
        for day in days:
            forecast, supports, audit = self.issued(day)
            # Exclusive midnight cut, before materializing any actual labels.
            if (day+1)*STEPS > cutoff_day*STEPS:
                raise ValueError("Unrevealed training labels")
            examples.append({"day": day, "forecast": forecast, "supports": supports,
                             "actual": self.data.actual[day*STEPS:(day+1)*STEPS].copy(),
                             "audit": audit})
        return examples


class PlanningBatch:
    """Vectorized deterministic planning transitions; no actual array exists here."""

    def __init__(self, forecasts, supports, price, initial_soc, initial_mode=None,
                 initial_power=None, deadband_kwh=2.0):
        self.forecasts = np.asarray(forecasts, float)
        self.supports = np.asarray(supports, float)
        self.price = np.asarray(price, float)
        self.soc = np.asarray(initial_soc, float).copy()
        self.n = len(self.soc)
        if (self.forecasts.shape != (self.n, STEPS, 2)
                or self.supports.shape != (self.n, STEPS, 9)
                or self.price.shape != (STEPS,)
                or not np.isfinite(self.forecasts).all()
                or not np.isfinite(self.supports).all()
                or not np.isfinite(self.price).all() or np.any(self.price <= 0)
                or not np.isfinite(self.soc).all()
                or np.any((self.soc < MIN_SOC) | (self.soc > MAX_SOC))):
            raise ValueError("Invalid planning inputs")
        self.mode = (np.zeros(self.n) if initial_mode is None
                     else np.broadcast_to(initial_mode, (self.n,)).astype(float).copy())
        self.previous_power = (np.zeros(self.n) if initial_power is None
                               else np.broadcast_to(initial_power, (self.n,)).astype(float).copy())
        self.deadband = deadband_kwh
        self.net = (self.forecasts[:, :, 0]-self.forecasts[:, :, 1])/6
        self.q80 = np.quantile(self.supports, .8, axis=2, method="inverted_cdf")
        self.spread = np.maximum(50, self.supports.std(axis=2))
        self.t = 0
        self.purchase = np.zeros((self.n, STEPS))
        self.charge = np.zeros_like(self.purchase)
        self.discharge = np.zeros_like(self.purchase)
        self.states = np.zeros((self.n, STEPS+1))
        self.states[:, 0] = self.soc

    def observe(self):
        t = self.t
        if t >= STEPS:
            raise ValueError("Episode already complete")
        columns = [np.full(self.n, np.sin(2*np.pi*t/STEPS)),
                   np.full(self.n, np.cos(2*np.pi*t/STEPS)),
                   np.full(self.n, t/STEPS), (self.soc-MIN_SOC)/(MAX_SOC-MIN_SOC),
                   self.mode, self.previous_power/5000,
                   self.net[:, t]/2000, self.forecasts[:, t, 1]/12000,
                   self.q80[:, t]/2000, self.spread[:, t]/1000,
                   np.full(self.n, self.price[t]/self.price.max()),
                   self.net[:, t:].mean(axis=1)/2000,
                   np.full(self.n, self.price[t:].mean()/self.price.max())]
        # Known future two-hour blocks; no realized SOC, net-load or reward.
        for k in range(8):
            start, stop = min(t+k*12, 143), min(t+(k+1)*12, 144)
            columns.extend((self.net[:, start:stop].mean(axis=1)/2000,
                            np.full(self.n, self.price[start:stop].mean()/self.price.max())))
        return np.column_stack(columns).astype(np.float32)

    def step(self, actions):
        actions = np.asarray(actions)
        if (self.t >= STEPS or actions.shape != (self.n,)
                or actions.dtype.kind not in "iu" or np.any((actions < 0) | (actions >= 45))):
            raise ValueError("Invalid action")
        battery = BATTERY_LEVELS[actions//5]*POWER_ENERGY
        charge = np.minimum(np.maximum(battery, 0), (MAX_SOC-self.soc)/ETA)
        discharge = np.minimum(np.maximum(-battery, 0), (self.soc-MIN_SOC)*ETA)
        charge = np.where(charge < self.deadband, 0, charge)
        discharge = np.where(discharge < self.deadband, 0, discharge)
        self.purchase[:, self.t] = np.maximum(
            0, self.q80[:, self.t] + charge-discharge
            + PURCHASE_LEVELS[actions % 5]*self.spread[:, self.t],
        )
        self.charge[:, self.t], self.discharge[:, self.t] = charge, discharge
        self.soc += ETA*charge-discharge/ETA
        self.states[:, self.t+1] = self.soc
        self.mode = np.where(charge+discharge > 0, np.sign(charge-discharge), self.mode)
        self.previous_power = 6*(charge-discharge)
        self.t += 1

    def plans(self):
        if self.t != STEPS:
            raise ValueError("Must lock the complete day before execution")
        return [{"purchase": self.purchase[i].copy(), "charge": self.charge[i].copy(),
                 "discharge": self.discharge[i].copy(), "states": self.states[i].copy()}
                for i in range(self.n)]


def score_plans(plans, actuals, price, config=DEFAULT_REWARD_CONFIG, initial_modes=None,
                initial_powers=None, final_days=None):
    """Hidden real execution only after all purchases have been locked."""
    rewards, details = [], []
    n = len(plans)
    modes = np.zeros(n, int) if initial_modes is None else initial_modes
    powers = np.zeros(n) if initial_powers is None else initial_powers
    finals = np.zeros(n, bool) if final_days is None else final_days
    for i, (plan, actual) in enumerate(zip(plans, actuals, strict=True)):
        detail, _ = execute_plan(plan, actual, price, plan["states"][0],
                                 Config(deadband_kwh=config.deadband_kwh), int(modes[i]))
        c, d = detail["charge"], detail["discharge"]
        power = 6*(c-d)
        previous = int(modes[i])
        reversals = np.zeros(STEPS)
        for t, mode in enumerate(np.sign(c-d).astype(int)):
            if mode:
                reversals[t] = previous*mode == -1
                previous = mode
        penalties = (config.throughput_yuan_per_kwh*(c+d)
                     + config.reversal_yuan*reversals
                     + config.ramp_yuan_per_kw*np.abs(np.diff(np.r_[powers[i], power])))
        reward = -(detail["fees"].sum(axis=1)+penalties)/config.reward_scale_yuan
        if config.terminal_value and not finals[i]:
            reward[-1] += (float(np.min(price))/ETA
                           * (detail["states"][-1]-detail["states"][0])
                           / config.reward_scale_yuan)
        rewards.append(reward)
        details.append(detail)
    return np.stack(rewards, axis=1).astype(np.float32), details


def collect_rollout(agent, examples, price, rng, n_envs=32, config=DEFAULT_REWARD_CONFIG):
    selected = [examples[i] for i in rng.integers(len(examples), size=n_envs)]
    initial = rng.uniform(MIN_SOC, MAX_SOC, size=n_envs)
    # Include the exact common February boundary in each training batch.
    initial[0] = 1421.7991105135516
    env = PlanningBatch(np.stack([x["forecast"] for x in selected]),
                        np.stack([x["supports"] for x in selected]), price, initial,
                        deadband_kwh=config.deadband_kwh)
    obs, actions, logs, values = [], [], [], []
    for _ in range(STEPS):
        observation = env.observe()
        action, logprob, value = agent.act(observation)
        obs.append(observation)
        actions.append(action)
        logs.append(logprob)
        values.append(value)
        env.step(action)
    rewards, details = score_plans(env.plans(), [x["actual"] for x in selected], price, config)
    return {"obs": np.stack(obs), "actions": np.stack(actions),
            "old_logprob": np.stack(logs), "values": np.stack(values),
            "rewards": rewards}, details


def plan_day(agent, forecast, supports, price, initial_soc, initial_mode=0,
             initial_power=0, config=DEFAULT_REWARD_CONFIG):
    env = PlanningBatch(np.asarray(forecast)[None], np.asarray(supports)[None],
                        price, [initial_soc], [initial_mode], [initial_power],
                        config.deadband_kwh)
    for _ in range(STEPS):
        actions, _, _ = agent.act(env.observe(), deterministic=True)
        env.step(actions)
    return env.plans()[0]
