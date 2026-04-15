"""Frank-lab-style choice-model parameterizations from `SpatialBanditTask`.

Faithful-in-spirit Python ports of three model families used in
`LorenFrankLab/SpatialBanditTask` (Julia), each fit by SGD on a single
session for direct comparison with the SSP models in notebook 01.

Two simplifications relative to the published code:
- **Flat 6-way well softmax.** The published models factor each trial as
  a 3-way stem softmax + (on stem-switch) a 2-way leaf softmax. We use
  P(well | history) directly so the per-trial log-likelihood is on the
  same scale as `MultinomialChoiceModel` / `CovariateChoiceModel`.
- **Single-session MLE.** No hierarchical EM across sessions / animals.
  We fit by SGD just like SSP's `.fit_sgd`.

Models:
- :class:`QLearnerModel` — per-well Q[w]; EWMA update on chosen well.
- :class:`BetaBernoulliModel` — per-well Beta posterior; the published
  "beta model" (Frank-lab metalearning paper).
- :class:`FrankHMMModel` — discrete latent over a fixed catalog of
  reward-prob configurations; volatility-driven transitions.

All three optionally include a `stay_bias` added to the previous well's
logit (perseveration), mirroring the Frank-lab parameterization. They do
NOT include the published model's stem-level βgo/βstay split or the
turn/spatial biases — those live in the published *action* hierarchy and
are absorbed here into a single softmax temperature + stay_bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import optax


def _stay_bias_vec(n_wells: int, prev_well: jnp.ndarray, stay_bias: jnp.ndarray, has_prev: jnp.ndarray) -> jnp.ndarray:
    """One-hot vector that adds `stay_bias` to `prev_well` when `has_prev`."""
    onehot = jax.nn.one_hot(prev_well, n_wells)
    return onehot * stay_bias * has_prev


def _per_trial_logp(values: jnp.ndarray, beta_temp: jnp.ndarray, prev_well: jnp.ndarray, stay_bias: jnp.ndarray, has_prev: jnp.ndarray) -> jnp.ndarray:
    """6-way softmax with optional stay-bias on the previous well."""
    n_wells = values.shape[0]
    logits = beta_temp * values + _stay_bias_vec(n_wells, prev_well, stay_bias, has_prev)
    return jax.nn.log_softmax(logits)


# ----------------------------------------------------------------------
# Q-learner
# ----------------------------------------------------------------------


def _q_learner_loglik(params: dict, choices: jnp.ndarray, rewards: jnp.ndarray) -> jnp.ndarray:
    n_wells = 6
    n_trials = choices.shape[0]
    alpha = jax.nn.sigmoid(params["alpha_logit"])
    beta_temp = jnp.exp(params["log_beta"])
    init_Q = jax.nn.sigmoid(params["init_Q_logit"])
    stay_bias = params["stay_bias"]

    Q0 = jnp.full(n_wells, init_Q)
    has_prev = jnp.concatenate([jnp.zeros(1), jnp.ones(n_trials - 1)])
    prev_well_seq = jnp.concatenate([jnp.zeros(1, dtype=choices.dtype), choices[:-1]])

    def step(Q, x):
        well_t, reward_t, has_prev_t, prev_well_t = x
        log_p = _per_trial_logp(Q, beta_temp, prev_well_t, stay_bias, has_prev_t)
        ll_t = log_p[well_t]
        Q_new = Q.at[well_t].add(alpha * (reward_t - Q[well_t]))
        return Q_new, ll_t

    xs = (choices, rewards.astype(jnp.float32), has_prev, prev_well_seq)
    _, lls = jax.lax.scan(step, Q0, xs)
    return lls.sum()


@dataclass
class QLearnerModel:
    """Q-learner with EWMA value update and single-temperature softmax.

    Free parameters
    ---------------
    alpha : learning rate, sigmoid-transformed.
    beta_temp : softmax inverse temperature, log-transformed.
    init_Q : initial Q value (shared across wells), sigmoid-transformed.
    stay_bias : real-valued perseveration bias on previous well.
    """

    n_wells: int = 6
    params_: dict = field(default_factory=dict)
    log_likelihoods_: np.ndarray | None = None

    def fit_sgd(
        self,
        choices: jnp.ndarray,
        rewards: jnp.ndarray,
        num_steps: int = 500,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> np.ndarray:
        params = {
            "alpha_logit": jnp.array(0.0),  # sigmoid(0) = 0.5
            "log_beta": jnp.log(jnp.array(2.0)),
            "init_Q_logit": jnp.array(0.0),
            "stay_bias": jnp.array(0.0),
        }
        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            ll, grads = jax.value_and_grad(_q_learner_loglik)(params, choices, rewards)
            updates, opt_state = opt.update(jax.tree_util.tree_map(lambda g: -g, grads), opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, ll

        lls = np.empty(num_steps, dtype=np.float64)
        for i in range(num_steps):
            params, opt_state, ll = step(params, opt_state)
            lls[i] = float(ll)
            if verbose and (i % 50 == 0 or i == num_steps - 1):
                print(f"  q-learner step {i}: ll = {ll:.2f}")
        self.params_ = params
        self.log_likelihoods_ = lls
        return lls

    @property
    def alpha_(self) -> float:
        return float(jax.nn.sigmoid(self.params_["alpha_logit"]))

    @property
    def beta_(self) -> float:
        return float(jnp.exp(self.params_["log_beta"]))

    @property
    def init_Q_(self) -> float:
        return float(jax.nn.sigmoid(self.params_["init_Q_logit"]))

    @property
    def stay_bias_(self) -> float:
        return float(self.params_["stay_bias"])

    def trajectories(self, choices: jnp.ndarray, rewards: jnp.ndarray) -> dict:
        """Return per-trial Q values, full log-probabilities, and chosen log-prob."""
        n_trials = choices.shape[0]
        alpha = self.alpha_
        beta_temp = self.beta_
        init_Q = self.init_Q_
        stay_bias = self.stay_bias_
        Q = np.full(self.n_wells, init_Q, dtype=np.float64)
        Qs = np.zeros((n_trials, self.n_wells))
        log_p = np.zeros((n_trials, self.n_wells))
        log_p_chosen = np.zeros(n_trials)
        for t in range(n_trials):
            has_prev = float(t > 0)
            prev_well = int(choices[t - 1]) if t > 0 else 0
            logits = beta_temp * Q
            if has_prev:
                logits = logits.copy()
                logits[prev_well] += stay_bias
            lp = logits - np.log(np.exp(logits).sum())
            log_p[t] = lp
            log_p_chosen[t] = lp[int(choices[t])]
            Qs[t] = Q
            w = int(choices[t])
            r = float(rewards[t])
            Q[w] += alpha * (r - Q[w])
        return {"Q": Qs, "log_p": log_p, "log_p_chosen": log_p_chosen}


# ----------------------------------------------------------------------
# Beta-Bernoulli (the published "beta model")
# ----------------------------------------------------------------------


def _beta_bernoulli_loglik(params: dict, choices: jnp.ndarray, rewards: jnp.ndarray) -> jnp.ndarray:
    n_wells = 6
    n_trials = choices.shape[0]
    beta_temp = jnp.exp(params["log_beta"])
    stay_bias = params["stay_bias"]
    a_baseline = jnp.exp(params["log_a_baseline"])
    decay = jax.nn.sigmoid(params["decay_logit"])

    has_prev = jnp.concatenate([jnp.zeros(1), jnp.ones(n_trials - 1)])
    prev_well_seq = jnp.concatenate([jnp.zeros(1, dtype=choices.dtype), choices[:-1]])

    def step(ab, x):
        well_t, reward_t, has_prev_t, prev_well_t = x
        Q = ab[:, 0] / (ab[:, 0] + ab[:, 1])
        log_p = _per_trial_logp(Q, beta_temp, prev_well_t, stay_bias, has_prev_t)
        ll_t = log_p[well_t]
        # Decay all wells toward (a_baseline, a_baseline) (uniform Beta).
        ab_decayed = (ab - a_baseline) * (1 - decay) + a_baseline
        # Update chosen well: a += r, b += (1-r).
        delta_a = jnp.zeros(n_wells).at[well_t].set(reward_t)
        delta_b = jnp.zeros(n_wells).at[well_t].set(1.0 - reward_t)
        ab_new = ab_decayed + jnp.stack([delta_a, delta_b], axis=-1)
        return ab_new, ll_t

    ab0 = jnp.ones((n_wells, 2)) * a_baseline
    xs = (choices, rewards.astype(jnp.float32), has_prev, prev_well_seq)
    _, lls = jax.lax.scan(step, ab0, xs)
    return lls.sum()


@dataclass
class BetaBernoulliModel:
    """Beta-Bernoulli per-well belief model — the published "beta model".

    Free parameters
    ---------------
    beta_temp : softmax inverse temperature, log-transformed.
    stay_bias : real-valued perseveration bias on previous well.
    a_baseline : prior pseudo-count for both alpha and beta (>= 0,
        log-transformed). Defaults free; the published model fixes it to 1.
    decay : per-trial pull toward baseline (Frank-lab `beta_decay`),
        sigmoid-transformed.

    Notes
    -----
    With `decay` near 0 the model accumulates indefinitely (counts grow
    unboundedly across trials and posterior becomes peaked).  With `decay`
    near 1 each trial nearly resets to the prior (no learning).
    """

    n_wells: int = 6
    params_: dict = field(default_factory=dict)
    log_likelihoods_: np.ndarray | None = None

    def fit_sgd(
        self,
        choices: jnp.ndarray,
        rewards: jnp.ndarray,
        num_steps: int = 500,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> np.ndarray:
        params = {
            "log_beta": jnp.log(jnp.array(2.0)),
            "stay_bias": jnp.array(0.0),
            "log_a_baseline": jnp.log(jnp.array(1.0)),
            "decay_logit": jnp.array(-3.0),  # sigmoid(-3) ~ 0.05 — small decay
        }
        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            ll, grads = jax.value_and_grad(_beta_bernoulli_loglik)(params, choices, rewards)
            updates, opt_state = opt.update(jax.tree_util.tree_map(lambda g: -g, grads), opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, ll

        lls = np.empty(num_steps, dtype=np.float64)
        for i in range(num_steps):
            params, opt_state, ll = step(params, opt_state)
            lls[i] = float(ll)
            if verbose and (i % 50 == 0 or i == num_steps - 1):
                print(f"  beta-bernoulli step {i}: ll = {ll:.2f}")
        self.params_ = params
        self.log_likelihoods_ = lls
        return lls

    @property
    def beta_(self) -> float:
        return float(jnp.exp(self.params_["log_beta"]))

    @property
    def stay_bias_(self) -> float:
        return float(self.params_["stay_bias"])

    @property
    def a_baseline_(self) -> float:
        return float(jnp.exp(self.params_["log_a_baseline"]))

    @property
    def decay_(self) -> float:
        return float(jax.nn.sigmoid(self.params_["decay_logit"]))

    def trajectories(self, choices: jnp.ndarray, rewards: jnp.ndarray) -> dict:
        n_trials = choices.shape[0]
        beta_temp = self.beta_
        stay_bias = self.stay_bias_
        a_baseline = self.a_baseline_
        decay = self.decay_
        ab = np.ones((self.n_wells, 2)) * a_baseline
        Qs = np.zeros((n_trials, self.n_wells))
        log_p = np.zeros((n_trials, self.n_wells))
        log_p_chosen = np.zeros(n_trials)
        for t in range(n_trials):
            Q = ab[:, 0] / (ab[:, 0] + ab[:, 1])
            logits = beta_temp * Q
            if t > 0:
                logits = logits.copy()
                logits[int(choices[t - 1])] += stay_bias
            lp = logits - np.log(np.exp(logits).sum())
            log_p[t] = lp
            log_p_chosen[t] = lp[int(choices[t])]
            Qs[t] = Q
            ab = (ab - a_baseline) * (1 - decay) + a_baseline
            w = int(choices[t])
            ab[w, 0] += float(rewards[t])
            ab[w, 1] += 1.0 - float(rewards[t])
        return {"Q": Qs, "log_p": log_p, "log_p_chosen": log_p_chosen}


# ----------------------------------------------------------------------
# HMM with fixed contingency catalog
# ----------------------------------------------------------------------


def get_contingency_catalog(reward_probs: tuple[float, ...] = (0.2, 0.2, 0.2, 0.5, 0.5, 0.8)) -> np.ndarray:
    """Generate all unique permutations of a per-well reward-prob multiset.

    Default mirrors the published `get_contingencies(n=3)` catalog: three
    wells at low, two at medium, one at high reward probability. Returns
    a (K, n_wells) array where K = 6!/(3!2!1!) = 60.
    """
    catalog = sorted(set(permutations(reward_probs)))
    return np.array(catalog, dtype=np.float64)


def _frank_hmm_loglik(params: dict, choices: jnp.ndarray, rewards: jnp.ndarray, contingencies: jnp.ndarray) -> jnp.ndarray:
    K, n_wells = contingencies.shape
    n_trials = choices.shape[0]
    volatility = jax.nn.sigmoid(params["volatility_logit"])
    beta_temp = jnp.exp(params["log_beta"])
    stay_bias = params["stay_bias"]

    diag = 1.0 - volatility
    off = volatility / (K - 1)
    T = off * jnp.ones((K, K)) + (diag - off) * jnp.eye(K)

    has_prev = jnp.concatenate([jnp.zeros(1), jnp.ones(n_trials - 1)])
    prev_well_seq = jnp.concatenate([jnp.zeros(1, dtype=choices.dtype), choices[:-1]])

    def step(alpha, x):
        well_t, reward_t, has_prev_t, prev_well_t = x
        alpha_pred = T @ alpha
        Q = contingencies.T @ alpha_pred  # (n_wells,)
        log_p = _per_trial_logp(Q, beta_temp, prev_well_t, stay_bias, has_prev_t)
        ll_t = log_p[well_t]
        emission = jnp.where(reward_t > 0.5, contingencies[:, well_t], 1.0 - contingencies[:, well_t])
        alpha_new = alpha_pred * emission
        alpha_new = alpha_new / alpha_new.sum()
        return alpha_new, ll_t

    alpha0 = jnp.ones(K) / K
    xs = (choices, rewards.astype(jnp.float32), has_prev, prev_well_seq)
    _, lls = jax.lax.scan(step, alpha0, xs)
    return lls.sum()


@dataclass
class FrankHMMModel:
    """HMM over a fixed catalog of per-well reward-probability configurations.

    Each latent state corresponds to one full 6-tuple of per-well reward
    probabilities, drawn from `contingencies`. The model fits only the
    softmax temperature, the stay bias, and the transition volatility —
    NOT the emission matrix. This is what makes the published HMM
    structurally different from `ContingencyBeliefModel` (which fits a
    K=2 emission matrix from scratch).

    Free parameters
    ---------------
    volatility : off-diagonal transition mass, sigmoid-transformed.
    beta_temp : softmax inverse temperature, log-transformed.
    stay_bias : real-valued perseveration bias on previous well.
    """

    contingencies: np.ndarray
    params_: dict = field(default_factory=dict)
    log_likelihoods_: np.ndarray | None = None

    @property
    def n_wells(self) -> int:
        return self.contingencies.shape[1]

    @property
    def n_states(self) -> int:
        return self.contingencies.shape[0]

    def fit_sgd(
        self,
        choices: jnp.ndarray,
        rewards: jnp.ndarray,
        num_steps: int = 500,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> np.ndarray:
        params = {
            "volatility_logit": jnp.array(-2.0),  # sigmoid(-2) ~ 0.12
            "log_beta": jnp.log(jnp.array(2.0)),
            "stay_bias": jnp.array(0.0),
        }
        contingencies = jnp.asarray(self.contingencies)
        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            ll, grads = jax.value_and_grad(_frank_hmm_loglik)(params, choices, rewards, contingencies)
            updates, opt_state = opt.update(jax.tree_util.tree_map(lambda g: -g, grads), opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, ll

        lls = np.empty(num_steps, dtype=np.float64)
        for i in range(num_steps):
            params, opt_state, ll = step(params, opt_state)
            lls[i] = float(ll)
            if verbose and (i % 50 == 0 or i == num_steps - 1):
                print(f"  frank-hmm step {i}: ll = {ll:.2f}")
        self.params_ = params
        self.log_likelihoods_ = lls
        return lls

    @property
    def volatility_(self) -> float:
        return float(jax.nn.sigmoid(self.params_["volatility_logit"]))

    @property
    def beta_(self) -> float:
        return float(jnp.exp(self.params_["log_beta"]))

    @property
    def stay_bias_(self) -> float:
        return float(self.params_["stay_bias"])

    def trajectories(self, choices: jnp.ndarray, rewards: jnp.ndarray) -> dict:
        n_trials = choices.shape[0]
        K = self.n_states
        volatility = self.volatility_
        beta_temp = self.beta_
        stay_bias = self.stay_bias_
        diag = 1.0 - volatility
        off = volatility / (K - 1)
        T = off * np.ones((K, K)) + (diag - off) * np.eye(K)
        alpha = np.ones(K) / K
        Qs = np.zeros((n_trials, self.n_wells))
        log_p = np.zeros((n_trials, self.n_wells))
        log_p_chosen = np.zeros(n_trials)
        alphas = np.zeros((n_trials, K))
        for t in range(n_trials):
            alpha = T @ alpha
            Q = self.contingencies.T @ alpha
            logits = beta_temp * Q
            if t > 0:
                logits = logits.copy()
                logits[int(choices[t - 1])] += stay_bias
            lp = logits - np.log(np.exp(logits).sum())
            log_p[t] = lp
            log_p_chosen[t] = lp[int(choices[t])]
            Qs[t] = Q
            alphas[t] = alpha
            w = int(choices[t])
            r = float(rewards[t])
            emission = self.contingencies[:, w] if r > 0.5 else (1.0 - self.contingencies[:, w])
            alpha = alpha * emission
            alpha = alpha / alpha.sum()
        return {"Q": Qs, "alpha": alphas, "log_p": log_p, "log_p_chosen": log_p_chosen}
