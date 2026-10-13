"""Offline probabilistic credit for GraphGPO-compatible WebShop batches."""

from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple
import numpy as np

StateAction = Tuple[str, str]
TERMINAL_STATES = ("url:SUCCESS", "text:success")


def canonical_state(state, url=None):
    """Use URL as the stable identity when available; text is an explicit fallback."""
    if url and str(url) not in ("SUCCESS", "success"):
        return "url:" + str(url)
    return "text:" + str(state)


def scoped_state(uid, state):
    """Namespace a state by task/prompt to prevent cross-task evidence leakage."""
    return "uid:" + str(uid) + "|" + str(state)


def transition_rows(non_tensor):
    """Return validated rollout rows; fail loudly instead of zip truncation."""
    keys = list(non_tensor["anchor_obs"])
    urls = list(non_tensor.get("anchor_url", [None] * len(keys)))
    nxt = list(non_tensor["next_obs"])
    nurls = list(non_tensor.get("next_url", [None] * len(keys)))
    acts = list(non_tensor["text_actions"])
    uids = list(non_tensor.get("uid", ["unknown"] * len(keys)))
    lengths = {len(keys), len(urls), len(nxt), len(nurls), len(acts), len(uids)}
    if len(lengths) != 1:
        raise ValueError("rollout transition fields have inconsistent lengths")
    return list(zip(uids, keys, urls, acts, nxt, nurls))


@dataclass
class TransitionPosterior:
    n_obs: Dict[StateAction, Counter]
    n_intervention: Dict[StateAction, Counter]
    alpha: float = 0.5

    def counts(self, source="pooled"):
        if source == "obs":
            return self.n_obs
        if source == "intervention":
            return self.n_intervention
        out = defaultdict(Counter)
        for m in (self.n_obs, self.n_intervention):
            for k, v in m.items():
                out[k].update(v)
        return out

    def probabilities(self, state, action, source="pooled"):
        c = self.counts(source).get((state, action), Counter())
        if not c:
            return {}
        den = sum(c.values()) + self.alpha * len(c)
        return {s: (v + self.alpha) / den for s, v in c.items()}


def build_posterior(non_tensor, intervention_mask=None, scope_by_uid=False):
    nobs = defaultdict(Counter)
    nint = defaultdict(Counter)
    rows = transition_rows(non_tensor)
    if intervention_mask is not None and len(intervention_mask) != len(rows):
        raise ValueError("intervention_mask length must match rollout rows")
    for i, (uid, s, u, a, t, nu) in enumerate(rows):
        src = (
            "intervention"
            if intervention_mask is not None and bool(intervention_mask[i])
            else "obs"
        )
        anchor = canonical_state(s, u)
        successor = canonical_state(t, nu)
        if scope_by_uid:
            anchor, successor = scoped_state(uid, anchor), scoped_state(uid, successor)
        (nint if src == "intervention" else nobs)[(anchor, str(a))][successor] += 1
    return TransitionPosterior(nobs, nint)


def q_values(
    posterior, state, actions, terminal_states=TERMINAL_STATES, horizon=8, continuation=None
):
    """Finite-horizon Q with terminal value 1 and no pseudo-success mass."""
    v = {str(s): float(s in terminal_states) for s in terminal_states}
    for _ in range(max(1, int(horizon))):
        nv = dict(v)
        for (s, a), dist in posterior.counts().items():
            nv[s] = max(
                nv.get(s, 0.0),
                sum(prob * v.get(t, 0.0) for t, prob in posterior.probabilities(s, a).items()),
            )
        v = nv
    return {
        str(a): sum(prob * v.get(t, 0.0) for t, prob in posterior.probabilities(state, a).items())
        for a in actions
    }


def all_q_values(posterior, terminal_states=TERMINAL_STATES, horizon=8):
    """Compute every observed Q(s,a) in one DP pass."""
    table = posterior.counts()
    v = {s: 1.0 for s in terminal_states}
    q = {}
    for _ in range(max(1, int(horizon))):
        q = {
            (s, a): sum(prob * v.get(t, 0.0) for t, prob in posterior.probabilities(s, a).items())
            for s, a in table
        }
        grouped = defaultdict(list)
        for (s, _), value in q.items():
            grouped[s].append(value)
        nv = {s: 1.0 for s in terminal_states}
        nv.update({s: max(values) for s, values in grouped.items()})
        v = nv
    return q


def policy_qva_values(posterior, policy=None, terminal_states=TERMINAL_STATES, horizon=8):
    """Compute finite-horizon Q^pi, V^pi and A^pi from transition evidence.

    ``policy`` is an optional mapping ``state -> {action: probability}``.  When
    no policy is supplied, the empirical action support is given a uniform
    policy.  This explicit policy evaluation is kept separate from the legacy
    ``all_q_values`` max-backup used by the offline comparison, so the online
    hook cannot silently turn Q^pi into an optimistic Q* estimate.
    """
    table = posterior.counts()
    actions_by_state = defaultdict(list)
    for state, action in table:
        actions_by_state[state].append(action)
    v = {str(s): 1.0 for s in terminal_states}
    # When online evidence is task-scoped (``uid:<task>|<state>``), preserve
    # the terminal semantics after namespacing instead of treating SUCCESS as
    # an ordinary unseen node.
    for (_, _), dist in table.items():
        for successor in dist:
            successor = str(successor)
            if any(successor.endswith("|" + terminal) for terminal in terminal_states):
                v[successor] = 1.0
    terminal_values = dict(v)
    q = {}
    for _ in range(max(1, int(horizon))):
        q = {
            (s, a): sum(prob * v.get(t, 0.0) for t, prob in posterior.probabilities(s, a).items())
            for s, a in table
        }
        nv = dict(terminal_values)
        for state, actions in actions_by_state.items():
            if policy is None:
                weights = {a: 1.0 / len(actions) for a in actions}
            else:
                raw = {a: max(0.0, float(policy.get(state, {}).get(a, 0.0))) for a in actions}
                total = sum(raw.values())
                weights = (
                    {a: value / total for a, value in raw.items()}
                    if total > 0
                    else {a: 1.0 / len(actions) for a in actions}
                )
            nv[state] = sum(weights[a] * q.get((state, a), 0.0) for a in actions)
        v = nv
    advantage = {(s, a): value - v.get(s, 0.0) for (s, a), value in q.items()}
    return q, v, advantage


def posterior_rank_probability(posterior, state, actions, M=500, rng=None):
    rng = rng or np.random.default_rng(0)
    acts = list(actions)
    stats = []
    counts = posterior.counts()
    # The success mass of a Dirichlet posterior has an exact Beta marginal;
    # vectorising this binary marginal keeps M=500 affordable on WebShop.
    for a in acts:
        c = counts.get((state, a), Counter())
        succ = sum(v for k, v in c.items() if k in TERMINAL_STATES)
        fail = sum(c.values()) - succ
        stats.append(rng.beta(succ + posterior.alpha, fail + posterior.alpha, int(M)))
    if len(stats) < 2:
        return 1.0, 0
    order = np.argsort([-x.mean() for x in stats])
    top, second = order[:2]
    return float(np.mean(stats[top] > stats[second])), min(
        sum(posterior.counts().get((state, acts[top]), Counter()).values()),
        sum(posterior.counts().get((state, acts[second]), Counter()).values()),
    )
