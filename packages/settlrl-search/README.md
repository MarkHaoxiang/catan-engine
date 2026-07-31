# settlrl-search

Re-determinizing Single-Observer ISMCTS over `settlrl-engine`, plus the
engine-only seam primitives the search and its callers share.

## Layout

- `ismcts/` — the search tree package (`make_tree`, `SearchConfig`): a custom
  fixed-capacity tree that determinizes a fresh `sample_world` per simulation
  and descends the live engine, filtering legality per simulation. `config.py`
  holds the configuration, `tree.py` the node store, `descent.py` the walk, and
  `loop.py` the orchestration and `make_tree` factory.
- `__init__.py` — the public wrapper: `make_search` / `make_search_weights` /
  `make_search_weights_value` (the `BeliefPolicy`, the AlphaZero policy
  target, and the policy-plus-root-value target), the `num_simulations=0`
  lookahead special case, and the trade/`num_trees` machinery.
- `expectimax.py` — `make_setup_search`, the compile-efficient beam
  expectimax over the setup phase.
- `_common.py` — shared prior/dice constants and the policy-weights types.
- `priors.py` — `TIER_SCORES`, the action-priority prior shared by the
  search's interior prior and greedy's scoring.
- `value.py` — the `Value` / `ValueFunction` seam types the search evaluates
  a leaf through.
- `policy.py` — the seat protocols (`Policy` / `BeliefPolicy` / `GameAgent` /
  `PolicyPrior` / `ValuePrior`) and the `AgentSpec` registry machinery.
- `sample.py` — `sample_world`: turn a `BeliefView` into one concrete world.
- `rows.py` — the flat action table decoded once (`ROW_TYPE` / `ROW_PARAMS`
  and the host mirrors).
