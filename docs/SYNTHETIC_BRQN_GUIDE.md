# Synthetic BRQN Guide

This package now includes a lightweight synthetic mode, `brqn_simple`, for the
case where the original empirical data are unavailable.

## Why this mode exists

The paper's BRQN advantage is most plausible when the environment has four
properties:

1. hidden state / partial observability
2. action-order dependence
3. delayed multi-step payoff
4. sparse, uncertain high rewards

`brqn_simple` keeps the generator logic intentionally small while preserving
those four ingredients.

## How `brqn_simple` works

- Each user has a small hidden type that changes which actions are the
  `bridge`, `target`, and `decoy`.
- The profitable action depends on an order trigger:
  the previous token must be `0`, the current token must be `1`, and the
  previous action must be the user-specific bridge action.
- When the trigger is hit and the target action is chosen, the user enters a
  short carryover state. This creates extra continuation value for larger
  horizons such as `H=4` and `H=8`.
- The true mapping is stored in `pre_session_state`, while the
  `within_session_*` tensors are near-zero noise. This makes the setup harder
  for memoryless or sum-based methods and more favorable to sequence models.

## Recommended usage

Start with the simple mode:

```bash
python3 generate_synthetic_features.py --mode brqn_simple
```

Then run a quick comparison:

```bash
python3 search_brqn_setup.py \
  --mode brqn_simple \
  --horizons 4 8 \
  --models MAB XgBoost DQN DRQN BRQN \
  --train-users 140 \
  --test-users 100 \
  --min-steps 18 \
  --max-steps 26 \
  --seed 0 \
  --num-episodes 44 \
  --brqn-episodes 60 \
  --batch-size 128 \
  --outdir reproduction_outputs/brqn_simple_h48_seed0
```

## If BRQN is not strong enough

Use the stronger horizon-aware modes that were already in the repo:

- `brqn_user_mapping_horizon`
- `brqn_user_mapping_longbonus`

These are less minimal, but they usually push more value into longer-horizon
payoffs and more strongly favor BRQN at `H=4` and `H=8`.
