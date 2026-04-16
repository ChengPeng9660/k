# Synthetic Reproduction Summary

## Synthetic files created

- `training_feature_batch.pkl`
- `testing_feature_batch.pkl`
- `testing.pkl`

These are synthetic, model-ready replacements for the missing empirical feature files.

They were generated with:

```bash
python3 generate_synthetic_features.py --train-users 220 --test-users 80 --min-steps 14 --max-steps 24
```

## Structure

Each sample follows the package's expected format:

```python
[
  pre_session_state,
  within_session_pre_state,
  within_session_state,
  action,
  reward,
  page,
  next_page,
  user_id,
]
```

## Synthetic runs completed

### MAB, H=4

Command:

```bash
EPISODE_HORIZON=4 python3 MAB.py
```

Result:

- mean for all reward: `1732.0754716981132`
- mean for accepted reward: `1509.5238095238096`

### DQN, H=4

Command:

```bash
EPISODE_HORIZON=4 NUM_EPISODES=40 BATCH_SIZE=1024 python3 DQN.py
```

Result:

- mean for all reward: `1950.0`
- mean for accepted reward: `1853.8461538461538`

## Notes

- This is a synthetic reproduction of the pipeline, not a reproduction of the paper's original empirical numbers.
- `XgBoost.py` is still blocked on this machine because `libomp.dylib` is missing.
- `DRQN.py` and `BRQN.py` are slower and may need longer runtime or smaller synthetic data / fewer episodes for quick tests.
