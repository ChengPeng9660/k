# Reproduction Notes

## What is in the package

- The archive contains model scripts, bundled logs, and a README PDF.
- The README states that the empirical e-commerce data used in the paper cannot be shared because of an NDA.
- The feature construction pipeline expects three raw input files:
  - `Footprints.csv`
  - `experiment_list.csv`
  - `experiment.csv`
- The model scripts expect feature files such as `training_feature_batch.pkl` and `testing_feature_batch.pkl` when empirical data is available.

## What was verified locally

- A batch runner was added: `run_reproduction.py`
- The model scripts were updated so `Episode_Horizon` can be set through the `EPISODE_HORIZON` environment variable without editing the source each time.
- Local dependencies were checked with Python 3.8.
- `sparse` and `xgboost` were installed with `pip`.

## Main reproduction blockers

### 1. Exact paper results are not reproducible from the public package alone

The empirical data used for the paper is not included in the archive, and the README explicitly says it cannot be shared.

### 2. The public scripts and bundled logs are not fully consistent

In the public scripts, the simulated-data branch usually fills `memory` only and does not populate `test_memory`.
That means the final rejection-sampling evaluation block does not run when empirical test data is absent.

Examples:

- `MAB.py`: simulated branch fills `memory` only, but evaluation requires `len(test_memory) > 0`
- `XgBoost.py`: same pattern
- `DQN.py`: same pattern
- `DRQN.py`: same pattern
- The bundled logs do include final evaluation metrics, which implies those logs were produced under a different data condition, most likely with empirical test data present

### 3. XGBoost currently has a platform runtime issue

On this machine, `xgboost` fails to load because `libomp.dylib` is missing:

- `XGBoost Library (libxgboost.dylib) could not be loaded`
- `Library not loaded: @rpath/libomp.dylib`

## Files added or adjusted

- `run_reproduction.py`
- `REPRODUCTION_NOTES.md`
- `BRQN.py`
- `BDQN.py`
- `DRQN.py`
- `DQN.py`
- `DNN_BayesianUpdate.py`
- `RNN_BayesianUpdate.py`
- `MAB.py`
- `XgBoost.py`

## How to run

Example:

```bash
python3 run_reproduction.py --models DQN BRQN --horizons 1 4 8
```

This writes logs and a CSV summary to:

- `reproduction_outputs/`

## What would be needed for a real end-to-end reproduction

One of these:

1. The original empirical feature files:
   - `training_feature_batch.pkl`
   - `testing_feature_batch.pkl`
2. The raw clickstream and experiment files so `feature_building.py` can rebuild them
3. A corrected public simulation/evaluation pipeline from the authors that reproduces the bundled logs without NDA data

