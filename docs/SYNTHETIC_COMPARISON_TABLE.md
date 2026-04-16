# Synthetic Comparison Table

## Run setting

- Dataset:
  - `train_users=80`
  - `test_users=30`
  - `min_steps=12`
  - `max_steps=16`
- Horizon:
  - `H=1`
- Synthetic feature files:
  - `training_feature_batch.pkl`
  - `testing_feature_batch.pkl`
  - `testing.pkl`

## Model comparison

| Model | mean for all reward | mean for accepted reward | Notes |
| --- | ---: | ---: | --- |
| MAB | 452.8301886792453 | 452.8301886792453 | Ran successfully |
| XgBoost | 467.6190476190476 | 515.9090909090909 | On this machine the script used the sklearn fallback because native `xgboost` could not load `libomp.dylib` |
| DQN | 348.7012987012987 | 364.1975308641975 | Ran successfully |
| DNN_BayesianUpdate | 378.3018867924528 | 378.3018867924528 | Ran successfully |
| BDQN | 514.2076502732241 | 514.2076502732241 | Ran successfully |
| DRQN | 473.1182795698925 | 473.1182795698925 | Ran successfully |
| RNN_BayesianUpdate | 610.7142857142857 | 610.7142857142857 | Ran successfully |
| BRQN | 422.60869565217394 | 423.83720930232556 | Ran successfully |

## Ranking by accepted reward

1. `RNN_BayesianUpdate` - `610.7142857142857`
2. `XgBoost` - `515.9090909090909`
3. `BDQN` - `514.2076502732241`
4. `DRQN` - `473.1182795698925`
5. `MAB` - `452.8301886792453`
6. `BRQN` - `423.83720930232556`
7. `DNN_BayesianUpdate` - `378.3018867924528`
8. `DQN` - `364.1975308641975`

## Log files

- `reproduction_outputs/synth_table/MAB_H1.log`
- `reproduction_outputs/synth_table/XgBoost_H1.log`
- `reproduction_outputs/synth_table/DQN_H1.log`
- `reproduction_outputs/synth_table/DNN_BayesianUpdate_H1.log`
- `reproduction_outputs/synth_table/BDQN_H1.log`
- `reproduction_outputs/synth_table/DRQN_H1.log`
- `reproduction_outputs/synth_table/RNN_BayesianUpdate_H1.log`
- `reproduction_outputs/synth_table/BRQN_H1.log`
