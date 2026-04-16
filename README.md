# replication

Replication project for:

Song, Y., & Sun, T. (2024). *Ensemble experiments to optimize interventions along the customer journey: A reinforcement learning approach*. Management Science, 70(8), 5115-5130.

## Project Metadata

- Date: Mar 31
- Group size: 8
- Type: test
- Students: Mingxia Li, Cheng Peng

## What This Repository Contains

- `code/`: cleaned replication scripts and synthetic-data tooling
- `docs/`: reproduction notes, summaries, and synthetic-environment guide
- `results/`: selected result tables and benchmark CSVs
- `meta/`: project metadata

## Scope

This repository packages our replication workflow around the public replication code and our follow-up synthetic-data experiments.

It includes:

- cleaned model scripts
- synthetic data generators
- benchmark search helpers
- reproduction notes
- selected result summaries

It does **not** include:

- proprietary empirical data from the original paper
- large generated `.pkl` artifacts
- full intermediate log dumps

## Main Takeaway

Our main finding is that the public package is runnable and useful for mechanism-oriented replication, but exact empirical replication is blocked by unavailable proprietary data.

We also explored targeted synthetic environments to study when BRQN can outperform baseline methods under longer horizons.

## Quick Start

Run a basic synthetic benchmark search:

```bash
cd code
python3 search_brqn_setup.py \
  --mode brqn_user_mapping \
  --horizons 4 8 \
  --models MAB XgBoost DQN BRQN \
  --train-users 140 \
  --test-users 90 \
  --min-steps 24 \
  --max-steps 36 \
  --seed 0 \
  --num-episodes 50 \
  --outdir ../results/tmp_run
```

## Notes

- The original paper's empirical data are not included in the public package.
- Some scripts were adjusted during replication to support cleaner horizon control and synthetic-data search.
- See `docs/REPRODUCTION_NOTES.md` for the main blockers and `docs/SYNTHETIC_BRQN_GUIDE.md` for the synthetic-environment logic.
