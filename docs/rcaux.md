# RC-aux for Latent World Models

## Motivation

During the September 24 guest lecture, I asked whether two states could be close in latent space but difficult to move between, such as when they are separated by a wall, and whether a reachability-aware distance would help with planning.

After the lecture, I found that this question is directly studied in [*Predictive but Not Plannable: RC-aux for Latent World Models*](https://arxiv.org/abs/2605.07278), published at **NeurIPS 2026** by Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, and Miki Haseyama.

The paper proposes RC-aux, a reachability-correction auxiliary objective that teaches a latent world model whether a goal state can be reached within a given action budget.

## Implementation

This contribution implements RC-aux in `stable-pretraining`, including:

- a budget-conditioned reachability head;
- temporal positive and hard-negative pairs;
- multi-horizon latent prediction;
- integration with the existing training interface;
- unit tests and a reproducible experiment.

## Reference

Wenyuan Li, Guang Li, Keisuke Maeda, Takahiro Ogawa, and Miki Haseyama.
“Predictive but Not Plannable: RC-aux for Latent World Models.”
NeurIPS 2026. [Paper](https://arxiv.org/abs/2605.07278) · [Official code](https://github.com/Guang000/RC-aux)
