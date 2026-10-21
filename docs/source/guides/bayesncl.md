# BayesNCL

Status: implementation in progress.

Paper: [Bayesian Gated Non-Negative Contrastive Learning](https://arxiv.org/abs/2605.28441)

Reference: [Official implementation](https://github.com/Peng-Cui-2003/BayesNCL)

BayesNCL extends contrastive learning with nonnegative projected
representations, a learned feature gate, and a Bernoulli KL penalty.

## Planned implementation

- Reuse the existing encoder and contrastive learning infrastructure.
- Add a gating head operating on detached backbone features.
- Implement straight-through binary gating and Bernoulli KL regularization.
- Preserve the reference activation and normalization behavior.
- Provide a BayesNCL method and composable forward function.
- Add tests for reference agreement, gradients, and numerical edge cases.
- Include a CIFAR training example and document validation results.