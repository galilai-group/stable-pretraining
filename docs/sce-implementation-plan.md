# SCE — Similarity Contrastive Estimation

This document outlines the planned implementation of Similarity Contrastive Estimation (SCE) in stable-pretraining.

**Paper:** https://arxiv.org/abs/2111.14585

**Reference implementation:** https://github.com/CEA-LIST/SCE

## Planned implementation

- Implement the SCE loss with soft relational similarity targets
- Integrate the online encoder, momentum encoder, and representation queue
- Support the existing stable-pretraining training interface
- Add unit and integration testing
- Validate the method using CIFAR-10
- Include reproducible experimental results and documentation

## Status

Work in progress