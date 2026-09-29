# Testing Guide

## Overview
Tests in stable-pretraining are categorized to separate fast unit tests from slow integration tests. This enables efficient CI/CD while maintaining comprehensive test coverage.

## Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast, no GPU, no downloads
- **Integration Tests** (`@pytest.mark.integration`): Full training runs
- **Regression Tests** (`@pytest.mark.regression`): CPU training on synthetic data, registry checks, and reference losses
- **GPU Tests** (`@pytest.mark.gpu`): Require CUDA
- **Slow Tests** (`@pytest.mark.slow`): Take > 1 minute
- **Download Tests** (`@pytest.mark.download`): Download data

## Running Tests

```bash
# Unit tests only (CI default)
python -m pytest -m unit

# Integration tests
python -m pytest -m integration

# CPU training regression tests
python -m pytest -m regression

# All tests
python -m pytest

# Exclude slow tests
python -m pytest -m "not slow"
```

## Measuring Coverage

Install the development dependencies with `pip install -e ".[dev,trackio,swanlab]"`, then run:

```bash
python -m pytest stable_pretraining/tests -m unit \
  --cov=stable_pretraining --cov-report=term-missing --cov-report=html
```

The terminal shows the overall line coverage percentage and missing lines per
file. Open `htmlcov/index.html` to inspect uncovered code. Configuration in
`pyproject.toml` excludes test files from the percentage.

Use `-m "unit or regression"` to include the CPU training regression suite.
Compare percentages using the same test selection and installed extras. The
whole-package denominator includes the optional JAX backend, even when JAX tests
are skipped because its dependencies are not installed. Line coverage records which
lines executed; assertions must still check correctness, failure handling, and
resume behavior. Skipped GPU tests do not validate GPU behavior.

## Test Organization

```
stable_pretraining/tests/
├── conftest.py              # Pytest configuration
├── utils.py                 # Test utilities and mocks
├── test_*_unit.py          # Unit tests
├── test_*_integration.py   # Integration tests
└── test_*.py               # Original tests
```

## Writing Unit Tests

### Use Test Utilities
```python
from stable_pretraining.tests.utils import MockImageDataset, create_mock_dataloader

@pytest.mark.unit
def test_loss_computation():
    # Test just the loss computation
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss_fn = spt.losses.NTXEntLoss(temperature=0.1)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0
```

### Best Practices

1. **Mock Dependencies**: Use `MockImageDataset` instead of real downloads
2. **Small Tensors**: Use `batch_size=4, image_size=32` for speed
3. **Fast Execution**: Tests should run in < 1 second
4. **Single Responsibility**: Test one component at a time
5. **Proper Markers**: Always mark tests appropriately

### Example Refactoring

**Before** (Integration test):
```python
def test_simclr():
    # Downloads ImageNette, requires GPU, full training
    train_dataset = spt.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    # ... full training loop
```

**After** (Unit test):
```python
@pytest.mark.unit
def test_simclr_loss():
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss_fn = spt.losses.NTXEntLoss(temperature=0.1)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
def test_simclr_training():
    # Original full test with proper markers
```

## CI Configuration

GitHub Actions runs PyTorch unit tests in four shards and JAX unit tests in a
separate CPU job with two host devices. Integration and regression tests have
their own jobs. A dependent coverage job requires all four `.coverage.<shard>`
artifacts plus `.coverage.jax`, combines them, and requires at least
**95% production line coverage**. The threshold applies to the combined suite,
not to individual shards or either backend alone. Relative coverage paths allow
data from separate workers to be combined. Codecov receives the combined report.

To check the same threshold locally, include the optional backend:

```bash
pip install -e ".[dev,trackio,swanlab,jax]"
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest stable_pretraining/tests -m unit --cov=stable_pretraining \
  --cov-report=term-missing --cov-fail-under=95
```

To reproduce a unit run locally:
```yaml
- name: Run Unit Tests
  run: python -m pytest stable_pretraining/ -m unit --verbose --cov=stable_pretraining --cov-report term
```
