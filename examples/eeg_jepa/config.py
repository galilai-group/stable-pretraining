"""Small valid reference configuration and optimizer-step schedules."""

import math
from dataclasses import dataclass


GPU_PREPROCESSING_ID = "gpu_fft_f64_v1"
GPU_PREPROCESSING_CASE = "edf_gpu_assembly+" + GPU_PREPROCESSING_ID


@dataclass(frozen=True)
class Config:
    """Synthetic correctness configuration; not a full pretraining recipe."""

    channels: int = 19
    patches: int = 4
    batch_size: int = 2
    total_steps: int = 4
    lr: float = 5e-4
    final_lr: float = 1e-5
    weight_decay: float = 0.05
    clip_value: float = 1.0
    lr_warmup_steps: int = 1
    context_warmup_steps: int = 2
    vcreg_warmup_steps: int = 2
    lambda_ctx: float = 0.5
    lambda_var: float = 1e-3
    lambda_cov: float = 1e-4
    vcreg_gamma: float = 1.0
    vcreg_sample_size: int = 8192
    ema_start: float = 0.996
    ema_end: float = 1.0

    def model_kwargs(self) -> dict:
        """Keep the reference tokenizer's required 200 samples and width."""
        return dict(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=64,
            seq_len=self.patches,
            n_layer=2,
            nhead=8,
            target_depths=(1,),
            pred_n_layer=1,
            pred_dim=20,
            ch_num=self.channels,
            dropout=0.1,
        )

    def mask_kwargs(self) -> dict:
        """Use Stage-1 N-MET primitives with small-grid minimum token counts."""
        return dict(
            ch_num=self.channels,
            patch_num=self.patches,
            min_keep_enc=2,
            min_keep_pred=2,
            target_support="nmet",
            target_mask_ratio=(0.40, 0.55),
            npred=10,
            pred_mask_scale=(0.06, 0.18),
            aspect_ratio=(0.15, 0.80),
            nmet_base_primitive_probs=(0.50, 0.25, 0.25),
            temporal_mask_scale=(0.25, 0.60),
            temporal_channel_scale=(0.45, 0.90),
            channel_mask_scale=(0.10, 0.35),
            channel_time_scale=(0.60, 1.00),
            full_channel_prob=0.15,
            full_channel_scale=(0.05, 0.16),
            topo_pair_prob=0.10,
            topo_pair_time_scale=(0.50, 1.00),
            full_topo_pair_prob=0.05,
            full_topo_pair_scale=(0.05, 0.20),
        )

    def momentum(self, step: int) -> float:
        """Match the reference's zero-based linear EMA schedule."""
        return self.ema_start + min(1.0, step / max(1, self.total_steps)) * (
            self.ema_end - self.ema_start
        )

    def lr_factor(self, step: int) -> float:
        """Match the reference WarmupCosine scheduler, including its floor."""
        if step < self.lr_warmup_steps:
            return step / max(1, self.lr_warmup_steps)
        progress = (step - self.lr_warmup_steps) / max(
            1, self.total_steps - self.lr_warmup_steps
        )
        return max(self.final_lr / self.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))


def warmup_factor(step: int, steps: int) -> float:
    """Return the reference's linear loss-weight warmup factor."""
    return step / steps if steps > 0 and step < steps else 1.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Repeated measurements with an explicit startup exclusion and worker matrix."""

    trials: int = 3
    workers: tuple[int, ...] = (0, 2)
    startup_steps: int = 2
    diagnostic: bool = False

    def validate(self, steps: int) -> None:
        """Reject incomplete comparisons before creating output artifacts."""
        if self.trials < 3:
            raise ValueError("Benchmark requires at least three trials")
        if not 1 <= self.startup_steps < steps:
            raise ValueError("Startup steps must leave at least one sustained update")
        if (
            len(set(self.workers)) != len(self.workers)
            or not {0, 2}.issubset(self.workers)
            or any(w < 0 for w in self.workers)
        ):
            raise ValueError(
                "Require distinct nonnegative CPU worker counts including 0 and 2"
            )
