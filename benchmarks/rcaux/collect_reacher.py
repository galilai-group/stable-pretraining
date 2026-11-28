"""Collect state-only DeepMind Control Reacher trajectories for the benchmark."""

import argparse
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from dm_control import suite


def collect(output: Path, episodes: int, steps: int, seed: int) -> None:
    """Save random-policy trajectories without rendering or image downloads.

    Args:
        output: Destination HDF5 file; existing files are not overwritten.
        episodes: Number of independently reset episodes.
        steps: Number of transitions per episode.
        seed: Seed for environment initialization and actions.
    """
    rng = np.random.default_rng(seed)
    env = suite.load("reacher", "easy", task_kwargs={"random": seed})
    observations, actions, offsets, lengths = [], [], [], []
    spec = env.action_spec()
    for _ in range(episodes):
        time_step = env.reset()
        offsets.append(len(observations))
        count = 0
        for _ in range(steps):
            observations.append(
                np.concatenate(
                    [np.asarray(v).ravel() for v in time_step.observation.values()]
                )
            )
            action = rng.uniform(spec.minimum, spec.maximum)
            actions.append(action)
            time_step = env.step(action)
            count += 1
            if time_step.last():
                break
        observations.append(
            np.concatenate(
                [np.asarray(v).ravel() for v in time_step.observation.values()]
            )
        )
        # The final observation has no outgoing transition and is never used as an action source.
        actions.append(np.zeros(spec.shape))
        lengths.append(count + 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as data:
        for name, value in {
            "observation": observations,
            "action": actions,
            "ep_offset": offsets,
            "ep_len": lengths,
        }.items():
            data.create_dataset(name, data=np.asarray(value), compression="gzip")
        data.attrs.update(
            domain="reacher",
            task="easy",
            policy="uniform_random",
            seed=seed,
            requested_steps=steps,
            observation_keys="position,to_target,velocity",
            dm_control_version=version("dm-control"),
            mujoco_version=version("mujoco"),
            numpy_version=np.__version__,
        )
    print(f"Saved {episodes} episodes to {output}")


def main() -> None:
    """Collect the deterministic dataset used in the documented comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    collect(args.output, args.episodes, args.steps, args.seed)


if __name__ == "__main__":
    main()
