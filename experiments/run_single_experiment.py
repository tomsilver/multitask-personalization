"""Script for running experiments with hydra."""

import logging
import time

import gymnasium as gym
import hydra
import pandas as pd
from omegaconf import DictConfig

from multitask_personalization.methods.approach import BaseApproach


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, approach={cfg.approach}")

    # Initialize.
    env = hydra.utils.instantiate(cfg.env, seed=cfg.seed)
    assert isinstance(env, gym.Env)
    approach = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        seed=cfg.seed,
    )
    assert isinstance(approach, BaseApproach)

    # Run a certain number of episodes and log metrics along the way.
    metrics: list[dict[str, float]] = []
    for episode in range(cfg.num_episodes):
        obs, info = env.reset()
        approach.reset(obs, info)
        episode_returns = 0.0
        episode_steps = 0
        episode_start_time = time.perf_counter()
        for _ in range(cfg.max_episode_length):
            act = approach.step()
            obs, rew, terminated, truncated, info = env.step(act)
            reward = float(rew)  # gym env rewards are SupportsFloat
            approach.update(obs, reward, terminated, info)
            episode_returns += reward
            episode_steps += 1
            if terminated or truncated:
                break
        episode_duration = time.perf_counter() - episode_start_time
        episode_metrics = {
            "episode": episode,
            "returns": episode_returns,
            "steps": episode_steps,
            "duration": episode_duration,
        }
        metrics.append(episode_metrics)

    # Aggregate and save results.
    df = pd.DataFrame(metrics)
    print(df)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
