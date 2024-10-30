"""Script for running experiments with hydra.

Example:
    python experiments/run_single_experiment.py -m seed=1,2,3 \
        approach=csp,random env=tiny
"""

import logging
import time
from pathlib import Path

import gymnasium as gym
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.methods.approach import BaseApproach


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, approach={cfg.approach}")
    logging.info("Full config:")
    logging.info(OmegaConf.to_yaml(cfg))

    # Initialize.
    env = hydra.utils.instantiate(cfg.env, seed=cfg.seed)
    assert isinstance(env, gym.Env)
    if cfg.record_videos:
        logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        env = gym.wrappers.RecordVideo(env, Path(logdir) / cfg.video_dir)
    env.action_space.seed(cfg.seed)
    approach = hydra.utils.instantiate(
        cfg.approach,
        action_space=env.action_space,
        seed=cfg.seed,
    )
    assert isinstance(approach, BaseApproach)
    approach.train()

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
    env.close()

    # Aggregate and print results.
    df = pd.DataFrame(metrics)
    with pd.option_context(
        "display.max_colwidth",
        None,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
    ):

        print(df)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
