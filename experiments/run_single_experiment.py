"""Script for running experiments with hydra.

Examples:
```
    python experiments/run_single_experiment.py +experiment=tiny_csp
    python experiments/run_single_experiment.py +experiment=pybullet_csp
```
"""

import logging
import time

import gymnasium as gym
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.methods.approach import ApproachFailure, BaseApproach


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, approach={cfg.approach}")
    logging.info("Full config:")
    logging.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, cfg.config_file)
    logging.info(f"Saved config to to {cfg.results_file}")

    # Initialize.
    env = hydra.utils.instantiate(cfg.env, seed=cfg.seed)
    assert isinstance(env, gym.Env)
    if cfg.record_videos:
        env = gym.wrappers.RecordVideo(env, cfg.video_dir)
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
        logging.info(f"Starting episode {episode}")
        obs, info = env.reset()
        assert "user_allows_explore" in info, (
            "Environments are required to report at reset whether the robot should "
            "explore or not. The user decides."
        )
        episode_returns = 0.0
        episode_steps = 0
        episode_start_time = time.perf_counter()
        try:
            approach.reset(obs, info)
            for _ in range(cfg.max_episode_length):
                act = approach.step()
                obs, rew, terminated, truncated, info = env.step(act)
                reward = float(rew)  # gym env rewards are SupportsFloat
                approach.update(obs, reward, terminated, info)
                episode_returns += reward
                episode_steps += 1
                if terminated or truncated:
                    break
        except ApproachFailure as e:
            logging.info(e)
        episode_duration = time.perf_counter() - episode_start_time
        episode_metrics = {
            "episode": episode,
            "user_allows_explore": info["user_allows_explore"],
            "returns": episode_returns,
            "steps": episode_steps,
            "duration": episode_duration,
            **approach.get_episode_metrics(),
        }
        logging.info(f"Finished episode with returns {episode_returns}")
        metrics.append(episode_metrics)
    env.close()

    # Aggregate and save results.
    df = pd.DataFrame(metrics)
    df.to_csv(cfg.results_file)
    logging.info(f"Wrote out results to {cfg.results_file}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
