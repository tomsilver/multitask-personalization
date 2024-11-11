"""Script for running experiments with hydra.

Examples:
```
    python experiments/run_single_experiment.py +experiment=tiny_csp
    python experiments/run_single_experiment.py +experiment=pybullet_csp
```
"""

import logging

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.methods.approach import ApproachFailure, BaseApproach


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env}, approach={cfg.approach}")
    logging.info("Full config:")
    logging.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, cfg.config_file)
    logging.info(f"Saved config to to {cfg.config_file}")

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
    obs, info = env.reset()
    assert "user_allows_explore" in info, (
        "Environments are required to report at reset whether the robot should "
        "explore or not. The user decides."
    )
    approach.reset(obs, info)
    for t in range(cfg.max_environment_steps):
        try:
            act = approach.step()
        except ApproachFailure as e:
            logging.info(e)
        obs, rew, terminated, truncated, info = env.step(act)
        assert np.isclose(rew, 0.0)
        assert not (terminated or truncated)
        approach.update(obs, float(rew), terminated, info)
        user_satisfaction = info.get("user_satisfaction", np.nan)
        step_metrics = {
            "step": t,
            "user_allows_explore": info["user_allows_explore"],
            "user_satisfaction": user_satisfaction,
            **approach.get_step_metrics(),
        }
        logging.info(f"Step {t} satisfaction: {user_satisfaction}")
        metrics.append(step_metrics)
    env.close()

    # Aggregate and save results.
    df = pd.DataFrame(metrics)
    df.to_csv(cfg.results_file)
    logging.info(f"Wrote out results to {cfg.results_file}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
