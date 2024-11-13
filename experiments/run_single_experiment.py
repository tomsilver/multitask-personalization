"""Script for running experiments with hydra.

Examples:
```
    python experiments/run_single_experiment.py +experiment=tiny_csp
    python experiments/run_single_experiment.py +experiment=pybullet_csp
```
"""

import logging
from pathlib import Path

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
    logging.info(f"Saved config to {cfg.config_file}")
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)
    logging.info(f"Created model directory at {cfg.model_dir}")

    # Create training environment, which should only be reset once.
    train_env = hydra.utils.instantiate(
        cfg.env, scene_spec=cfg.scene_spec, seed=cfg.seed
    )
    assert isinstance(train_env, gym.Env)
    if cfg.record_train_videos:
        train_env = gym.wrappers.RecordVideo(
            train_env, str(Path(cfg.video_dir) / "train")
        )
    train_env.action_space.seed(cfg.seed)

    # Create eval environment, which will be reset all the time.
    eval_seed = cfg.seed + cfg.eval_seed_offset
    eval_env = hydra.utils.instantiate(
        cfg.env, scene_spec=cfg.scene_spec, seed=eval_seed
    )
    assert isinstance(eval_env, gym.Env)
    if cfg.record_eval_videos:
        eval_env = gym.wrappers.RecordVideo(eval_env, str(Path(cfg.video_dir) / "eval"))
    eval_env.action_space.seed(eval_seed)

    # Create two copies of the approach. The eval approach will load model files
    # from the train approach to do evaluation without losing track of state in
    # the training approach.
    train_approach = hydra.utils.instantiate(
        cfg.approach,
        scene_spec=cfg.scene_spec,
        action_space=train_env.action_space,
        seed=cfg.seed,
    )
    assert isinstance(train_approach, BaseApproach)
    train_approach.train()
    eval_approach = hydra.utils.instantiate(
        cfg.approach,
        scene_spec=cfg.scene_spec,
        action_space=eval_env.action_space,
        seed=eval_seed,
    )
    assert isinstance(eval_approach, BaseApproach)
    eval_approach.eval()

    # Log training and eval metrics separately.
    train_metrics: list[dict[str, float]] = []
    eval_metrics: list[dict[str, float]] = []

    # Reset the training environment, one time only.
    obs, info = train_env.reset()
    # Reset the training approach, one time only.
    train_approach.reset(obs, info)
    # Main training and eval loop.
    for t in range(cfg.max_environment_steps + 1):
        # Check if it's time to eval.
        if t % cfg.eval_frequency == 0:
            # Save the models from the training approach and load them into the
            # eval approach.
            step_model_dir = model_dir / str(t)
            step_model_dir.mkdir(exist_ok=True)
            train_approach.save(step_model_dir)
            eval_approach.load(step_model_dir)
            # Run evaluation.
            step_eval_metrics = _evaluate_approach(eval_approach, eval_env, cfg, t)
            eval_metrics.append(step_eval_metrics)
        # Eval on the last time step but don't train anymore.
        if t >= cfg.max_environment_steps:
            break
        # Continue training.
        try:
            act = train_approach.step()
        except ApproachFailure as e:
            logging.info(e)
        obs, rew, terminated, truncated, info = train_env.step(act)
        assert np.isclose(rew, 0.0)
        assert not (terminated or truncated)
        train_approach.update(obs, float(rew), terminated, info)
        user_satisfaction = info.get("user_satisfaction", np.nan)
        step_train_metrics = {
            "step": t,
            "user_satisfaction": user_satisfaction,
            **train_approach.get_step_metrics(),
        }
        logging.info(f"Step {t} satisfaction: {user_satisfaction}")
        train_metrics.append(step_train_metrics)
    train_env.close()
    eval_env.close()

    # Aggregate and save results.
    train_df = pd.DataFrame(train_metrics)
    train_df.to_csv(cfg.train_results_file)
    logging.info(f"Wrote out training results to {cfg.train_results_file}")

    eval_df = pd.DataFrame(eval_metrics)
    eval_df.to_csv(cfg.eval_results_file)
    logging.info(f"Wrote out eval results to {cfg.eval_results_file}")


def _evaluate_approach(
    eval_approach: BaseApproach, eval_env: gym.Env, cfg: DictConfig, training_step: int
) -> dict[str, float]:
    """Evaluate the given approach and return metrics."""
    # Evaluate for a given number of trials.
    cumulative_user_satisfactions: list[float] = []
    for eval_trial_idx in range(cfg.num_eval_trials):
        logging.info(f"Starting eval trial {eval_trial_idx}")
        seed = cfg.seed + cfg.eval_seed_offset + eval_trial_idx
        obs, info = eval_env.reset(seed=seed)
        # Reset the approach.
        eval_approach.reset(obs, info)
        # Main eval loop.
        cumulative_user_satisfaction = 0.0
        for _ in range(cfg.max_eval_episode_length):
            try:
                act = eval_approach.step()
            except ApproachFailure as e:
                logging.info(e)
            obs, rew, terminated, truncated, info = eval_env.step(act)
            assert np.isclose(float(rew), 0.0)
            assert not (terminated or truncated)
            eval_approach.update(obs, float(rew), terminated, info)
            user_satisfaction = info.get("user_satisfaction", 0.0)
            cumulative_user_satisfaction += user_satisfaction
            if cfg.terminate_eval_episode_on_nonzero and user_satisfaction != 0:
                break
        cumulative_user_satisfactions.append(cumulative_user_satisfaction)
    step_eval_metrics: dict[str, float] = {"training_step": training_step}
    for idx, cus in enumerate(cumulative_user_satisfactions):
        step_eval_metrics[f"eval_episode_{idx}_user_satisfaction"] = cus
    return step_eval_metrics


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
