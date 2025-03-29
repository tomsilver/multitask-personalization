"""Script for running experiments with hydra.

See README for examples.
"""

import logging
from pathlib import Path

import gymnasium as gym
from multitask_personalization.utils import RecordVideoWithIntermediateFrames
import hydra
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from multitask_personalization.methods.approach import BaseApproach


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(
        f"Running seed={cfg.seed}, env={cfg.env_name}, approach={cfg.approach_name}"
    )
    logging.info("Full config:")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logging.info(OmegaConf.to_yaml(resolved_cfg))
    OmegaConf.save(resolved_cfg, cfg.config_file)
    logging.info(f"Saved config to {cfg.config_file}")
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)
    logging.info(f"Created model directory at {cfg.model_dir}")
    saved_state_dir = Path(cfg.saved_state_dir)
    saved_state_dir.mkdir(exist_ok=True)
    logging.info(f"Created saved state directory at {cfg.saved_state_dir}")

    # Sanity check config.
    assert cfg.env.max_environment_steps % cfg.env.eval_frequency == 0

    # Initialize weights and biases.
    if cfg.wandb.enable:
        wandb.config = resolved_cfg  # type: ignore
        assert cfg.wandb.entity is not None
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group if cfg.wandb.group else None,
            name=cfg.wandb.run_name if cfg.wandb.run_name else None,
            dir=cfg.wandb.dir,
        )

    # Create training environment, which should only be reset once.
    train_env_cfg = OmegaConf.merge(cfg.env.env, cfg.env.train_env)
    train_env = hydra.utils.instantiate(train_env_cfg, seed=cfg.seed)
    assert isinstance(train_env, gym.Env)
    if cfg.record_train_videos:
        train_env = RecordVideoWithIntermediateFrames(
            train_env, str(Path(cfg.video_dir) / "train")
        )
    train_env.action_space.seed(cfg.seed)

    # Create eval environment, which will be reset all the time.
    eval_seed = cfg.seed + cfg.eval_seed_offset
    eval_env_cfg = OmegaConf.merge(cfg.env.env, cfg.env.eval_env)
    eval_env = hydra.utils.instantiate(eval_env_cfg, seed=eval_seed)
    assert isinstance(eval_env, gym.Env)
    if cfg.record_eval_videos:
        eval_env = RecordVideoWithIntermediateFrames(eval_env, str(Path(cfg.video_dir) / "eval"))
    eval_env.action_space.seed(eval_seed)

    # Create two copies of the approach. The eval approach will load model files
    # from the train approach to do evaluation without losing track of state in
    # the training approach.
    train_approach = hydra.utils.instantiate(
        cfg.approach,
        train_env.unwrapped.scene_spec,
        train_env.action_space,
        seed=cfg.seed,
    )
    assert isinstance(train_approach, BaseApproach)
    train_approach.train()
    eval_approach = hydra.utils.instantiate(
        cfg.approach,
        eval_env.unwrapped.scene_spec,
        eval_env.action_space,
        seed=eval_seed,
    )
    assert isinstance(eval_approach, BaseApproach)
    eval_approach.eval()

    # Log training and eval metrics separately.
    train_metrics: list[dict[str, float]] = []
    eval_metrics: list[dict[str, float]] = []

    # Catch any exceptions so we can debug from the last saved state.
    try:

        # Reset the training environment, one time only.
        obs, info = train_env.reset()
        # Reset the training approach, one time only.
        train_approach.reset(obs, info)
        # Main training and eval loop.
        for t in range(cfg.env.max_environment_steps + 1):
            if t % cfg.train_logging_interval == 0:
                logging.info(f"Starting training step {t}")
            # Check if it's time to eval.
            if cfg.env.eval_frequency > 0 and t % cfg.env.eval_frequency == 0:

                # Save the models from the training approach and load them into the
                # eval approach.
                step_model_dir = model_dir / str(t)
                step_model_dir.mkdir(exist_ok=True)
                train_approach.save(step_model_dir)
                eval_approach.load(step_model_dir)
                # Run evaluation.
                step_eval_metrics = _evaluate_approach(eval_approach, eval_env, cfg, t)
                if cfg.wandb.enable:
                    wandb_metrics = {
                        f"eval/{k}": v for k, v in step_eval_metrics.items()
                    }
                    del wandb_metrics["eval/training_step"]
                    wandb.log(wandb_metrics, step=t)
                eval_metrics.append(step_eval_metrics)
                logging.info("Resuming training")
            # Eval on the last time step but don't train anymore.
            if t >= cfg.env.max_environment_steps:
                break
            # Continue training.
            act = train_approach.step()
            obs, rew, _, _, info = train_env.step(act)
            assert np.isclose(rew, 0.0)
            # During training, there is no such thing as termination.
            terminated = False
            train_approach.update(obs, float(rew), terminated, info)
            user_satisfaction = info.get("user_satisfaction", np.nan)
            env_video_should_pause = info.get("env_video_should_pause", False)
            step_train_metrics = {
                "step": t,
                "execution_time": t * cfg.env.dt,
                "user_satisfaction": user_satisfaction,
                "env_video_should_pause": env_video_should_pause,
                **train_approach.get_step_metrics(),
            }
            if cfg.wandb.enable:
                wandb_metrics = {f"train/{k}": v for k, v in step_train_metrics.items()}
                del wandb_metrics["train/step"]
                wandb.log(wandb_metrics, step=t)
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

        if cfg.wandb.enable:
            wandb.finish()

    except BaseException as e:
        logging.warning("Crashed! Saving environment states before finishing.")
        train_env.unwrapped.save_state(saved_state_dir / "crash_train_env_state.p")
        eval_env.unwrapped.save_state(saved_state_dir / "crash_eval_env_state.p")

        train_env.close()
        eval_env.close()

        # Aggregate and save results.
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv(cfg.train_results_file)
        logging.info(
            f"Wrote out INCOMPLETE training results to {cfg.train_results_file}"
        )

        eval_df = pd.DataFrame(eval_metrics)
        eval_df.to_csv(cfg.eval_results_file)
        logging.info(f"Wrote out INCOMPLETE eval results to {cfg.eval_results_file}")

        logging.critical(e, exc_info=True)


def _evaluate_approach(
    eval_approach: BaseApproach, eval_env: gym.Env, cfg: DictConfig, training_step: int
) -> dict[str, float]:
    """Evaluate the given approach and return metrics."""
    # Evaluate for a given number of trials.
    cumulative_user_satisfactions: list[float] = []
    logging.info("Starting evaluation")
    for eval_trial_idx in range(cfg.env.num_eval_trials):
        seed = cfg.seed + cfg.eval_seed_offset + eval_trial_idx
        obs, info = eval_env.reset(seed=seed)
        # Reset the approach.
        eval_approach.reset(obs, info)
        # Main eval loop.
        cumulative_user_satisfaction = 0.0
        for _ in range(cfg.env.max_eval_episode_length):
            act = eval_approach.step()
            obs, rew, terminated, truncated, info = eval_env.step(act)
            assert np.isclose(float(rew), 0.0)
            eval_approach.update(obs, float(rew), terminated, info)
            user_satisfaction = info.get("user_satisfaction", 0.0)
            cumulative_user_satisfaction += user_satisfaction
            if terminated or truncated:
                break
        cumulative_user_satisfactions.append(cumulative_user_satisfaction)
    step_eval_metrics: dict[str, float] = {
        "training_step": training_step,
        "training_execution_time": training_step * cfg.env.dt,
    }
    for idx, cus in enumerate(cumulative_user_satisfactions):
        step_eval_metrics[f"eval_episode_{idx}_user_satisfaction"] = cus
    step_eval_metrics["eval_mean_user_satisfaction"] = float(
        np.mean(cumulative_user_satisfactions)
    )
    return step_eval_metrics


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
