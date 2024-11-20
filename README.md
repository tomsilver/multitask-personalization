# multitask-personalization

![workflow](https://github.com/tomsilver/multitask-personalization/actions/workflows/ci.yml/badge.svg)

## Requirements

- Python 3.10+
- Tested on MacOS Catalina

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`
3. Create an OpenAI API key for using LLMs and `export OPENAI_API_KEY=<key>`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.

## Running Experiments

The main command for running experiments is below. This will be updated as we go on.

```
python experiments/run_single_experiment.py -m \
    env=tiny,pybullet \
    approach=ours,nothing_personal,exploit_only \
    seed="range(1, 11)" \
    wandb.enable=True wandb.group=main wandb.run_name="\${env}-\${approach}-\${seed}" wandb.entity=$WANDB_USER
```

Here's an example command to run a much shorter, cheaper set of all experiments:
```
python experiments/run_single_experiment.py -m \
    env=tiny,pybullet \
    approach=ours,nothing_personal,exploit_only \
    seed="range(1, 3)" \
    env.max_environment_steps=100 \
    env.eval_frequency=50 \
    env.num_eval_trials=1 \
    csp_solver.min_num_satisfying_solutions=1 \
    csp_solver.max_iters=100
```
