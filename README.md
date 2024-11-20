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

### Local

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

### G2 (SLURM cluster)

If this is your first time ever using G2, do the following after sshing in:

1. Create a new SSH key and add it to your GitHub account ([ref](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?tool=webui)).
2. Optionally add an alias `sl` for monitoring jobs: `echo "alias sl='squeue --format=\"%.18i %.9P %.42j %.8u %.8T %.10M %.6D %R\"'" >> ~/.bashrc`
3. Set up conda:
```
echo "export PATH=/share/apps/anaconda3/2021.05/bin:$PATH" >> ~/.bashrc

conda init bash

[exit and log back in]
```

If this is your first time using _this repo_ on G2, do the following:

1. Clone the repo: `git clone git@github.com:tomsilver/multitask-personalization.git`
2. cd into it: `cd multitask-personalization`
3. Create a conda env: `conda create --name multitask-personalization python=3.10`
4. Active the conda env: `conda activate multitask-personalization`
5. Install the repo: `pip install -e ".[develop]"`
6. Test that it worked: `pytest -s tests/methods/test_csp_approach.py`
7. Install the hydra SLURM launcher: `pip install hydra-submitit-launcher`
8. Install magic-wormhole for downloading results: `pip install magic-wormhole` (on both G2 and your local machine)

If everything is installed and you're ready to run jobs:

Use the same commands above, but with `hydra/launcher=g2` added to the command line. For example:

```
python experiments/run_single_experiment.py -m env=tiny approach=ours seed="range(1, 5)" hydra/launcher=g2
```

Note that the launch command will hang due to an open issue (https://github.com/facebookresearch/hydra/issues/2479). Control-C after a while and verify that the jobs were launched.
