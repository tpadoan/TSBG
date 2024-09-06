import os
import torch as th
import numpy as np
import random
import gymnasium as gym
from typing import Any
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances


from sb3_SY import ScotlandYard


N_TRIALS = 50  # Maximum number of trials
N_JOBS = 1  # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 40  # Number of evaluations during the training
N_TIMESTEPS = int(2e6)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 16
N_EVAL_EPISODES = 10
ENV_ID = ScotlandYard
TIMEOUT = int(60 * 120)  # 2h timeout


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """
    Sampler for PPO hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    # Discount factor between 0.9 and 0.9999
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    # 64, 128, , ... 8192
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 12, 20)
    # Learning rate
    learning_rate = trial.suggest_float("lr", 1e-4, 4e-4, log=True)
    # PPO learning epochs
    n_epochs = trial.suggest_categorical(
        "n_epochs", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
    # PPO minibatch size
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # PPO entropy coefficient
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 1e-2, log=True)

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("n_steps", n_steps)

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
    }


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 2,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    ### Environment input parameters
    random_start = True
    num_detectives = 3
    max_turns = 10
    reveal_every = 0

    env_kwargs = {
        "random_start": random_start,
        "num_detectives": num_detectives,
        "max_turns": max_turns,
        "reveal_every": reveal_every,
    }

    # 1. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    eval_envs = make_vec_env(
        ENV_ID,
        n_envs=N_EVAL_ENVS,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
    )

    # 2. Sample hyperparameters and update the keyword arguments
    kwargs = {
        "policy": "MlpPolicy",
        "env": eval_envs,
        "device": device,
        "verbose": 1,
    }

    kwargs.update(sample_ppo_params(trial))

    # Create the RL model
    model = MaskablePPO(**kwargs)

    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    eval_callback = TrialEvalCallback(
        eval_envs,
        trial,
        N_EVAL_EPISODES,
        EVAL_FREQ,
        deterministic=False,
    )

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def main():
    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        storage="sqlite:///ppo_rl_gnn.db",
        study_name="sb3-parallel",
        load_if_exists=True,
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=None)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("study_results_ppo_RL_GNN.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()
