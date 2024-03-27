from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import os
import argparse
import json

ENV = gym.make("CartPole-v1")

NUM_REPETITIONS = 20
EVAL_EPISODES = 20

config = {
    "ER": False,
    "TN": False,
    "anneal": True,
    "num_episodes": 200,
    "eval_episodes": EVAL_EPISODES,
    "eval_interval": 10,
    "num_repetitions": NUM_REPETITIONS,
    "lr": 1e-3,
    "explr": "egreedy 0.3",
    "gamma": 1,
    "steps": 50,
    "batch_size": 32,
}


def run_ablation(ER: str, TR: str):
    config["ER"] = ER
    config["TN"] = TR
    args = argparse.Namespace(**config)

    episodes, average_returns = average_over_repetitions(ENV, args)

    a = np.array(average_returns)
    e = np.array(episodes)
    np.save(f'episodes_{ER}_{TR}.npy', e)
    np.save(f'returns_{ER}_{TR}.npy', a)


if __name__ == "__main__":
    run_ablation("True", "True")
    # run_ablation("True", "False")
    # run_ablation("False", "True")
    # run_ablation("False", "False")

    print("Done")
