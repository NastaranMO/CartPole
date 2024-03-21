from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import os
import argparse

ENV = gym.make("CartPole-v1")

NUM_REPETITIONS = 5
EVAL_EPISODES = 20


def experiment():
    ##############################################################################################################
    # Effect of exploration strategy
    ##############################################################################################################
    print("Running Experiment: Effect of exploration strategy")
    policy = "egreedy"
    # epsilons = [0.03, 0.1, 0.3]
    epsilons = [0.3]
    learning_rate = 1e-3
    Plot = LearningCurvePlot(
        title="Exploration: $\epsilon$-greedy vs softmax exploration"
    )
    Plot.set_ylim(0, 500)

    # for epsilon in epsilons:
    #     args = argparse.Namespace(
    #         ER=False,
    #         TN=False,
    #         anneal=False,
    #         num_episodes=200,
    #         eval_episodes=EVAL_EPISODES,
    #         eval_interval=10,
    #         num_repetitions=NUM_REPETITIONS,
    #         lr=learning_rate,
    #         explr="egreedy " + str(epsilon),
    #         gamma=1,
    #     )
    #     episodes, average_returns = average_over_repetitions(ENV, args)
    #     Plot.add_curve(
    #         episodes,
    #         average_returns,
    #         label=r"$\epsilon$-greedy, $\epsilon $ = {}".format(epsilon),
    #     )

    policy = "softmax"
    # temps = [0.01, 0.1, 1]
    temps = [1]
    for temp in temps:
        args = argparse.Namespace(
            ER=False,
            TN=False,
            anneal=False,
            num_episodes=200,
            eval_episodes=EVAL_EPISODES,
            eval_interval=10,
            num_repetitions=NUM_REPETITIONS,
            lr=learning_rate,
            explr="softmax " + str(temp),
            gamma=1,
        )
        episodes, average_returns = average_over_repetitions(ENV, args)
        Plot.add_curve(
            episodes,
            average_returns,
            label=r"Softmax, $\tau $ = {}".format(temp),
        )

    Plot.save(name="exploration.png")


if __name__ == "__main__":
    experiment()
