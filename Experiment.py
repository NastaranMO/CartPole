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
    # Effect of learning rate
    ##############################################################################################################
    # print("Running Experiment: Effect of learning rate")
    # learning_rate = [1e-2, 1e-3, 1e-4]
    # Plot = LearningCurvePlot(title="Impact of Learning Rate Variations")
    # Plot.set_ylim(0, 500)
    # for lr in learning_rate:
    #     args = argparse.Namespace(
    #         ER=False,
    #         TN=False,
    #         anneal=False,
    #         num_episodes=200,
    #         eval_episodes=EVAL_EPISODES,
    #         eval_interval=10,
    #         num_repetitions=NUM_REPETITIONS,
    #         lr=lr,
    #         explr="egreedy 0.3",
    #         gamma=1,
    #     )
    #     episodes, average_returns = average_over_repetitions(ENV, args)
    #     Plot.add_curve(
    #         episodes,
    #         average_returns,
    #         label=r"lr = {}".format(lr),
    #     )
    # Plot.save(name="learning-rate.png")
    ##############################################################################################################
    # Effect of exploration strategy
    ##############################################################################################################
    # print("Running Experiment: Effect of exploration strategy")
    # # Egreedy policy
    # epsilons = [0.03, 0.1, 0.3]
    # learning_rate = 1e-3
    # Plot = LearningCurvePlot(
    #     title="Exploration: $\epsilon$-greedy vs softmax vs greedy policy"
    # )
    # Plot.set_ylim(0, 500)

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

    # # Softmax policy
    # temps = [0.01, 0.1, 1]
    # for temp in temps:
    #     args = argparse.Namespace(
    #         ER=False,
    #         TN=False,
    #         anneal=False,
    #         num_episodes=200,
    #         eval_episodes=EVAL_EPISODES,
    #         eval_interval=10,
    #         num_repetitions=NUM_REPETITIONS,
    #         lr=learning_rate,
    #         explr="softmax " + str(temp),
    #         gamma=1,
    #     )
    #     episodes, average_returns = average_over_repetitions(ENV, args)
    #     Plot.add_curve(
    #         episodes,
    #         average_returns,
    #         label=r"Softmax, $\tau $ = {}".format(temp),
    #     )

    # # Greedy policy
    # args = argparse.Namespace(
    #     ER=False,
    #     TN=False,
    #     anneal=False,
    #     num_episodes=200,
    #     eval_episodes=EVAL_EPISODES,
    #     eval_interval=10,
    #     num_repetitions=NUM_REPETITIONS,
    #     lr=learning_rate,
    #     explr="greedy 1",
    #     gamma=1,
    # )
    # episodes, average_returns = average_over_repetitions(ENV, args)
    # Plot.add_curve(episodes, average_returns, label="Greedy")

    # Plot.save(name="exploration.png")

    ##############################################################################################################
    # DQN with DQN-ER
    ##############################################################################################################
    Plot = LearningCurvePlot(title="Compare DQN with DQN−ER")
    Plot.set_ylim(0, 500)
    learning_rate = 1e-3
    args = argparse.Namespace(
        ER=True,
        TN=False,
        anneal=False,
        num_episodes=200,
        eval_episodes=EVAL_EPISODES,
        eval_interval=10,
        num_repetitions=NUM_REPETITIONS,
        lr=learning_rate,
        explr="egreedy 0.3",
        gamma=1,
    )
    episodes, average_returns = average_over_repetitions(ENV, args)
    Plot.add_curve(episodes, average_returns, label="DQN-ER")

    args = argparse.Namespace(
        ER=False,
        TN=False,
        anneal=False,
        num_episodes=200,
        eval_episodes=EVAL_EPISODES,
        eval_interval=10,
        num_repetitions=NUM_REPETITIONS,
        lr=learning_rate,
        explr="egreedy 0.3",
        gamma=1,
    )
    episodes, average_returns = average_over_repetitions(ENV, args)
    Plot.add_curve(episodes, average_returns, label="DQN")

    args = argparse.Namespace(
        ER=False,
        TN=True,
        anneal=False,
        num_episodes=200,
        eval_episodes=EVAL_EPISODES,
        eval_interval=10,
        num_repetitions=NUM_REPETITIONS,
        lr=learning_rate,
        explr="egreedy 0.3",
        gamma=1,
    )
    episodes, average_returns = average_over_repetitions(ENV, args)
    Plot.add_curve(episodes, average_returns, label="DQN-TN")

    args = argparse.Namespace(
        ER=True,
        TN=True,
        anneal=False,
        num_episodes=200,
        eval_episodes=EVAL_EPISODES,
        eval_interval=10,
        num_repetitions=NUM_REPETITIONS,
        lr=learning_rate,
        explr="egreedy 0.3",
        gamma=1,
    )
    episodes, average_returns = average_over_repetitions(ENV, args)
    Plot.add_curve(episodes, average_returns, label="DQN-TN-ER")

    Plot.save(name="dqn-TN-ER.png")


if __name__ == "__main__":
    experiment()
