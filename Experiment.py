from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import os
import argparse
import json

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
    # learning_rate = 1e-3
    # args = argparse.Namespace(
    #     ER=True,
    #     TN=False,
    #     anneal=False,
    #     num_episodes=200,
    #     eval_episodes=EVAL_EPISODES,
    #     eval_interval=10,
    #     num_repetitions=NUM_REPETITIONS,
    #     lr=learning_rate,
    #     explr="egreedy 0.3",
    #     steps=50,
    #     gamma=1,
    # )
    # episodes, average_returns = average_over_repetitions(ENV, args)
    # Plot.add_curve(episodes, average_returns, label="DQN-ER")

    # args = argparse.Namespace(
    #     ER=False,
    #     TN=False,
    #     anneal=False,
    #     num_episodes=200,
    #     eval_episodes=EVAL_EPISODES,
    #     eval_interval=10,
    #     num_repetitions=NUM_REPETITIONS,
    #     lr=learning_rate,
    #     explr="egreedy 0.3",
    #     steps=50,
    #     gamma=1,
    # )
    # episodes, average_returns = average_over_repetitions(ENV, args)
    # Plot.add_curve(episodes, average_returns, label="DQN")

    # args = argparse.Namespace(
    #     ER=False,
    #     TN=True,
    #     anneal=False,
    #     num_episodes=200,
    #     eval_episodes=EVAL_EPISODES,
    #     eval_interval=10,
    #     num_repetitions=NUM_REPETITIONS,
    #     lr=learning_rate,
    #     explr="egreedy 0.3",
    #     steps=50,
    #     gamma=1,
    # )
    # episodes, average_returns = average_over_repetitions(ENV, args)
    # Plot.add_curve(episodes, average_returns, label="DQN-TN")

    # args = argparse.Namespace(
    #     ER=True,
    #     TN=True,
    #     anneal=False,
    #     num_episodes=200,
    #     eval_episodes=EVAL_EPISODES,
    #     eval_interval=10,
    #     num_repetitions=NUM_REPETITIONS,
    #     lr=learning_rate,
    #     explr="egreedy 0.3",
    #     steps=50,
    #     gamma=1,
    # )
    # episodes, average_returns = average_over_repetitions(ENV, args)
    # print(average_returns)
    # Plot.add_curve(episodes, average_returns, label="DQN-TN-ER")

    # Plot.save(name="dqn-TN-ER.png")

    # args = argparse.Namespace(
    #     ER=True,
    #     TN=True,
    #     anneal=True,
    #     num_episodes=200,
    #     eval_episodes=5,
    #     eval_interval=10,
    #     num_repetitions=5,
    #     lr=1.3886531790755207e-5,
    #     explr="softmax 1.2576454147068064",
    #     gamma=1,
    #     steps=49,
    #     batch_size=128,
    # )

    # episodes, average_returns = average_over_repetitions(ENV, args)
    # Plot.add_curve(episodes, average_returns, label="DQN-TN-ER")

    # Plot.save(name="dqn-TN-ER-mybest.png")
    episodes_return = np.load("episodes_Nastaran.npy")
    average_returns = np.load("returns_Nastaran.npy")
    outputfile = open("outputfile_Nastaran.json", "r")
    dict_list = json.load(outputfile)
    i = 0
    for dict in dict_list:
        args = argparse.Namespace(
            ER=dict["ER"],
            TN=dict["TN"],
            anneal=dict["anneal"],
            num_episodes=dict["num_episodes"],
            eval_episodes=dict["eval_episodes"],
            eval_interval=dict["eval_interval"],
            num_repetitions=dict["num_repetitions"],
            lr=dict["lr"],
            explr=dict["explr"],
            gamma=1,
            steps=dict["steps"],
            batch_size=dict["batchsize"],
        )

        Plot.add_curve(
            episodes_return[i],
            average_returns[i],
            label=f"experiment {i+1}",
        )
        i += 1
    Plot.save(name="dqn-TN-ER-20.png")


if __name__ == "__main__":
    experiment()
