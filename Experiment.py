from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import os
import argparse
import json

ENV = gym.make("CartPole-v1")
NUM_REPETITIONS = 20
NUM_EPISODES = 200
EVAL_EPISODES = 20
NUM_INTERVALS = 10
LEARNING_RATE = 1e-4
NUM_STEPS = 50
GAMMA = 1

# Default arguments for DQN-TN-ER
args = argparse.Namespace(
    ER=True,
    TN=True,
    anneal=True,
    num_episodes=200,
    eval_episodes=EVAL_EPISODES,
    eval_interval=10,
    num_repetitions=NUM_REPETITIONS,
    lr=LEARNING_RATE,
    # explr="softmax 0.1",
    explr="egreedy 0.3",
    gamma=1,
    steps=50,
)


def run_experiment(Plot, label: str, values, method):
    for value in values:
        if method == "exploration":
            args.explr = label + " " + str(value)
        elif method == "learning_rate":
            args.lr = value
        elif method == "num_steps":
            args.steps = value

        episodes, average_returns, std = average_over_repetitions(ENV, args)
        Plot.add_curve(
            episodes,
            average_returns,
            label=label + " " + str(value),
            std_dev = std
        )
        path = "./results/std/"
        if not os.path.exists(path):
            os.mkdir(path)
        path += f"/{str(args)}"
        np.save(path, [episodes, average_returns])


##############################################################################################################
# Effect of exploration strategy
##############################################################################################################
def run_experiment_exploration_strategy():
    print("Running Experiment: Effect of exploration strategy")
    Plot = LearningCurvePlot(
        title="Exploration: $\epsilon$-greedy vs softmax vs greedy policy"
    )
    Plot.set_ylim(0, 500)

    epsilon = [0.03, 0.1, 0.3]
    temp = [0.01, 0.1, 1]

    run_experiment(Plot, "egreedy", epsilon, "exploration")
    run_experiment(Plot, "softmax", temp, "exploration")
    run_experiment(Plot, "greedy", [1], "exploration")

    Plot.save(name="exploration.png")


##############################################################################################################
# Effect of learning rate
##############################################################################################################
def run_experiment_learning_rate():
    print("Running Experiment: Effect of learning rate")
    learning_rate = [1e-2, 1e-3, 1e-4, 1e-5]
    Plot = LearningCurvePlot(title="Impact of Learning Rate Variations")
    run_experiment(Plot, "lr", learning_rate, "learning_rate")
    Plot.save(name="learning-rate.png")


##############################################################################################################
# Effect of numebr of steps
##############################################################################################################
def run_experiment_num_steps():
    print("Running Experiment: Effect of number of steps")
    Plot = LearningCurvePlot(title="Impact of number of steps")
    steps = [10, 30, 50, 100]
    run_experiment(Plot, "steps", steps, "num_steps")
    Plot.save(name="num-steps.png")


##############################################################################################################
# Effect of network architecture
##############################################################################################################
def run_experiment_network_architecture():
    pass


def experiment():
    # run_experiment_exploration_strategy()
    # run_experiment_learning_rate()
    run_experiment_num_steps()
    # run_experiment_network_architecture()


if __name__ == "__main__":
    experiment()
