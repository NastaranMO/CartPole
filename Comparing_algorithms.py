from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import argparse

ENV = gym.make("CartPole-v1")

NUM_REPETITIONS = 20
NUM_EPISODES = 200
EVAL_EPISODES = 20
NUM_INTERVALS = 10
LEARNING_RATE = 1e-4
NUM_STEPS = 50
GAMMA = 1
BATCH_SIZE = 32


##############################################################################################################
# Compare different algorithms
##############################################################################################################
def experiment():
    Plot = LearningCurvePlot(title="Compare DQN with DQN−ER, DQN−TN, DQN−TN−ER")
    Plot.set_ylim(0, 500)
    algorithms = [
        {"ER": True, "TN": False, "label": "DQN-ER"},
        {"ER": False, "TN": False, "label": "DQN"},
        {"ER": False, "TN": True, "label": "DQN-TN"},
        {"ER": True, "TN": True, "label": "DQN-TN-ER"},
    ]
    for algorithm in algorithms:
        args = argparse.Namespace(
            ER=algorithm["ER"],
            TN=algorithm["TN"],
            anneal=False,
            num_episodes=NUM_EPISODES,
            eval_episodes=EVAL_EPISODES,
            eval_interval=NUM_INTERVALS,
            num_repetitions=NUM_REPETITIONS,
            lr=LEARNING_RATE,
            explr="egreedy 0.3",
            steps=NUM_STEPS,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
        )
        run_experiment(ENV, Plot, args, label=algorithm["label"])
    Plot.save(name="dqn-TN-ER.png")


def run_experiment(env, Plot, args, label: str):
    episodes, average_returns = average_over_repetitions(env, args)
    np.save(f"./results/episodes_{label}", episodes)
    np.save(f"./results/returns_{label}", average_returns)
    Plot.add_curve(episodes, average_returns, label=label)


if __name__ == "__main__":
    experiment()
