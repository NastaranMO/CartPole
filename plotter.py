import numpy as np
import matplotlib.pyplot as plt
import glob


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Episode Return")
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, x, y, label=None, std_dev=None):
        """y: vector of average reward results
        label: string to appear as label in plot legend"""
        if label is not None:
            self.ax.plot(x, y, label=label)
        else:
            self.ax.plot(x, y)

        if std_dev is not None:
            self.ax.fill_between(x, y - std_dev, y + std_dev, alpha=0.2)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls="--", c="k", label=label)

    def save(self, name="test.png"):
        """name: string for filename of saved figure"""
        self.ax.legend(loc="upper left")
        self.fig.savefig(name, dpi=300)


# plot = LearningCurvePlot(title="Average Double DQN Performance Over Repetitions")
plot = LearningCurvePlot(title="Comparing Double DQN and DQN-ER-TN Performance")
episodes = np.load("./results/27-03/episodes_DQN-TN-ER.npy")
average_returns = np.load("./results/27-03/returns_DQN-TN-ER.npy")
sd = np.load("./results/27-03/std_DQN-TN-ER.npy")
plot.add_curve(episodes, average_returns, label="DQN-ER-TN", std_dev=sd)
episodes, average_returns, sd = np.load(
    "results/Namespace(ER=True, TN=True, anneal=False, num_episodes=200, steps=50, eval_episodes=20, eval_interval=10, num_repetitions=20, lr=0.0001, explr='egreedy 0.3', gamma=1).npy"
)
# print(average_returns, sd)

plot.add_curve(episodes, average_returns, label="Double-DQN", std_dev=sd)
plot.save(name="compare-DQN-doubleDQN.png")

# Plot different results
# path = "results/"
# files = glob.glob(path + "*.npy")
# for file in files:
#     episodes, average_returns = np.load(file)
#     print(episodes)
#     print(average_returns)
#     plot = LearningCurvePlot(title="Average DQN Performance Over Repetitions")
#     plot.add_curve(episodes, average_returns, label="Average Return")
#     plot.save(name="dqn.png")
