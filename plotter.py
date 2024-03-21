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

    def add_curve(self, x, y, label=None):
        """y: vector of average reward results
        label: string to appear as label in plot legend"""
        if label is not None:
            self.ax.plot(x, y, label=label)
        else:
            self.ax.plot(x, y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls="--", c="k", label=label)

    def save(self, name="test.png"):
        """name: string for filename of saved figure"""
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


# episodes, average_returns = np.load(
#     "results/Namespace(ER=False, TN=False, anneal=False, num_episodes=200, eval_episodes=20, eval_interval=10, num_repetitions=20, lr=0.001, explr='egreedy 0.3', gamma=1).npy"
# )
# print(episodes)
# print(average_returns)

# plot = LearningCurvePlot(title="Average DQN Performance Over Repetitions")
# plot.add_curve(episodes, average_returns, label="Average Return")
# plot.save(name="dqn.png")

# Plot different results
path = "results/"
files = glob.glob(path + "*.npy")
for file in files:
    episodes, average_returns = np.load(file)
    print(episodes)
    print(average_returns)
    plot = LearningCurvePlot(title="Average DQN Performance Over Repetitions")
    plot.add_curve(episodes, average_returns, label="Average Return")
    plot.save(name="dqn.png")
