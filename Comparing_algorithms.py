from plotter import LearningCurvePlot
from AgentTrainer import average_over_repetitions
import gymnasium as gym
import numpy as np
import argparse
from matplotlib import pyplot as plt

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
    episodes, average_returns, std = average_over_repetitions(env, args)
    np.save(f"./results/27-03/episodes_{label}", episodes)
    np.save(f"./results/27-03/returns_{label}", average_returns)
    np.save(f"./results/27-03/std_{label}", std)
    Plot.add_curve(episodes, average_returns, label=label, std_dev=std)


def plot_final_performance(final_returns, final_std_devs, config_labels):
    """
    Plots the final performance of different configurations of a DQN agent for the ablation study..
    """
    plt.close('all') # Close all existing plots

    assert len(final_returns) == len(final_std_devs) == len(config_labels), "All input lists must have the same length."

    fig, ax = plt.subplots()  # Create a new figure and axis for the bar plot

    y_pos = np.arange(len(config_labels))

    # Create the bar plot
    bars = ax.bar(y_pos, final_returns, align='center', alpha=0.7, capsize=5)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(config_labels)
    ax.set_ylabel('Average Return at Final Episode')  # Set the y-axis label for the bar plot
    ax.set_xlabel('Configuration')  # Set the x-axis label for the bar plot
    ax.set_title('Ablation Study Results - Final Step Performance')

    # Add error bars to indicate standard deviation at the final step
    ax.errorbar(y_pos, final_returns, yerr=final_std_devs, fmt='none', ecolor='black', capthick=2, capsize=5)

    # Optionally, annotate standard deviation values above the bars
    for bar, std in zip(bars, final_std_devs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'±{std:.2f}',
                ha='center', va='bottom', fontsize=8, rotation=90)

    plt.show()


def load_arrays():
    final_return_dqn_er_tn = np.load('./results/27-03/returns_DQN-TN-ER.npy')[-1]
    final_std_dev_dqn_er_tn = np.load('./results/27-03/std_DQN-TN-ER.npy')[-1]
    final_return_dqn_er = np.load('./results/27-03/returns_DQN-ER.npy')[-1]
    final_std_dev_dqn_er = np.load('./results/27-03/std_DQN-ER.npy')[-1]
    final_return_dqn_tn = np.load('./results/27-03/returns_DQN-TN.npy')[-1]
    final_std_dev_dqn_tn = np.load('./results/27-03/std_DQN-TN.npy')[-1]
    final_return_dqn = np.load('./results/27-03/returns_DQN.npy')[-1]
    final_std_dev_dqn = np.load('./results/27-03/std_DQN.npy')[-1]

    return ([final_return_dqn_er_tn, final_return_dqn_er, final_return_dqn_tn, final_return_dqn],
            [final_std_dev_dqn_er_tn, final_std_dev_dqn_er, final_std_dev_dqn_tn, final_std_dev_dqn])


if __name__ == "__main__":
    experiment()

    # Load the arrays containing the final returns and standard deviations for each configuration
    returns, std_devs = load_arrays()
    config_labels = ['DQN-ER-TN', 'DQN-ER', 'DQN-TN', 'DQN']

    # Plot the final performance of the different configurations (Ablation Study)
    plot_final_performance(returns, std_devs, config_labels)
