import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.optim as optim
from scipy.signal import savgol_filter
from collections import deque
import tqdm
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from utils import argmax, smooth

env = gym.make("CartPole-v1")
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

num_episodes = 200
num_eval_episodes = 20
eval_interval = 10

num_repetitions = 20
num_evaluation_points = num_episodes // eval_interval
returns_over_repetitions = np.zeros((num_repetitions, num_evaluation_points))

learning_rate = 1e-3

# Check for GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

np.random.seed(42)
torch.manual_seed(42)


class NeuralNetwork(nn.Module):
    """
    A neural network developed for approximating the Q-value function for a Deep Q-Network (DQN) algorithm.

    Attributes:
        dqn_model (torch.nn.Sequential): The sequence of layers comprising the neural network model.

    Parameters:
        num_states (int): The number of states in the environment.
        num_actions (int): The number of possible actions, agent can take.

    Methods:
        forward(x): Performs a forward pass through the network.
        init_weights(): Initializes the weights of the neural network.
    """

    def __init__(self, num_states, num_actions):
        super(NeuralNetwork, self).__init__()

        self.dqn_model = nn.Sequential(
            nn.Linear(num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions),
        )

        self.init_weights()

    def forward(self, x):
        return self.dqn_model(x)

    def init_weights(self):
        for layer in self.dqn_model:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)


class DQN_Agent:
    """
    DQN agent uses a neural network to approximate the Q-value function and uses Experience Replay and Target Network to improve stability and efficiency during training.

    Attributes:
        - steps_done (int): Counter for the number of steps taken.
        - learning_rate (float): The learning rate for the optimizer.
        - policy_net (NeuralNetwork): The neural network used to approximate the policy Q-value function.
        - target_net (NeuralNetwork): A second network used for calculating target values, mirroring the policy_net.
        - optimizer (torch.optim.Adam): The optimization algorithm.
        - memory (deque): A buffer for storing experiences.
        - gamma (float): The discount factor.
        - batch_size (int): The size of the batch of experiences sampled from memory.

    Parameters:
        - learning_rate (float): Learning rate for the optimizer.
        - discount_factor (float): Discount factor.
        - batch_size (int): Number of experiences to sample from memory.
        - **kwargs: Additional keyword arguments.

    Methods:
        - select_action: Select an action based on the current state and policy.
        - get_sample: Sample a batch of experiences from the replay buffer.
        - evaluate: Evaluate the performance of the agent in the environment.
    """

    def __init__(self, learning_rate=1e-3, discount_factor=1, batch_size=32, **kwargs):
        self.steps_done = 0
        self.learning_rate = learning_rate
        self.policy_net = NeuralNetwork(num_states=4, num_actions=2)
        self.target_net = NeuralNetwork(num_states=4, num_actions=2)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )  # Synchronize weights of target and policy network
        self.memory = deque([], maxlen=1000000)
        self.gamma = discount_factor
        self.batch_size = batch_size

    def select_action(self, state, policy_str="egreedy 0.05"):
        policy, value = policy_str.split(" ")
        policy_dict = {policy: float(value)}
        # Epsilon-greedy policy
        if list(policy_dict.keys())[0] == "egreedy":
            if np.random.rand() <= policy_dict["egreedy"]:
                return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
            else:
                with torch.no_grad():
                    action_values = self.policy_net(state)
                    return torch.argmax(action_values)
        # Greedy policy
        elif list(policy_dict.keys())[0] == "greedy":
            with torch.no_grad():
                action_values = self.policy_net(state)
                return torch.argmax(action_values)
        # Softmax policy
        elif list(policy_dict.keys())[0] == "softmax":
            with torch.no_grad():
                action_values = self.policy_net(state)
                # Probabilities of taking each action for state s
                p = torch.nn.functional.softmax(action_values, dim=1).numpy()
                action = torch.multinomial(torch.tensor(p), 1)
                return action

    def get_sample(self):
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
        - A list of experience tuples.
        """
        return random.sample(self.memory, self.batch_size)

    def evaluate(self, max_episode_length=501):
        """
        Evaluate the performance of the agent in the environment.
        This function runs the agent for a fixed number of evaluation episodes, using the current policy to select actions.
        Each episode runs for a maximum of 'max_episode_length' steps or until the episode terminates.
        The function calculates the average return across all evaluation episodes.

        Parameters:
        - max_episode_length (int): the maximum number of steps in an episode.

        Returns:
        - mean_return (float): the average return across all evaluation episodes.
        """
        returns = []
        for _ in range(num_eval_episodes):
            state = env.reset()[0]
            R_episode = 0
            for _ in range(max_episode_length):
                state = torch.from_numpy(state).float().unsqueeze(0)
                action = self.select_action(state, policy_str="greedy 0")
                next_state, reward, episode_done, episode_truncated, _ = env.step(
                    action.item()
                )
                R_episode += reward
                if episode_done or episode_truncated:
                    break
                state = next_state
            returns.append(R_episode)
        mean_return = np.mean(returns)
        return mean_return
