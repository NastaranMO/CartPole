import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import torch.optim as optim
from collections import deque
import tqdm
import random
import gymnasium as gym
import os

from DQA import DQN_Agent
from plotter import LearningCurvePlot
from utils import smooth, linear_anneal


def main(raw_args=None):

    ##############################################################################################################
    # Parsing command line arguments
    ##############################################################################################################

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ER", default=False, action="store_true", help="Experience Replay"
    )
    argparser.add_argument(
        "--TN", default=False, action="store_true", help="Target Network"
    )
    argparser.add_argument(
        "--anneal", default=False, action="store_true", help="annealing"
    )
    argparser.add_argument(
        "--num_episodes", default=200, type=int, help="training episodes"
    )
    argparser.add_argument(
        "--eval_episodes", default=20, type=int, help="Evaluation episodes"
    )
    argparser.add_argument(
        "--eval_interval", default=10, type=int, help="Evaluation Interval"
    )
    argparser.add_argument("--num_repetitions", default=20, type=int, help="repetions")
    argparser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    argparser.add_argument(
        "--explr", default="egreedy 0.3", type=str, help="Exploration Strategy"
    )
    argparser.add_argument("--gamma", default=1, type=float, help="Discount Factor")

    args = argparser.parse_args(raw_args)

    print(f"Running with arguments: {args}")

    ##############################################################################################################
    # Initialising Environment
    ##############################################################################################################

    env = gym.make("CartPole-v1")

    ##############################################################################################################
    # Running Experiment
    ##############################################################################################################

    results = average_over_repetitions(env, args)

    ##############################################################################################################
    # Saving Results
    ##############################################################################################################

    path = "./results"
    if not os.path.exists(path):
        os.mkdir(path)
    path += f"/{str(args)}"
    np.save(path, results)


def DQN_learning(env, args):

    agent = DQN_Agent(learning_rate=args.lr)
    reward_means = []
    for e in range(args.num_episodes):
        state = env.reset()[0]  # Sample initial state
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_done = False  # Completed the episode
        episode_truncated = False  # For example reaching the maximum number of steps

        while not (episode_done or episode_truncated):
            if args.anneal:
                args.explr = linear_anneal(
                    t=e, T=args.num_episodes, start=1, final=0.05, percentage=0.5
                )
                args.explr = "egreedy " + str(args.explr)
            action = agent.select_action(state, policy_str=args.explr).reshape(
                1, 1
            )  # Sample action (e.g, epsilon-greedy)
            action_index = action.item()
            next_state, reward, episode_done, episode_truncated, _ = env.step(
                action_index
            )  # Simulate environment
            reward = torch.tensor([reward])
            next_state = (
                None
                if episode_done
                else torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            )  # If the epsidoe terminates no next state

            # Store experience in buffer
            agent.memory.append((state, action, next_state, reward))
            state = next_state

            # Sample a batch of experiences
            if len(agent.memory) >= agent.batch_size:
                experiences = agent.get_sample()
                states_tuple, actions_tuple, next_states_tuple, rewards_tuple = zip(
                    *experiences
                )  # Unpack the batch
                # Convert to tensors
                states_batch = torch.cat(states_tuple)
                actions_batch = torch.cat(actions_tuple)
                rewards_batch = torch.cat(rewards_tuple)

                # Calculate the current estimated Q-values by following the current policy
                current_q_values = agent.policy_net(states_batch).gather(
                    1, actions_batch
                )

                # Calculate the target Q-values by Q-learning update rule
                next_state_values = torch.zeros(agent.batch_size)
                for i in range(len(next_states_tuple)):
                    if next_states_tuple[i] is not None:
                        with torch.no_grad():  # Speed up the computation by not tracking gradients
                            next_state_values[i] = agent.target_net(
                                next_states_tuple[i]
                            ).max(1)[0]
                target_q_values = (next_state_values * agent.gamma) + rewards_batch

                # Update current policy
                # criterion = torch.nn.SmoothL1Loss() # Compute Huber loss <= works better
                criterion = torch.nn.MSELoss()
                loss = criterion(current_q_values, target_q_values.unsqueeze(1))
                agent.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100) # Clip gradients
                agent.optimizer.step()

                # Syncronize target and policy network to stabilize learning
                agent.steps_done += 1
                if agent.steps_done % 50 == 0:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # Evaluate the performance every eval_interval episodes
        if e % args.eval_interval == 0:
            print("Episode: {}".format(e))
            returns = agent.evaluate()
            print(f"Evaluation: reward for episode {e} is {returns}")
            reward_means.append(returns)

    return reward_means


def average_over_repetitions(env, args):
    smoothing_window = 9

    print(f"Running with arguments =====>: {args}")
    num_evaluation_points = args.num_episodes // args.eval_interval
    returns_over_repetitions = np.zeros((args.num_repetitions, num_evaluation_points))

    for i in tqdm.tqdm(range(args.num_repetitions)):
        returns = DQN_learning(env, args)
        returns_over_repetitions[i] = np.array(returns)

    # Plotting the average performance
    episodes = np.arange(num_evaluation_points) * args.eval_interval
    average_returns = np.mean(returns_over_repetitions, axis=0)
    average_returns = smooth(average_returns, smoothing_window)
    return np.array([episodes, average_returns])


if __name__ == "__main__":
    main()
