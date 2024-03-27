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
        "--ER", default=True, action="store_true", help="Experience Replay"
    )
    # TODO: I changed the default value of TN to True
    argparser.add_argument(
        "--TN", default=True, action="store_true", help="Target Network"
    )

    argparser.add_argument(
        "--anneal", default=False, action="store_true", help="annealing"
    )
    argparser.add_argument(
        "--num_episodes", default=200, type=int, help="training episodes"
    )
    argparser.add_argument(
        "--steps", default=50, type=int, help="steps to update target network"
    )
    argparser.add_argument(
        "--eval_episodes", default=20, type=int, help="Evaluation episodes"
    )
    argparser.add_argument(
        "--eval_interval", default=10, type=int, help="Evaluation Interval"
    )
    # TODO: I changed the default value of num_repetitions to 5
    argparser.add_argument("--num_repetitions", default=5, type=int, help="repetions")
    argparser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
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
    agent = DQN_Agent(**vars(args))
    reward_means = []
    for e in range(args.num_episodes):
        state = env.reset()[0]  # Sample initial state
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_done = False  # Completed the episode
        episode_truncated = False  # For example reaching the maximum number of steps

        while not (episode_done or episode_truncated):
            # Do the annealing gradually to reduce the exploration
            if args.anneal:
                args.explr = linear_anneal(
                    t=e, T=args.num_episodes, start=1, final=0.05, percentage=0.5
                )
                args.explr = "egreedy " + str(args.explr)
            # Action selection
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
            if args.ER:
                agent.memory.append((state, action, next_state, reward))
            # Update policy
            if is_pure_DQN(args.ER, args.TN):
                pure_DQN(agent, state, action, next_state, reward)
            elif args.ER == False and args.TN == True:
                dqn_TN(agent, state, action, next_state, reward, args)
            elif len(agent.memory) >= agent.batch_size and args.ER:
                dqn_er(args, agent)
            state = next_state
        # Evaluate the performance every eval_interval episodes
        if e % args.eval_interval == 0:
            print("Episode: {}".format(e))
            returns = agent.evaluate()
            print(f"Evaluation: reward for episode {e} is {returns}")
            reward_means.append(returns)

    return reward_means


def is_pure_DQN(ER, TN):
    return ER == False and TN == False


def dqn_er(args, agent):
    # Sample a batch of experiences
    experiences = agent.get_sample()
    states_tuple, actions_tuple, next_states_tuple, rewards_tuple = zip(
        *experiences
    )  # Unpack the batch
    # Convert to tensors
    states_batch = torch.cat(states_tuple)
    actions_batch = torch.cat(actions_tuple)
    rewards_batch = torch.cat(rewards_tuple)

    # Calculate the current estimated Q-values by following the current policy
    current_q_values = agent.policy_net(states_batch).gather(1, actions_batch)
    agent.steps_done += 1
    # Calculate the target Q-values by Q-learning update rule
    next_state_values = torch.zeros(agent.batch_size)
    if args.TN:
        net = agent.target_net
    else:
        net = agent.policy_net
    for i in range(len(next_states_tuple)):
        if next_states_tuple[i] is not None:
            with torch.no_grad():  # Speed up the computation by not tracking gradients
                next_state_values[i] = agent.policy_net(next_states_tuple[i]).max(1)[0]
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
    if agent.steps_done % args.steps == 0 & args.TN:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())


def pure_DQN(agent, state, action, next_state, reward):
    current_q_values = agent.policy_net(state).gather(1, action)
    next_state_values = torch.zeros(1)
    if next_state is not None:
        with torch.no_grad():
            next_state_values = agent.policy_net(next_state).max(1)[0].detach()
    target_q_values = (next_state_values * agent.gamma) + reward

    criterion = torch.nn.MSELoss()
    loss = criterion(current_q_values, target_q_values.unsqueeze(1))
    agent.optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100) # Clip gradients
    agent.optimizer.step()


def dqn_TN(agent, state, action, next_state, reward, args):
    current_q_values = agent.policy_net(state).gather(1, action)
    next_state_values = torch.zeros(1)
    if next_state is not None:
        with torch.no_grad():
            next_state_values = agent.target_net(next_state).max(1)[0].detach()
    target_q_values = (next_state_values * agent.gamma) + reward

    criterion = torch.nn.MSELoss()
    loss = criterion(current_q_values, target_q_values.unsqueeze(1))
    agent.optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100) # Clip gradients
    agent.optimizer.step()
    if agent.steps_done % args.steps == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())


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
