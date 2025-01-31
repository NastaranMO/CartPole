import numpy as np
import torch
from utils import smooth
import tqdm
import gymnasium as gym
import argparse

from DQA import DQN_Agent

env = gym.make("CartPole-v1")
# Default arguments for DQN-TN-ER
args = argparse.Namespace(
    ER=True,
    TN=True,
    anneal=True,
    num_episodes=200,
    eval_episodes=20,
    eval_interval=10,
    num_repetitions=20,
    lr=1e-4,
    # explr="softmax 0.1",
    explr="egreedy 0.3",
    gamma=1,
    steps=50,
    batch_size=32,
)


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
            agent.memory.append((state, action, next_state, reward))
            # Update policy
            if len(agent.memory) >= agent.batch_size:
                double_dqn(args, agent)
            state = next_state
        # Evaluate the performance every eval_interval episodes
        if e % args.eval_interval == 0:
            print("Episode: {}".format(e))
            returns = agent.evaluate()
            print(f"Evaluation: reward for episode {e} is {returns}")
            reward_means.append(returns)

    return reward_means


def double_dqn(args, agent):
    # Sample a batch of experiences
    print("batch size", agent.batch_size)
    experiences = agent.get_sample()
    states_tuple, actions_tuple, next_states_tuple, rewards_tuple = zip(
        *experiences
    )  # Unpack the batch
    # Convert to tensors
    states_batch = torch.cat(states_tuple)
    actions_batch = torch.cat(actions_tuple)
    rewards_batch = torch.cat(rewards_tuple)
    next_states_batch = torch.cat([s for s in next_states_tuple if s is not None])

    # Predict the Q-values
    predicted_q_values = agent.policy_net(states_batch).gather(1, actions_batch)
    agent.steps_done += 1
    # Calculate the target Q-values by Q-learning update rule
    # next_state_actions = torch.zeros(agent.batch_size)
    _, next_state_actions = agent.policy_net(next_states_batch).max(1)

    print("next_state_actions", next_state_actions)
    next_state_values = agent.target_net(next_states_batch).gather(
        1, next_state_actions.unsqueeze(1)
    )
    target_q_values = (next_state_values * agent.gamma) + rewards_batch

    # Update current policy
    # criterion = torch.nn.SmoothL1Loss() # Compute Huber loss <= works better
    criterion = torch.nn.MSELoss()
    loss = criterion(predicted_q_values, target_q_values.unsqueeze(1))
    agent.optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100) # Clip gradients
    agent.optimizer.step()

    # Syncronize target and policy network to stabilize learning
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
    average_returns_std = np.std(returns_over_repetitions, axis=0)
    average_returns = smooth(average_returns, smoothing_window)
    return np.array([episodes, average_returns, average_returns_std])


if __name__ == "__main__":
    average_over_repetitions(env, args)
