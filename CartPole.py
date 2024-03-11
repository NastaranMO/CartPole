import gymnasium as gym
import numpy as np
from DeepQLearningAgent import DeepQLearningAgent
import matplotlib
import matplotlib.pyplot as plt

number_of_repetitions = 20
number_of_episodes = 200
l = number_of_episodes // 10
result = np.zeros((number_of_repetitions, l))

env = gym.make("CartPole-v1")
env_eval = gym.make("CartPole-v1")

for rep in range(number_of_repetitions):
    mean_rewards = []
    DQN_agent = DeepQLearningAgent(env.observation_space.shape[0], env.action_space.n)
    # Train the agent
    for ep in range(number_of_episodes):
        s = env.reset()
        s = np.reshape(s, [1, DQN_agent.n_states])
        done = False
        while not done:
            a = DQN_agent.select_action(s, policy="egreedy", epsilon=0.1)
            s_prime, r, done, _ = env.step(a)
            s_prime = np.reshape(s_prime, [1, DQN_agent.n_states])
            DQN_agent.remember(s, a, r, s_prime, done)
            
            if done:
                break
            else:
                s = s_prime
            
        # DQN_agent.replay()
        # Evaluate the agent
        if ep % 10 == 0:
            mean_reward = DQN_agent.evaluate(env_eval)
            mean_rewards.append(mean_reward)

    result[rep] = np.array(mean_rewards)
    
    # DQN_agent.save_model(f"models/DQN_agent_{rep}.h5")
    # print(f"Model {rep} trained and saved")

# Plotting the average performance
ks = np.arange(l) * 10
avs = np.mean(result, axis=0)
maxs = np.max(result, axis=0)
mins = np.min(result, axis=0)

plt.fill_between(ks, mins, maxs, alpha=0.1)
plt.plot(ks, avs, '-o', markersize=1)

plt.xlabel('Episode', fontsize=15)
plt.ylabel('Avg. Return', fontsize=15)