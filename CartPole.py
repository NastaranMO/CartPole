import gymnasium as gym
import numpy as np
from DeepQLearningAgent import DeepQLearningAgent
import matplotlib
import matplotlib.pyplot as plt
from Helper import LearningCurvePlot, smooth
import tensorflow as tf
tf. config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

smoothing_window = 19
number_of_repetitions = 20
number_of_episodes = 100
l = number_of_episodes // 10
result = np.zeros((number_of_repetitions, l))

env = gym.make("CartPole-v1")
env_eval = gym.make("CartPole-v1")

for rep in range(number_of_repetitions):
    mean_rewards = []
    DQN_agent = DeepQLearningAgent(env.observation_space.shape[0], env.action_space.n)
    # Train the agent
    for ep in range(number_of_episodes):
        s = env.reset()[0]
        s = np.reshape(s, [1, DQN_agent.n_states])
        done = False
        while not done:
            a = DQN_agent.select_action(s)
            s_prime, r, done, _, _ = env.step(a)
            s_prime = np.reshape(s_prime, [1, DQN_agent.n_states])
            DQN_agent.remember(s, a, r, s_prime, done)
            
            if done:
                break
            else:
                s = s_prime
            
        DQN_agent.replay(32)
        # Evaluate the agent
        if ep % 10 == 0:
            print("Episode: {}".format(ep))
            mean_reward = DQN_agent.evaluate(env_eval)
            mean_rewards.append(mean_reward)
            print(f"Reward for episode {ep} is {mean_reward}")
    result[rep] = np.array(mean_rewards)
    # learning_curve = np.mean(np.array(result),axis=0)
    # if smoothing_window is not None: 
    #     learning_curve = smooth(learning_curve,smoothing_window)
    # DQN_agent.save_model(f"models/DQN_agent_{rep}.h5")
    # print(f"Model {rep} trained and saved")

# Plotting the average performance
smoothed_result = smooth(np.mean(result, axis=0), smoothing_window)
ks = np.arange(l) * 10
# avs = np.mean(result, axis=0)
maxs = np.max(result, axis=0)
mins = np.min(result, axis=0)

plt.fill_between(ks, mins, maxs, alpha=0.1)
plt.plot(ks, smoothed_result, '-o', markersize=1)

plt.xlabel('Episode', fontsize=15)
plt.ylabel('Avg. Return', fontsize=15)
plt.savefig('plot-replay-r5-ep100-w19.png', dpi=300)