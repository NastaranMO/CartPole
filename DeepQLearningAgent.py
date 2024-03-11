import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
from Helper import softmax, argmax

# Create a DQN agent
class DeepQLearningAgent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.model = self._build_DQN_model()
        self.memory = deque(maxlen=2000)
        
    def remember(self, s, a, r, s_prime, done):
        self.memory.append((s, a, r, s_prime, done))

    def _build_DQN_model(self):
        model = Sequential()

        model.add(Dense(68, input_dim=self.n_states, activation='relu', kernel_initializer= 'glorot_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer= 'glorot_uniform'))
        model.add(Dense(self.n_actions, activation='linear', kernel_initializer= 'glorot_uniform'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def select_action(self, s, policy="egreedy", epsilon=None, temp=None):
        if policy == "greedy":
            a = argmax(self.model.predict(s, verbose=0))
        elif policy == "egreedy":
            # Explore: take a random action
            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)
            # Exploit: take the greedy action
            else:
                a = argmax(self.model.predict(s, verbose=0))
        # elif policy == "softmax":
        #     if temp is None:
        #         raise KeyError("Provide a temperature")
        #     # Probabilities of taking each action for state s
        #     p = softmax(self.Q_sa[s,], temp)
        #     # Choose the action with the highest probability
        #     a = np.random.choice(self.n_actions, p=p)
              
        return a
    


    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_predicted = self.model.predict(state, verbose=0)
            target_predicted[0][action] = target
            self.model.fit(state, target, epochs=1, verbose=0)

    def evaluate(self, eval_env, number_of_eval_episodes = 30, max_episode_length=500):
        # Evaluate the performance for 30 episodes
        total_rewards = []
        
        for e in range(number_of_eval_episodes):
            s = eval_env.reset()[0]
            s = np.reshape(s, [1, self.n_states])
            episode_rewards = 0

            for t in range(max_episode_length):
                # Select the best action
                a = self.select_action(s, policy="greedy")
                # Take the action
                s_prime, r, done, _, _ = eval_env.step(a)
                episode_rewards += r
                s_prime = np.reshape(s_prime, [1, self.n_states])
                if done:
                    break
                else:
                    s = s_prime

            total_rewards.append(episode_rewards)
        
        # np.sum or mean??
        return np.mean(total_rewards)