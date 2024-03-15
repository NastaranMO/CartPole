import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play, PlayPlot
#env = gym.make('CartPole-v1', render_mode='rgb_array')


mapping = {
            "a": np.array(0),
            "d": np.array(1),
            }

#default_action = np.array([0,0,0])

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew,]

plotter = PlayPlot(callback, 150, ["reward"])

env = gym.make("CartPole-v1", render_mode='rgb_array')
play(env, zoom=3, keys_to_action=mapping, callback=plotter.callback)