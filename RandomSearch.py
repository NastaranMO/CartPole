import os
import time
import argparse
import json
import numpy as np

import gymnasium as gym
from AgentTrainer import average_over_repetitions

# temp, temp2 = "egreedy 0.3".split(' ')
# policy_dict = {temp: float(temp2)}
# print(policy_dict)

# path = "./results"
# if not os.path.exists(path):
#     os.mkdir(path)
ENV = gym.make("CartPole-v1")

def main(raw_args=None):
    f = open('outputfile_Tom.json')
    dict_list = json.load(f)
    dict_list = [dict_list[0]]
    episodes_list = []
    average_returns_list = []

    for dict in dict_list:
        args = argparse.Namespace(
        ER=dict['ER'],
        TN=dict['TN'],
        anneal=dict['anneal'],
        num_episodes=dict['num_episodes'],
        eval_episodes=dict['eval_episodes'],
        eval_interval=dict['eval_interval'],
        num_repetitions=dict['num_repetitions'],
        lr=dict['lr'],
        explr=dict['explr'],
        gamma=dict['gamma'],
        steps=dict['steps'],
        batch_size=dict['batchsize'])
        
        episodes, average_returns = average_over_repetitions(ENV, args)
        episodes_list.append(episodes)
        average_returns_list.append(average_returns)
    e = np.array(episodes_list)
    a = np.array(average_returns_list)
    np.save('episodes_Tom', e)
    np.save('returns_Tom', a)

if __name__ == "__main__":
    main()


