import os
import time
import argparse
import json
import numpy as np

import gymnasium as gym
from AgentTrainer import average_over_repetitions

ENV = gym.make("CartPole-v1")

def main(raw_args=None):
    '''
    Random search over

    Continuous hyperparamters
    gamma (0.9, 1)
    learning_rate log(-5, -2)
    epsilon (0, 0.3)
    temp (1, 10)

    Discrete hyperparameters:
    steps : 1 - 60
    batch size: [16, 32, 64, 128]
    '''
    # Create dictionary of random hyperparameters
    n = 1

    gamma_arr = (np.ones(shape=(n)) - np.random.rand(n) * 0.1)  #0 1 -> 0.9 1  
    learning_rate_arr = np.power(10, (np.random.rand(n)*-3 -2*np.ones(shape=(n))))
    batch_size = [np.random.choice([16, 32, 64, 128]) for i in range(n)] 
    steps = np.random.randint(1, 60, n)
    epsilon = np.random.rand(n) * 0.3
    temp =  np.power(10, np.random.rand(n) * 3 -2*np.ones(shape=(n)))

    policy = [np.random.choice(['egreedy', 'softmax']) for i in range(n)]

    value = []
    for i, pol in enumerate(policy):
        if pol == 'egreedy':
            value.append(f'egreedy {epsilon[i]}')
        elif pol == 'softmax':
            value.append(f'softmax {temp[i]}')
        else: 
            print('Garbage in programmeren sadge')


    dict_list = []
    for i in range(n):
        dict = {
            'ER':True,
            'TN':True,
            'anneal':True,
            'num_episodes':200,
            'eval_episodes':5,
            'eval_interval':10,
            'num_repetitions':5,
            'lr':learning_rate_arr[i],
            'explr': value[i],
            'gamma':gamma_arr[i],
            'steps' : int(steps[i]),
            'batchsize' : int(batch_size[i])
        }
        dict_list.append(dict)

    # Perform experiment
    episodes_list = []
    average_returns_list = []
    std_list = []
    for dict in dict_list:
        # set parameters
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
        # run experiment
        episodes, average_returns, std = average_over_repetitions(ENV, args)
        # store results
        episodes_list.append(episodes)
        average_returns_list.append(average_returns)
        std_list.append(std)
    e = np.array(episodes_list)
    a = np.array(average_returns_list)
    s = np.array(std_list)

    path = "./results/randomsearch"
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(path, [e, a, s])

if __name__ == "__main__":
    main()


