import numpy as np
import argparse
import json

'''
First random search over: 
Continuous hyperparamters
gamma (0.9, 1)
learning_rate log(-5, -2)
epsilon (0, 0.3)
temp (1, 10)

Ablation studies:
ER, TN, Anneal 
'''
n = 20

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
#print(dict_list[30])

with open('outputfile_Nastaran.json', 'w') as fout:
    json.dump(dict_list, fout)

# amnt of configs: 20 per person

# 3 repetitions per config

# implement annealing 

# TODO 
# AgentTrainer.py: implement training loop with logs 
# Make script for running different arguments




