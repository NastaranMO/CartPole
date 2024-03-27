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

#gamma_arr = (np.ones(shape=(n)) - np.random.rand(n) * 0.1)  #0 1 -> 0.9 1  
learning_rate_arr = np.power(10, (np.random.rand(n)*-3 -2*np.ones(shape=(n))))
batch_size = [np.random.choice([16, 32, 64, 128]) for i in range(n)] 
steps = np.random.randint(20, 70, n)
epsilon = np.random.rand(n) * 0.3
temp =  np.power(10, np.random.rand(n) * 3 -2*np.ones(shape=(n)))

# policy = [np.random.choice(['egreedy', 'softmax']) for i in range(n)]
policy = ['egreedy' for i in range(n)]

value = []
for i, pol in enumerate(policy):
    if pol == 'egreedy':
        value.append(f'egreedy {epsilon[i]}')
    elif pol == 'softmax':
        value.append(f'softmax {temp[i]}')
    else: 
        print('Garbage in programmeren sadge')

f = open('outputfile_Tom.json')
dict_list = json.load(f)
dict = dict_list[6]

ER = [False, False, True, True]
TN = [False, True, False, True]

dict_list = []
for i in range(4):
    dict_1 = {
        'ER':ER[i],
        'TN':TN[i],
        'anneal':True,
        'num_episodes': dict['num_episodes'],
        'eval_episodes': 20,
        'eval_interval':dict['eval_interval'],
        'num_repetitions': 20,
        'lr':0.001,
        'explr': 'egreedy ',
        'gamma': dict['gamma'],
        'steps' :dict['steps'],
        'batch_size': dict['batchsize']
    }
    dict_list.append(dict_1)
#print(dict_list[30])

for i in range(n):
    dict_1 = {
        'ER':True,
        'TN':True,
        'anneal':True,
        'num_episodes': 200,
        'eval_episodes': 20,
        'eval_interval':dict['eval_interval'],
        'num_repetitions': 20,
        'lr':0.001,
        'explr': dict['explr'],
        'gamma': dict['gamma'],
        'steps' :dict['steps'],
        'batch_size': dict['batchsize']
    }


# print(dict_list)
with open('outputfile_ablation.json', 'w') as fout:
    json.dump(dict_list, fout)

# amnt of configs: 20 per person

# 3 repetitions per config

# implement annealing 

# TODO 
# AgentTrainer.py: implement training loop with logs 
# Make script for running different arguments




