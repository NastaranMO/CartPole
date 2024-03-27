import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from plotter import LearningCurvePlot

Plot = LearningCurvePlot()

episodes_1_return = np.load("episodes_Nastaran.npy")
episodes_2_return = np.load("episodes_Tom.npy")
episodes_3_return = np.load("episodes_Rajiv.npy")

average_1_returns = np.load("returns_Nastaran.npy")
average_2_returns = np.load("returns_Tom.npy")
average_3_returns = np.load("returns_Rajiv.npy")

outputfile_1 = open("outputfile_Nastaran.json", "r")
outputfile_2 = open("outputfile_Tom.json", "r")
outputfile_3 = open("outputfile_Rajiv.json", "r")

dict_list_1 = json.load(outputfile_1)
dict_list_2 = json.load(outputfile_2)
dict_list_3 = json.load(outputfile_3)

#episodes_return = np.concatenate(
#    (episodes_1_return, episodes_2_return, episodes_3_return)
#)

# average_returns = np.concatenate(
#     (average_1_returns, average_2_returns, average_3_returns)
# )

# Nastaran: i = 0

# dict_list = [dict_list_1[0]]
# episodes_return = [episodes_1_return[0]]
# average_returns = [average_1_returns[0]] 

dict_list = dict_list_2
episodes_return = episodes_2_return
average_returns = average_2_returns
#print(dict_list)

i = 0
indices = [6, 16]

for dict in dict_list:
    args = argparse.Namespace(
        ER=dict["ER"],
        TN=dict["TN"],
        anneal=dict["anneal"],
        num_episodes=dict["num_episodes"],
        eval_episodes=dict["eval_episodes"],
        eval_interval=dict["eval_interval"],
        num_repetitions=dict["num_repetitions"],
        lr=dict["lr"],
        explr=dict["explr"],
        gamma=1,
        steps=dict["steps"],
        batch_size=dict["batchsize"],
    )

    Plot.add_curve(
        episodes_return[indices[i]],
        average_returns[indices[i]],
        label=f"experiment {indices[i]+1}",
    )
    i += 1
    if i >= len(indices):
        break

Plot.save(name="test.png")