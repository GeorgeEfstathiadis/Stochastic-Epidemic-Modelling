import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymc3 as pm3
import arviz as az

sys.path.append('.')

from gillespie_algo import sir_simulate
from pmcmc import *
from helpers import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)
data1 = np.loadtxt(f"data/simulated_datasets/sir_noisy.csv", delimiter=", ")

timepoinst = [11, 7, 3]
tmp = timepoinst[2]
data2 = data1[:tmp, :]
directory = f"timepoints_level/tmp_{tmp}/"

results_directory = "data/" + directory
graphs_directory = "graphs/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# load results
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

## burn-in
burn_in = 1000

thetas2 = thetas[burn_in:, :]
sampled_trajs2 = sampled_trajs[:, burn_in:, :]

## apply thinning
thinning = 1

thetas3 = thetas2[::thinning]
sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

def traj_inf(par, population, tmp):
    simulation = sir_simulate(population, par, 15-tmp, False)
    traj = np.zeros((15-tmp, 3))
    for j in range(15-tmp):
        try:
            idxf = np.argwhere(np.floor(simulation['time']) == j)[-1][0]
            traj[j, :] = simulation['s'][idxf], simulation['i'][idxf], simulation['r'][idxf]
        except IndexError:
            traj[j, :] = traj[j-1, :]
    return traj

predicted_trajs = Parallel(n_jobs=4)(
                delayed(traj_inf)(
                    thetas3[j],
                    sampled_trajs3[-1, j],
                    tmp,
                )
                for j in range(len(thetas3))
            )
predicted_trajs = np.array(predicted_trajs)
full_trajs = np.zeros((len(thetas3), 15, 3))
for i in range(len(thetas3)):
    full_trajs[i] = np.concatenate((sampled_trajs3[:,i], predicted_trajs[i]), axis=0)

means = [np.mean(full_trajs[:, :, i], axis=0) for i in range(3)]
percentiles = [np.percentile(full_trajs[:, :, i], [5, 95], axis=0) for i in range(3)]
lines1 = plt.plot(
    range(len(data1)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data1)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(3):
    plt.plot(
        range(len(data1)), means[i], "red", linewidth=1.5
    )
    plt.fill_between(range(len(data1)), percentiles[i][0], percentiles[i][1], color='orange', alpha=.8)
lines2 = plt.plot(range(len(data1)), dataset.iloc[:, 1], "b", linewidth=2)
lines3 = plt.plot(range(len(data1)), dataset.iloc[:, 2], "g", linewidth=2)
lines4 = plt.plot(range(len(data1)), dataset.iloc[:, 3], "c", linewidth=2)
lines5 = plt.plot(range(len(data1[:tmp+1])), data1[:tmp+1, 0], "b--", linewidth=2)
lines6 = plt.plot(range(len(data1[:tmp+1])), data1[:tmp+1, 1], "g--", linewidth=2)
lines7 = plt.plot(range(len(data1[:tmp+1])), data1[:tmp+1, 2], "c--", linewidth=2)
plt.axvline(x=tmp, color='black', linestyle='--')
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
    ["particle trajectories", "hidden susceptible", "hidden infected",
    "hidden removed", "observed susceptible", "observed infected",
    "observed removed"],
)
plt.savefig(graphs_directory + "trajectories3.png", bbox_inches='tight')
plt.show()

