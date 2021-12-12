import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import binom, norm
import json
from tqdm import tqdm

sys.path.append('.')

from gillespie_algo import sir_simulate
from pmcmc import *
from abc_algo import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)

# noisy processs
data1 = np.array([[0, 0, 0]])
ratio_noise = 0.1
for t in range(dataset.shape[0]):
    sus = dataset.iloc[t, 1] + np.random.normal(0, ratio_noise * dataset.iloc[t, 1])
    inf = dataset.iloc[t, 2] + np.random.normal(0, ratio_noise * dataset.iloc[t, 2])
    rem = dataset.iloc[t, 3] + np.random.normal(0, ratio_noise * dataset.iloc[t, 3])
    data1 = np.append(data1, [[sus, inf, rem]], axis=0)
data1 = data1[1:]

# plt.plot(range(15), dataset.iloc[:, 1], label='Susceptible')
# plt.plot(range(15), dataset.iloc[:, 2], label='Infected')
# plt.plot(range(15), dataset.iloc[:, 3], label='Removed')
# plt.plot(range(15), data1[:, 0], label='Susceptible_noise')
# plt.plot(range(15), data1[:, 1], label='Infected_noise')
# plt.plot(range(15), data1[:, 2], label='Removed_noise')
# plt.legend()
# plt.show()

thetas, sampled_trajs = abc_algo(data1, 100, 200, {'beta': [0, 5], 'gamma': [0, 5]})
thetas = np.array([[thetas['beta'][i], thetas['gamma'][i]] for i in range(len(thetas['beta']))])

results_directory = "rejection_sampling_results/test0/"
graphs_directory = "rejection_sampling_results/test0/"

results_directory = "data/" + results_directory
graphs_directory = "graphs/" + graphs_directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# # save results
np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")

# # load results
# thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
# sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
# sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
# sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
# sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

## plot thetas
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 0],
    color="blue",
    linewidth=1,
    label="beta",
)
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma.png", bbox_inches='tight')
plt.show()

## plot thetas - scatterplot
plt.scatter(thetas[:, 0], thetas[:, 1])
plt.scatter(beta, gamma, color="red")
plt.savefig(graphs_directory + "beta_gamma2.png", bbox_inches='tight')
plt.show()

## plot thetas - histogram
plt.hist(thetas[:, 0], bins=10, alpha=0.5, label="beta")
plt.hist(thetas[:, 1], bins=10, alpha=0.5, label="gamma")
plt.axvline(beta)
plt.axvline(gamma)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma3.png", bbox_inches='tight')
plt.show()


## plot trajectories
lines1 = plt.plot(
    range(len(dataset)), sampled_trajs[:, :, 1].transpose(), "orange", linewidth=1
)
plt.plot(
    range(len(dataset)), sampled_trajs[:, :, 2].transpose(), "orange", linewidth=1
)
plt.plot(
    range(len(dataset)), sampled_trajs[:, :, 3].transpose(), "orange", linewidth=1
)
lines4 = plt.plot(range(len(dataset)), dataset.iloc[:, 1:], "k", linewidth=2)
lines5 = plt.plot(range(len(dataset)), data1, "k--", linewidth=2)
plt.legend(
    lines1[:1] + lines4 + lines5,
    ["simulated trajectories", "hidden compartments", "observed compartments"],
)
plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
plt.show()
