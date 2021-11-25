import os
import time
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import binom, norm
from tqdm import tqdm

from gillespie_algo import seir_simulate
from pmcmc import *

t = np.linspace(0, 10, num=200)
y0 = (4800, 0, 20, 0)
parameters = [4, 1, 1]

dataset = seir_simulate_discrete(y0, t, parameters[0], parameters[1], parameters[2])

# plot dataset
plt.plot(dataset.time, dataset.susceptible, label="Susceptible")
plt.plot(dataset.time, dataset.exposed, label="Exposed")
plt.plot(dataset.time, dataset.infected, label="Infected")
plt.plot(dataset.time, dataset.removed, label="Removed")
plt.legend()
plt.show()

# observed process - HMM + all
data2 = np.array([[0, 0, 0, 0]])
prob_obs = 0.1
for t in range(dataset.shape[0]):
    sus = np.random.binomial(dataset.iloc[t, 1], prob_obs)
    exp = np.random.binomial(dataset.iloc[t, 2], prob_obs)
    inf = np.random.binomial(dataset.iloc[t, 3], prob_obs)
    rec = np.random.binomial(dataset.iloc[t, 4], prob_obs)
    data2 = np.append(data2, np.array([[sus, exp, inf, rec]]), axis=0)
data2 = data2[1:]

# test gillespie algo
sampled_traj = seir_simulate(y0, np.array(parameters), 10, False)
plt.plot(sampled_traj['time'], sampled_traj['s'], label="Susceptible")
plt.plot(sampled_traj['time'], sampled_traj['e'], label="Exposed")
plt.plot(sampled_traj['time'], sampled_traj['i'], label="Infected")
plt.plot(sampled_traj['time'], sampled_traj['r'], label="Removed")
plt.legend()
plt.show()


# Test particles
zetas, hidden_process, ancestry_matrix = particle_filter(
    data2,
    np.array(parameters),
    XYTransition(False, p=0.1),
    n_particles=1000,
    n_population=4820,
    mu=20,
    jobs=-1,
)
## visualise results per particle
for i in range(len(data2)):
    for j in range(100):
        plt.scatter(i, hidden_process[i, j, 2], color="black")
        if i > 0:
            i_ancestor = int(ancestry_matrix[i, j])
            plt.plot(
                range(i - 1, i + 1),
                [
                    hidden_process[i - 1, i_ancestor, 2],
                    hidden_process[i, j, 2],
                ],
                color="orange",
                linewidth=0.5,
            )
plt.plot(
    range(len(data2)), dataset.infected, color="blue", linewidth=2
)
plt.show()


## visualise path sampler
trajectories = [particle_path_sampler(hidden_process, ancestry_matrix) for _ in range(20)]
for trajectory in trajectories:
    for j in range(trajectory.shape[1]):
        plt.plot(
            range(trajectory.shape[0]),
            trajectory[:, j],
            color="orange",
            linewidth=1,
        )
for j in range(trajectory.shape[1]):
        plt.plot(
            range(trajectory.shape[0]), dataset.iloc[:, j+1], color="blue", linewidth=1
        )
plt.show()

## consistent likelihoods
likelihoods100 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(
        data2,
        parameters,
        XYTransition(False, p=0.1),
        n_particles=100,
        n_population=4820,
        mu=20,
        jobs=-1,
    )
    likelihoods100 = np.append(likelihoods100, zetas[-1])

likelihoods1000 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(
        data2,
        parameters,
        XYTransition(False, p=0.1),
        n_particles=1000,
        n_population=4820,
        mu=20,
        jobs=-1,
    )
    likelihoods1000 = np.append(likelihoods1000, zetas[-1])