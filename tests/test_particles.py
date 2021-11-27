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

from gillespie_algo import sir_simulate
from pmcmc import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
parameters = [2, 1]

dataset = sir_simulate_discrete(y0, t, parameters[0], parameters[1])

# observed process - HMM
data1 = np.array([])
prob_obs = 0.1
for t in range(dataset.shape[0]):
    y = np.random.binomial(dataset.iloc[t, 2], prob_obs)
    data1 = np.append(data1, y)

# observed process - HMM + all
data2 = np.array([[0, 0, 0]])
prob_obs = 0.1
for t in range(dataset.shape[0]):
    sus = np.random.binomial(dataset.iloc[t, 1], prob_obs)
    inf = np.random.binomial(dataset.iloc[t, 2], prob_obs)
    rec = np.random.binomial(dataset.iloc[t, 3], prob_obs)
    data2 = np.append(data2, np.array([[sus, inf, rec]]), axis=0)
data2 = data2[1:]

# observed infected
data3 = np.array(np.round(dataset.iloc[:, 2]).astype(int))

# observed infected and removed
data4 = np.array(np.round(dataset.iloc[:, 1:-1]).astype(int))


# Test particles

## times
t = time.time()
zetas, hidden_process, ancestry_matrix = particle_filter(
    data2,
    np.array(parameters),
    False,
    .1,
    n_particles=100,
    n_population=4820,
    mu=20,
    jobs=1,
)
print(time.time() - t)

t = time.time()
zetas, hidden_process, ancestry_matrix = particle_filter(
    data2,
    np.array(parameters),
    observations=False,
    probs=.1,
    n_particles=1000,
    n_population=4820,
    mu=20,
    jobs=-1,
)
print(time.time() - t)


## visualise results per particle
for i in range(len(data2)):
    for j in range(100):
        plt.scatter(i, hidden_process[i, j, 1], color="black")
        if i > 0:
            i_ancestor = int(ancestry_matrix[i, j])
            plt.plot(
                range(i - 1, i + 1),
                [
                    hidden_process[i - 1, i_ancestor, 1],
                    hidden_process[i, j, 1],
                ],
                color="orange",
                linewidth=0.5,
            )
plt.plot(
    range(len(data2)), dataset.infected, color="blue", linewidth=2
)
plt.show()


## visualise path sampler
trajectory = particle_path_sampler(hidden_process, ancestry_matrix)
for j in range(trajectory.shape[1]):
    plt.plot(
        range(trajectory.shape[0]),
        trajectory[:, j],
        color="orange",
        linewidth=1,
    )
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