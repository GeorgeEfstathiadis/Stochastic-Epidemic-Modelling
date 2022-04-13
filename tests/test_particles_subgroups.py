import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')

from gillespie_algo import *
from pmcmc import *


population = np.array([[2000, 30, 0], [3000, 40, 0]])
beta = np.array([[5, 2], [1, 3]])
gamma = .5
t = np.linspace(0, 14, num=200)

# discrete sir with subgroups
data = sir_subgroups_simulate_discrete(population, t, beta, gamma)

plt.plot(data.time, data.susceptible0, label='s_0')
plt.plot(data.time, data.susceptible1, label='s_1')
plt.plot(data.time, data.infected0, label='i_0')
plt.plot(data.time, data.infected1, label='i_1')
plt.plot(data.time, data.removed0, label='r_0')
plt.plot(data.time, data.removed1, label='r_1')
plt.legend()
plt.show()

# HMM
# observed process - HMM + all
data2 = np.loadtxt('data\simulated_datasets\sir_subgrps.csv', delimiter=', ')

# particle filter
zetas, hidden_process, ancestry_matrix = particle_filter(data2, ModelType.SIR_SUBGROUPS, (beta, gamma), n_population=np.sum(population, axis=1), mu=population[:, 1], n_particles=150)

## results per particle viz
col = 5
for i in range(len(data2)):
    for j in range(150):
        plt.scatter(i, hidden_process[i, j, col], color="black")
        if i > 0:
            i_ancestor = int(ancestry_matrix[i, j])
            plt.plot(
                range(i - 1, i + 1),
                [
                    hidden_process[i - 1, i_ancestor, col],
                    hidden_process[i, j, col],
                ],
                color="orange",
                linewidth=0.5,
            )
plt.plot(
    range(len(data2)), data.iloc[:-1, col], color="blue", linewidth=2
)
plt.show()

## visualise path sampler
trajectories = [particle_path_sampler(hidden_process, ancestry_matrix) for _ in range(5)]
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
            range(data.shape[0]), data.iloc[:, j], color="blue", linewidth=1
        )
plt.show()

## test times 10 particles (0.66s)
t = time.time()
zetas, hidden_process, ancestry_matrix = particle_filter(data2, ModelType.SIR_SUBGROUPS, (beta, gamma), n_population=np.sum(population, axis=1), mu=population[:, 1], n_particles=10)
print(time.time() - t)
## test times 100 particles (5.34s)
t = time.time()
zetas, hidden_process, ancestry_matrix = particle_filter(data2, ModelType.SIR_SUBGROUPS, (beta, gamma), n_population=np.sum(population, axis=1), mu=population[:, 1], n_particles=100)
print(time.time() - t)

## test consistencies
likelihoods100 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(data2, ModelType.SIR_SUBGROUPS, (beta, gamma), n_population=np.sum(population, axis=1), mu=population[:, 1], n_particles=100)
    likelihoods100 = np.append(likelihoods100, zetas[-1])

likelihoods1000 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(data2, ModelType.SIR_SUBGROUPS, (beta, gamma), n_population=np.sum(population, axis=1), mu=population[:, 1], n_particles=1000)
    likelihoods1000 = np.append(likelihoods1000, zetas[-1])