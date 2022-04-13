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
max_time = 14
t = np.linspace(0, max_time, num=200)

# testing
beta_test = np.array([[6.8338187, 0.95137037], [0.53205031, 3.29273452]])

data = sir_subgroups_simulate_discrete(population, t, beta, gamma)

conditions = sir_subgroups_simulate(
    population, beta,
    gamma, max_time, False
    )

plt.plot(conditions['time'], conditions['s_0'], label='s_0_simulated')
plt.plot(conditions['time'], conditions['i_0'], label='i_0_simulated')
plt.plot(conditions['time'], conditions['r_0'], label='r_0_simulated')
plt.plot(conditions['time'], conditions['s_1'], label='s_1_simulated')
plt.plot(conditions['time'], conditions['i_1'], label='i_1_simulated')
plt.plot(conditions['time'], conditions['r_1'], label='r_1_simulated')
plt.plot(data.time, data.susceptible0, label='s_0')
plt.plot(data.time, data.susceptible1, label='s_1')
plt.plot(data.time, data.infected0, label='i_0')
plt.plot(data.time, data.infected1, label='i_1')
plt.plot(data.time, data.removed0, label='r_0')
plt.plot(data.time, data.removed1, label='r_1')
plt.legend()
plt.show()

## discrete
data = sir_subgroups_simulate_discrete(population, t, beta, gamma)
data2 = np.array([[0 for _ in range(data.shape[1] - 1)]])
prob_obs = 0.1
for idx in range(data.shape[0]):
    res = []
    for c in range(data.shape[1]-1):
        res.append(np.random.binomial(data.iloc[idx, c], prob_obs))
    data2 = np.append(data2, np.array([res]), axis=0)
data2 = data2[1:]

np.savetxt('data\simulated_datasets\sir_subgrps.csv', data2, delimiter=', ')

## simulated data - discrete
conditions = sir_subgroups_simulate(
    population, beta,
    gamma, max_time, False
    )
data = pd.DataFrame(conditions)
data.time = np.floor(data.time).astype(int)
ids = []
for t in range(max_time):
    id_sel = next(idx for idx in range(data.shape[0]) if data.loc[idx, 'time'] == t)
    ids.append(id_sel)
data = data.iloc[ids, :].reset_index(drop=True)

data2 = np.array([[0 for _ in range(data.shape[1] - 1)]])
prob_obs = 0.1
for idx in range(data.shape[0]):
    res = []
    for c in range(1, data.shape[1]):
        res.append(np.random.binomial(data.iloc[idx, c], prob_obs))
    data2 = np.append(data2, np.array([res]), axis=0)
data2 = data2[1:]
np.savetxt('data\simulated_datasets\sir_subgrps.csv', data2, delimiter=', ')