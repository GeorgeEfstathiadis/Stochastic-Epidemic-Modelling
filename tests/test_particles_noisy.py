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
from helpers import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)
# data2 = np.array([[0, 0, 0]])
# prob_obs = 0.01
# for t in range(dataset.shape[0]):
#     sus = np.random.normal(dataset.iloc[t, 1], prob_obs * dataset.iloc[t, 1])
#     inf = np.random.normal(dataset.iloc[t, 2], prob_obs * dataset.iloc[t, 2])
#     rec = np.random.normal(dataset.iloc[t, 3], prob_obs * dataset.iloc[t, 3])
#     data2 = np.append(data2, np.array([[sus, inf, rec]]), axis=0)
# data2 = data2[1:].astype(int)
# np.savetxt("data/simulated_datasets/sir_underreported.csv", data2, delimiter=", ")
data1 = np.loadtxt("data/simulated_datasets/sir_noisy.csv", delimiter=", ")


## consistent likelihoods
likelihoods10 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(
        data1,
        ModelType.SIR,
        np.array([beta, gamma]),
        True,
        .1,
        n_particles=10,
        n_population=4820,
        mu=20,
        jobs=-1,
    )
    likelihoods10 = np.append(likelihoods10, zetas[-1])

likelihoods100 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(
        data1,
        ModelType.SIR,
        np.array([beta, gamma]),
        True,
        .1,
        n_particles=100,
        n_population=4820,
        mu=20,
        jobs=-1,
    )
    likelihoods100 = np.append(likelihoods100, zetas[-1])

likelihoods1000 = np.array([])
for _ in range(10):
    zetas, _, _ = particle_filter(
        data1,
        ModelType.SIR,
        np.array([beta, gamma]),
        True,
        .1,
        n_particles=1000,
        n_population=4820,
        mu=20,
        jobs=-1,
    )
    likelihoods1000 = np.append(likelihoods1000, zetas[-1])

np.mean(likelihoods10)
np.var(likelihoods10)

np.mean(likelihoods100)
np.var(likelihoods100)

np.mean(likelihoods1000)
np.var(likelihoods1000)