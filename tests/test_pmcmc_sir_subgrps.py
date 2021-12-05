import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import binom, norm
from tqdm import tqdm

sys.path.append('.')

from gillespie_algo import *
from pmcmc import *

population = np.array([[1000, 15, 0], [1500, 20, 0], [2000, 25, 0]])
beta = np.array([[15, 5, 2], [1, 8, 4], [1, 5, 10]])
gamma = .5
t = np.linspace(0, 10, num=200)

data = sir_subgroups_simulate_discrete(population, t, beta, gamma)
data2 = np.array([[0 for _ in range(data.shape[1] - 1)]])
prob_obs = 0.1
for t in range(data.shape[0]):
    res = []
    for c in range(data.shape[1]-1):
        res.append(np.random.binomial(data.iloc[t, c], prob_obs))
    data2 = np.append(data2, np.array([res]), axis=0)
data2 = data2[1:]

parameters0 = [5, 1, 1, 1, 5, 1, 1, 1, 5, 1]
sigma = np.eye(len(parameters0))
h = .5
n_population = np.sum(population, axis=1)
mu = population[:, 1].tolist()

thetas, likelihoods, sampled_trajs = particle_mcmc(
    data2,
    parameters0,
    h,
    sigma=sigma,
    n_population=n_population,
    mu=mu,
    n_particles=100,
    )