import os
import sys

import numpy as np
sys.path.append('.')

from gillespie_algo import sir_simulate
from pmcmc import *


t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)

results_directory = "pmcmc_sir/test2/"
results_directory = "data/" + results_directory
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
thetas2 = thetas[100:, :]
thetas3 = thetas2[::20]
thetas_unique = np.unique(thetas3, axis=0)
theta_proposal = thetas[-1].tolist()
# sigma = np.cov(thetas_unique.T, ddof=0)
sigma = np.array([
    [8.56210710e-03, 4.96880438e-03],
    [4.96880438e-03, 3.20130528e-03],
])
h = 5

prob = .05

data2 = np.loadtxt(f"data/simulated_datasets/sir_under_{prob}.csv", delimiter=", ")
thetas, likelihoods, sampled_trajs = particle_mcmc(
    data2,
    ModelType.SIR,
    theta_proposal,
    h,
    adaptive=True,
    sigma=sigma,
    n_chains=6000,
    observations=False,
    probs=prob,
    n_particles=100,
    n_population=4820,
    mu=20,
    jobs=-1,
)
directory = f"under_level2/under_{prob}/"

results_directory = "data/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# # save results
np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")
