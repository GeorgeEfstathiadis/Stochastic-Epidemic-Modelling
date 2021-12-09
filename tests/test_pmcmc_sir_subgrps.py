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

parameters0 = [ 9.4684914, 0.10717154, 4.35573724, 45.1529173, 3.78493284,
       12.39147448, 34.67666588, 0.62944, 1.61596184, 0.62277591]
sigma = np.array([[ 1.56140507e+01,  1.73377189e-01, -1.45541106e+01,
        -8.06075758e+00, -1.77848545e+00,  5.05147772e+00,
        -7.29374167e+00,  2.60021653e+00,  3.47646588e+00,
         3.85690347e-02],
       [ 1.73377189e-01,  1.04154430e+01, -2.85286703e+00,
        -1.41154869e+01,  8.57752185e-01,  6.36215893e-01,
        -4.56789170e+00, -1.95827674e+00,  1.37037140e+00,
        -1.26473773e-02],
       [-1.45541106e+01, -2.85286703e+00,  5.19064560e+01,
        -2.53220527e+01,  6.36433196e+00, -6.52494463e+00,
         3.96656278e+00, -5.33534689e+00, -7.35839027e+00,
         6.69970349e-02],
       [-8.06075758e+00, -1.41154869e+01, -2.53220527e+01,
         1.51964241e+02, -5.59015037e+00, -8.67078535e+00,
         3.46361243e+01,  3.66154298e+00, -1.21203792e+01,
         6.10935840e-02],
       [-1.77848545e+00,  8.57752185e-01,  6.36433196e+00,
        -5.59015037e+00,  5.61441125e+00, -4.51641519e+00,
         1.79528304e+00, -1.33873718e+00, -8.92510929e-01,
         1.71926406e-02],
       [ 5.05147772e+00,  6.36215893e-01, -6.52494463e+00,
        -8.67078535e+00, -4.51641519e+00,  1.76918128e+01,
        -1.40963686e+01,  1.66123287e-01,  2.32456578e+00,
        -3.20529042e-03],
       [-7.29374167e+00, -4.56789170e+00,  3.96656278e+00,
         3.46361243e+01,  1.79528304e+00, -1.40963686e+01,
         3.57382133e+01, -3.63365559e+00, -5.49919442e+00,
         5.70664054e-02],
       [ 2.60021653e+00, -1.95827674e+00, -5.33534689e+00,
         3.66154298e+00, -1.33873718e+00,  1.66123287e-01,
        -3.63365559e+00,  8.75804131e+00,  7.31773972e-01,
         2.84291479e-02],
       [ 3.47646588e+00,  1.37037140e+00, -7.35839027e+00,
        -1.21203792e+01, -8.92510929e-01,  2.32456578e+00,
        -5.49919442e+00,  7.31773972e-01,  7.52351766e+00,
         3.43482998e-02],
       [ 3.85690347e-02, -1.26473773e-02,  6.69970349e-02,
         6.10935840e-02,  1.71926406e-02, -3.20529042e-03,
         5.70664054e-02,  2.84291479e-02,  3.43482998e-02,
         3.93398031e-02]])
h = .1
n_population = np.sum(population, axis=1)
mu = population[:, 1].tolist()

thetas, likelihoods, sampled_trajs = particle_mcmc(
    data2,
    "sir_subgroups",
    parameters0,
    h,
    sigma=sigma,
    n_population=n_population,
    mu=mu,
    n_particles=100,
    n_chains = 20000,
    )

results_directory = "pmcmc_sbgrps/test1/"
graphs_directory = "pmcmc_sbgrps/test1/"

results_directory = "data/" + results_directory
graphs_directory = "graphs/" + graphs_directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# save results
np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_exposed.csv", sampled_trajs[:, :, 1], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 2], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 3], delimiter=",")

# # load results
# thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
# likelihoods = np.loadtxt(results_directory + "likelihoods.csv", delimiter=",")
# sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
# sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
# sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
# sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

# ## burn-in
# burn_in = 1000

# thetas2 = thetas[burn_in:, :]
# likelihoods2 = likelihoods[burn_in:]
# sampled_trajs2 = sampled_trajs[:, burn_in:, :]

# ## apply thinning
# thinning = 5

# thetas3 = thetas2[::thinning]
# likelihoods3 = likelihoods2[::thinning]
# sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

# ## calculate posterior variance
# thetas_unique = np.unique(thetas3, axis=0)
# sigma = np.cov(thetas_unique.T, ddof=0)

# ## plot thetas
# for i in range(thetas3.shape[1]-1):
#     plt.plot(
#             range(len(likelihoods3)),
#             thetas3[:, i],
#             color="blue",
#             linewidth=1,
#             label=f"beta{i}",
#         )
#     plt.legend()
#     plt.savefig(graphs_directory + f"beta{i}.png", bbox_inches='tight')
#     plt.show()

# plt.plot(
#     range(len(likelihoods3)),
#     thetas3[:, -1],
#     color="red",
#     linewidth=1,
#     label="pobs",
# )
# plt.legend()
# plt.savefig(graphs_directory + "gamma.png", bbox_inches='tight')
# plt.show()

# ## plot thetas - histogram
# for i in range(thetas3.shape[1]-1):
#     plt.hist(thetas3[:, i], bins=10, alpha=0.5, label=f"beta{i}")
#     plt.axvline(beta[i//3, i%3])
#     plt.legend()
#     plt.savefig(graphs_directory + f"beta{i}_hist.png", bbox_inches='tight')
#     plt.show()

# plt.hist(thetas3[:, -1], bins=10, alpha=0.5, label="gamma")
# plt.axvline(gamma)
# plt.legend()
# plt.savefig(graphs_directory + "gamma_hist.png", bbox_inches='tight')
# plt.show()


# ## plot likelihoods
# plt.plot(range(len(likelihoods3)), likelihoods3, color="blue", linewidth=1)
# plt.savefig(graphs_directory + "likelihoods.png", bbox_inches='tight')
# plt.show()

# ## plot trajectories
# lines1 = plt.plot(
#     range(len(data2)), sampled_trajs3[:, :, 0], "orange", linewidth=1
# )
# plt.plot(
#     range(len(data2)), sampled_trajs3[:, :, 1], "orange", linewidth=1
# )
# plt.plot(
#     range(len(data2)), sampled_trajs3[:, :, 2], "orange", linewidth=1
# )
# lines4 = plt.plot(range(len(data2)), dataset.iloc[:, 1:], "k", linewidth=2)
# lines5 = plt.plot(range(len(data2)), data2, "k--", linewidth=2)
# plt.legend(
#     lines1[:1] + lines4 + lines5,
#     ["particle trajectories", "hidden infected", "observed infected"],
# )
# plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
# plt.show()


# ## test acceptance rate
# acceptance_rate = 100*len(np.unique(thetas[:, 0]))/len(thetas[:, 0])
# acceptance_rate2 = 100*len(np.unique(thetas3[:, 0]))/len(thetas3[:, 0])