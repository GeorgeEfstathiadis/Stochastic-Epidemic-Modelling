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

from gillespie_algo import sir_simulate
from pmcmc import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

data1 = np.loadtxt("data/simulated_datasets/sir_noise.csv", delimiter=", ")

theta_proposal = [3, 2]
sigma = np.eye(2)
h = .5
# results_directory = "pmcmc_adaptive/test1/"
# results_directory = "data/" + results_directory
# thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
# thetas2 = thetas[100:, :]
# thetas3 = thetas2[::20]
# thetas_unique = np.unique(thetas3, axis=0)
# theta_proposal = thetas[-1].tolist()
# sigma = np.cov(thetas_unique.T, ddof=0)
# h = 10


thetas, likelihoods, sampled_trajs = particle_mcmc(
    data1,
    "sir",
    theta_proposal,
    h,
    sigma = sigma,
    n_chains=5000,
    observations = True,
    probs=.5,
    n_particles=100,
    n_population=4820,
    mu=20,
    jobs=-1,
)
results_directory = "pmcmc_noisy/test0/"
graphs_directory = "pmcmc_noisy/test0/"

results_directory = "data/" + results_directory
graphs_directory = "graphs/" + graphs_directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# # save results
np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")

# # load results
# thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
# likelihoods = np.loadtxt(results_directory + "likelihoods.csv", delimiter=",")
# sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
# sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
# sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
# sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

# ## burn-in
# burn_in = 100

# thetas2 = thetas[burn_in:, :]
# likelihoods2 = likelihoods[burn_in:]
# sampled_trajs2 = sampled_trajs[:, burn_in:, :]

# ## apply thinning
# thinning = 10

# thetas3 = thetas2[::thinning]
# likelihoods3 = likelihoods2[::thinning]
# sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

# ## calculate posterior variance
# thetas_unique = np.unique(thetas3, axis=0)
# sigma = np.cov(thetas_unique.T, ddof=0)

# ## plot thetas
# plt.plot(
#     range(len(likelihoods3)),
#     thetas3[:, 0],
#     color="blue",
#     linewidth=1,
#     label="beta",
# )
# plt.plot(
#     range(len(likelihoods3)),
#     thetas3[:, 1],
#     color="orange",
#     linewidth=1,
#     label="gamma",
# )
# plt.legend()
# plt.savefig(graphs_directory + "beta_gamma.png", bbox_inches='tight')
# plt.show()

# ## plot thetas - scatterplot
# plt.scatter(thetas3[:, 0], thetas3[:, 1])
# plt.scatter(beta, gamma, color="red")
# plt.savefig(graphs_directory + "beta_gamma2.png", bbox_inches='tight')
# plt.show()

# ## plot thetas - histogram
# plt.hist(thetas3[:, 0], bins=10, alpha=0.5, label="beta")
# plt.hist(thetas3[:, 1], bins=10, alpha=0.5, label="gamma")
# plt.axvline(beta)
# plt.axvline(gamma)
# plt.legend()
# plt.savefig(graphs_directory + "beta_gamma3.png", bbox_inches='tight')
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