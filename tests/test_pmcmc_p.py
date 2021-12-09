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
beta, gamma, probs = 2, 1, .1

dataset = sir_simulate_discrete(y0, t, beta, gamma)

# # observed process - HMM
# data1 = np.array([])
# prob_obs = 0.1
# for t in range(dataset.shape[0]):
#     y = np.random.binomial(dataset.iloc[t, 2], prob_obs)
#     data1 = np.append(data1, y)

# observed process - HMM + all
data2 = np.array([[0, 0, 0]])
for t in range(dataset.shape[0]):
    sus = np.random.binomial(dataset.iloc[t, 1], probs)
    inf = np.random.binomial(dataset.iloc[t, 2], probs)
    rec = np.random.binomial(dataset.iloc[t, 3], probs)
    data2 = np.append(data2, np.array([[sus, inf, rec]]), axis=0)
data2 = data2[1:]

# # observed infected
# data3 = np.array(np.round(dataset.iloc[:, 2]).astype(int))

# # observed infected and removed
# data4 = np.array(np.round(dataset.iloc[:, 1:-1]).astype(int))

theta_proposal = [1.933708  , 0.95707765, 0.09917913]
sigma = np.array([[ 8.56210710e-03,  4.96880438e-03, -2.94152350e-05],
       [ 4.96880438e-03,  3.20130528e-03, -1.73813239e-05],
       [-2.94152350e-05, -1.73813239e-05,  2.68921978e-06]])
h = 5
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
    data2,
    "sir",
    theta_proposal,
    h,
    sigma = sigma,
    n_chains=5000,
    observations = False,
    probs=None,
    n_particles=100,
    n_population=4820,
    mu=20,
    jobs=-1,
)
results_directory = "pmcmc_p/test2/"
graphs_directory = "PMCMC_2_1_.1/test2/"

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
# thinning = 20

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
# plt.plot(
#     range(len(likelihoods3)),
#     thetas3[:, 2],
#     color="red",
#     linewidth=1,
#     label="pobs",
# )
# plt.legend()
# plt.savefig(graphs_directory + "theta.png", bbox_inches='tight')
# plt.show()

# ## plot thetas - histogram
# plt.hist(thetas3[:, 0], bins=10, alpha=0.5, label="beta")
# plt.hist(thetas3[:, 1], bins=10, alpha=0.5, label="gamma")
# plt.hist(thetas3[:, 2], bins=10, alpha=0.5, label="probs")
# plt.axvline(beta)
# plt.axvline(gamma)
# plt.axvline(probs)
# plt.legend()
# plt.savefig(graphs_directory + "theta2.png", bbox_inches='tight')
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