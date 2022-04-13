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

data = sir_subgroups_simulate_discrete(population, t, beta, gamma)
data2 = np.loadtxt('data/simulated_datasets/sir_subgrps.csv', delimiter=', ')

# create data3 with columns being sum of subgroups
data3 = data2[:, :3] + data2[:, 3:6]

n_population = np.sum(population, axis=1)
mu = population[:, 1].tolist()

theta_proposal = [4, 1, 2, 4, 1]
sigma = np.eye(len(theta_proposal))
h = .5

# thetas, likelihoods, sampled_trajs = particle_mcmc(
#     data3,
#     ModelType.SIR_SUBGROUPS2,
#     theta_proposal,
#     h,
#     sigma=sigma,
#     n_chains=1000,
#     n_population=n_population,
#     mu=mu,
#     n_particles=100,
#     )

directory = "pmcmc_sbgrps2/test_10/"

results_directory = "data/" + directory
graphs_directory = "graphs/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# # save results
# np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
# np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_susceptible0.csv", sampled_trajs[:, :, 0], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_infected0.csv", sampled_trajs[:, :, 1], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_recovered0.csv", sampled_trajs[:, :, 2], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_susceptible1.csv", sampled_trajs[:, :, 3], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_infected1.csv", sampled_trajs[:, :, 4], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_recovered1.csv", sampled_trajs[:, :, 5], delimiter=",")

# load results
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
likelihoods = np.loadtxt(results_directory + "likelihoods.csv", delimiter=",")
sampled_trajs_susceptible0 = np.loadtxt(results_directory + "sampled_trajs_susceptible0.csv", delimiter=",")
sampled_trajs_infected0 = np.loadtxt(results_directory + "sampled_trajs_infected0.csv", delimiter=",")
sampled_trajs_recovered0 = np.loadtxt(results_directory + "sampled_trajs_recovered0.csv", delimiter=",")
sampled_trajs_susceptible1 = np.loadtxt(results_directory + "sampled_trajs_susceptible1.csv", delimiter=",")
sampled_trajs_infected1 = np.loadtxt(results_directory + "sampled_trajs_infected1.csv", delimiter=",")
sampled_trajs_recovered1 = np.loadtxt(results_directory + "sampled_trajs_recovered1.csv", delimiter=",")
sampled_trajs = np.stack((sampled_trajs_susceptible0, sampled_trajs_infected0, sampled_trajs_recovered0, sampled_trajs_susceptible1, sampled_trajs_infected1, sampled_trajs_recovered1), axis=-1)

## burn-in
burn_in = 0

thetas2 = thetas[burn_in:, :]
likelihoods2 = likelihoods[burn_in:]
sampled_trajs2 = sampled_trajs[:, burn_in:, :]

## apply thinning
thinning = 1

thetas3 = thetas2[::thinning]
likelihoods3 = likelihoods2[::thinning]
sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

## calculate posterior variance
thetas_unique = np.unique(thetas3, axis=0)
sigma = np.cov(thetas_unique.T, ddof=0)

## plot betas
for i in range(thetas3.shape[1]-1):
    plt.plot(
            range(len(likelihoods3)),
            thetas3[:, i],
            color="blue",
            linewidth=1,
            label=f"beta{i}",
        )
    plt.plot(
        range(len(likelihoods3)),
        [beta[i // len(population), i % len(population)]] * len(likelihoods3),
        color="red",
        linestyle="--",
        linewidth=.5,
        label=f"true beta{i}",
    )
    plt.legend()
    plt.savefig(graphs_directory + f"beta{i}.png", bbox_inches='tight')
    plt.show()

## plot gamma
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, -1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.plot(
    range(len(likelihoods3)),
    [gamma] * len(likelihoods3),
    color="red",
    linestyle="--",
    linewidth=.5,
    label="true gamma",
)
plt.legend()
plt.savefig(graphs_directory + "gamma.png", bbox_inches='tight')
plt.show()

## plot thetas - histogram
for i in range(thetas3.shape[1]-1):
    sns.kdeplot(thetas3[:, i], shade=True, label=f"beta{i}")
    plt.axvline(beta[i//len(population), i%len(population)])
    plt.legend()
    plt.savefig(graphs_directory + f"beta{i}_hist.png", bbox_inches='tight')
    plt.show()

sns.kdeplot(thetas3[:, -1], shade=True, label="gamma")
plt.axvline(gamma)
plt.legend()
plt.savefig(graphs_directory + "gamma_hist.png", bbox_inches='tight')
plt.show()


## plot likelihoods
plt.plot(range(len(likelihoods3)), likelihoods3, color="blue", linewidth=1)
plt.savefig(graphs_directory + "likelihoods.png", bbox_inches='tight')
plt.show()

## plot trajectories
lines1 = plt.plot(
    range(len(data2)), sampled_trajs3[:, :, 0], "orange", linewidth=1
)
for i in range(1, sampled_trajs3.shape[2]):
    plt.plot(
        range(len(data2)), sampled_trajs3[:, :, i], "orange", linewidth=1
    )

lines4 = plt.plot(range(len(data2)), data.iloc[:-1, :-1], "k", linewidth=2)
lines5 = plt.plot(range(len(data2)), data3, "k--", linewidth=2)
plt.legend(
    lines1[:1] + lines4 + lines5,
    ["particle trajectories", "hidden infected", "observed infected"],
)
plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
plt.show()

## trajectories - ci
means = [np.mean(sampled_trajs3[:, :, i], axis=1) for i in range(sampled_trajs3.shape[2])]
percentiles = [np.percentile(sampled_trajs3[:, :, i], [5, 95], axis=1) for i in range(sampled_trajs3.shape[2])]
lines1 = plt.plot(
    range(len(data2)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data2)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(1, sampled_trajs3.shape[2]):
    plt.plot(
        range(len(data2)), means[i], "red", linewidth=1.5
    )
    plt.fill_between(range(len(data2)), percentiles[i][0], percentiles[i][1], color='orange', alpha=.8)
lines4 = plt.plot(range(len(data2)), data.iloc[:-1, :-1], "k", linewidth=2)
lines5 = plt.plot(range(len(data2)), data3, "k", linestyle="--", linewidth=2)
plt.legend(
    lines1[:1] + lines4[:1] + lines5,
    ["particle trajectories", "hidden infected", "observed infected"],
)
plt.savefig(graphs_directory + "trajectories2.png", bbox_inches='tight')
plt.show()



## test acceptance rate
acceptance_rate = 100*len(np.unique(thetas[:, 0]))/len(thetas[:, 0])
acceptance_rate2 = 100*len(np.unique(thetas3[:, 0]))/len(thetas3[:, 0])