import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

sys.path.append('.')

from gillespie_algo import *
from pmcmc import *
from helpers import *

population = np.array([[2000, 30, 0], [3000, 40, 0]])
beta = np.array([[5, 2], [1, 3]])
gamma = .5
max_time = 14
t = np.linspace(0, max_time, num=200)

data = sir_subgroups_simulate_discrete(population, t, beta, gamma)
data2 = np.loadtxt('data\simulated_datasets\sir_subgrps.csv', delimiter=', ')

# parameters0 = [4, 1, 1, 4, 1]
# n_population = np.sum(population, axis=1)
# mu = population[:, 1].tolist()
# h = 1

# thetas, likelihoods, sampled_trajs = particle_mcmc(
#     data2,
#     ModelType.SIR_SUBGROUPS,
#     parameters0,
#     h,
#     sigma=None,
#     n_population=n_population,
#     mu=mu,
#     n_particles=100,
#     n_chains=1000,
#     )

# directory = "likelihood_map/sbgrps/run1/"
directory = "pmcmc_sbgrps/test_50/"

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
            range(len(thetas3)),
            thetas3[:, i],
            color="blue",
            linewidth=1,
            label=f"beta{i}",
        )
    plt.plot(
        range(len(thetas3)),
        [beta[i // len(population), i % len(population)]] * len(thetas3),
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
    range(len(thetas3)),
    thetas3[:, -1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.plot(
    range(len(thetas3)),
    [gamma] * len(thetas3),
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


## plot likelihood*prior
parameters0 = [5, 2, 1, 3, .5]
sigma = np.array([[ 9.92932933e-03, -1.55148010e-03, -2.88252820e-03,
         1.06775266e-03,  4.30805482e-05],
       [-1.55148010e-03,  7.58934319e-03,  1.18080491e-04,
        -1.97263662e-03,  1.37145500e-04],
       [-2.88252820e-03,  1.18080491e-04,  5.94141566e-03,
        -2.74992441e-04,  1.52805272e-05],
       [ 1.06775266e-03, -1.97263662e-03, -2.74992441e-04,
         3.16009591e-03,  3.39485677e-06],
       [ 4.30805482e-05,  1.37145500e-04,  1.52805272e-05,
         3.39485677e-06,  4.93981158e-05]])
h = 10
res = likelihoods3 * multivariate_normal.pdf(thetas3, np.array(parameters0), h*sigma)
plt.plot(range(len(res)), res, color="blue", linewidth=1)
plt.savefig(graphs_directory + "likelihoods2.png", bbox_inches='tight')
plt.show()


## plot trajectories
colors = ["tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
lines1 = plt.plot(
    range(len(data2)), sampled_trajs3[:, :, 0], "orange", linewidth=1
)
for i in range(1, sampled_trajs3.shape[2]):
    plt.plot(
        range(len(data2)), sampled_trajs3[:, :, i], "orange", linewidth=1
    )
for i in range(data2.shape[1]):
    color = colors[i]
    globals()[f"lines{i+2}"] = plt.plot(
        range(len(data2)), data.iloc[:-1, i], color, linewidth=2,
    )
    globals()[f"lines{data2.shape[1]+i+2}"] = plt.plot(
        range(len(data2)), data2[:, i], color, linestyle="--", linewidth=2,
    )
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7 + lines8 + lines9 +
    lines10 + lines11 + lines12 + lines13,
    ["particle trajectories", "hidden susceptible (group1)", "hidden infected (group1)",
    "hidden removed (group1)", "hidden susceptible (group2)", "hidden infected (group2)",
    "hidden removed (group2)", "observed susceptible (group1)", "observed infected (group1)",
    "observed removed (group1)", "observed susceptible (group2)", "observed infected (group2)",
    "observed removed (group2)"],
)
plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
plt.show()

## trajectories - ci
means = [
    np.mean(sampled_trajs3[:, :, i], axis=1) for i in range(sampled_trajs3.shape[2])
]
percentiles = [
    np.percentile(sampled_trajs3[:, :, i], [5, 95], axis=1) for i in range(sampled_trajs3.shape[2])
]
lines1 = plt.plot(
    range(len(data2)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data2)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(1, sampled_trajs3.shape[2]):
    plt.plot(
        range(len(data2)), means[i], "red", linewidth=1.5
    )
    plt.fill_between(range(len(data2)), percentiles[i][0], percentiles[i][1], color='orange', alpha=.8)
for i in range(data2.shape[1]):
    color = colors[i]
    globals()[f"lines{i+2}"] = plt.plot(
        range(len(data2)), data.iloc[:-1, i], color, linewidth=2
    )
    globals()[f"lines{data2.shape[1]+i+2}"] = plt.plot(
        range(len(data2)), data2[:, i], color, linestyle="--", linewidth=2
    )
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7 + lines8 + lines9 +
    lines10 + lines11 + lines12 + lines13,
    ["particle trajectories", "hidden susceptible (group1)", "hidden infected (group1)",
    "hidden removed (group1)", "hidden susceptible (group2)", "hidden infected (group2)",
    "hidden removed (group2)", "observed susceptible (group1)", "observed infected (group1)",
    "observed removed (group1)", "observed susceptible (group2)", "observed infected (group2)",
    "observed removed (group2)"],
)
plt.savefig(graphs_directory + "trajectories2.png", bbox_inches='tight')
plt.show()

## test acceptance rate
acceptance_rate = 100*len(np.unique(thetas[:, 0]))/len(thetas[:, 0])
acceptance_rate2 = 100*len(np.unique(thetas3[:, 0]))/len(thetas3[:, 0])

for i, val in enumerate(["beta0", "beta1", "beta2", "beta3", "alpha"]):
    ci_array = np.round(mean_credible_interval(thetas3[:, i]), 3)
    med = np.round(np.median(thetas3[:, i]), 3)
    print(f"{val}, mean: {ci_array[0]}, median: {med}, 95% CI: ({ci_array[1]}, {ci_array[2]})")

