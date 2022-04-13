import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sys.path.append('.')

from gillespie_algo import sir_simulate
from pmcmc import *
from abc_algo import *
from helpers import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)

# # noisy processs
# data1 = np.array([[0, 0, 0]])
# ratio_noise = 0.1
# for t in range(dataset.shape[0]):
#     sus = dataset.iloc[t, 1] + np.random.normal(0, ratio_noise * dataset.iloc[t, 1])
#     inf = dataset.iloc[t, 2] + np.random.normal(0, ratio_noise * dataset.iloc[t, 2])
#     rem = dataset.iloc[t, 3] + np.random.normal(0, ratio_noise * dataset.iloc[t, 3])
#     data1 = np.append(data1, [[sus, inf, rem]], axis=0)
# data1 = data1[1:]

data1 = np.loadtxt("data/simulated_datasets/sir_noisy.csv", delimiter=", ")

# plt.plot(range(15), dataset.iloc[:, 1], label='Susceptible')
# plt.plot(range(15), dataset.iloc[:, 2], label='Infected')
# plt.plot(range(15), dataset.iloc[:, 3], label='Removed')
# plt.plot(range(15), data1[:, 0], label='Susceptible_noise')
# plt.plot(range(15), data1[:, 1], label='Infected_noise')
# plt.plot(range(15), data1[:, 2], label='Removed_noise')
# plt.legend()
# plt.show()

# thetas, sampled_trajs = abc_algo(data1, 1000, 150, {'beta': [0, 5], 'gamma': [0, 5]})
# thetas = np.array([[thetas['beta'][i], thetas['gamma'][i]] for i in range(len(thetas['beta']))])

results_directory = "rejection_sampling_results/run1/"
graphs_directory = "rejection_sampling_results/run1/"

results_directory = "data/" + results_directory
graphs_directory = "graphs/" + graphs_directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# # # save results
# np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")

# load results
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

## plot thetas
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 0],
    color="blue",
    linewidth=1,
    label="beta",
)
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma.png", bbox_inches='tight')
plt.show()

## plot beta
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 0],
    color="blue",
    linewidth=1,
    label="beta",
)
plt.plot(
    range(thetas.shape[0]),
    [beta] * thetas.shape[0],
    color="red",
    linestyle="--",
    linewidth=.5,
    label="true beta",
)
plt.legend()
plt.savefig(graphs_directory + "beta.png", bbox_inches='tight')
plt.show()

## plot gamma
plt.plot(
    range(thetas.shape[0]),
    thetas[:, 1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.plot(
    range(thetas.shape[0]),
    [gamma] * thetas.shape[0],
    color="red",
    linestyle="--",
    linewidth=.5,
    label="true gamma",
)
plt.legend()
plt.savefig(graphs_directory + "gamma.png", bbox_inches='tight')
plt.show()


## plot thetas - scatterplot
plt.scatter(thetas[:, 0], thetas[:, 1])
plt.scatter(beta, gamma, color="red")
plt.savefig(graphs_directory + "beta_gamma2.png", bbox_inches='tight')
plt.show()

## plot thetas - histogram
sns.kdeplot(thetas[:, 0], shade=True, label="beta")
sns.kdeplot(thetas[:, 1], shade=True, label="gamma")
plt.axvline(beta)
plt.axvline(gamma)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma3.png", bbox_inches='tight')
plt.show()


## plot trajectories
lines1 = plt.plot(
    range(len(data1)), sampled_trajs[:, :, 1].transpose(), "orange", linewidth=1
)
plt.plot(
    range(len(data1)), sampled_trajs[:, :, 2].transpose(), "orange", linewidth=1
)
plt.plot(
    range(len(data1)), sum(data1[0]) - sampled_trajs[:, :, 1].transpose() - sampled_trajs[:, :, 2].transpose(), "orange", linewidth=1
)
lines2 = plt.plot(range(len(data1)), dataset.iloc[:, 1], "b", linewidth=2)
lines3 = plt.plot(range(len(data1)), dataset.iloc[:, 2], "g", linewidth=2)
lines4 = plt.plot(range(len(data1)), dataset.iloc[:, 3], "c", linewidth=2)
lines5 = plt.plot(range(len(data1)), data1[:, 0], "b--", linewidth=2)
lines6 = plt.plot(range(len(data1)), data1[:, 1], "g--", linewidth=2)
lines7 = plt.plot(range(len(data1)), data1[:, 2], "c--", linewidth=2)
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
    ["particle trajectories", "hidden susceptible", "hidden infected",
    "hidden removed", "observed susceptible", "observed infected",
    "observed removed"],
)
plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
plt.show()

## plot trajectories - ci
means = [np.mean(sampled_trajs[:, :, i+1], axis=0) for i in range(2)]
means.append(np.mean(sum(data1[0]) - sampled_trajs[:, :, 1].transpose() - sampled_trajs[:, :, 2].transpose(), axis=1))
percentiles = [np.percentile(sampled_trajs[:, :, i+1], [5, 95], axis=0) for i in range(2)]
percentiles.append(np.percentile(sum(data1[0]) - sampled_trajs[:, :, 1].transpose() - sampled_trajs[:, :, 2].transpose(), [5, 95], axis=1))
lines1 = plt.plot(
    range(len(data1)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data1)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(1, 3):
    plt.plot(
        range(len(data1)), means[i], "red", linewidth=1.5
    )
    plt.fill_between(range(len(data1)), percentiles[i][0], percentiles[i][1], color='orange', alpha=.8)
lines2 = plt.plot(range(len(data1)), dataset.iloc[:, 1], "b", linewidth=2)
lines3 = plt.plot(range(len(data1)), dataset.iloc[:, 2], "g", linewidth=2)
lines4 = plt.plot(range(len(data1)), dataset.iloc[:, 3], "c", linewidth=2)
lines5 = plt.plot(range(len(data1)), data1[:, 0], "b--", linewidth=2)
lines6 = plt.plot(range(len(data1)), data1[:, 1], "g--", linewidth=2)
lines7 = plt.plot(range(len(data1)), data1[:, 2], "c--", linewidth=2)
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
    ["particle trajectories", "hidden susceptible", "hidden infected",
    "hidden removed", "observed susceptible", "observed infected",
    "observed removed"],
)
plt.savefig(graphs_directory + "trajectories2.png", bbox_inches='tight')
plt.show()


for i, val in enumerate(["beta", "gamma"]):
    ci_array = np.round(mean_credible_interval(thetas[:, i]), 3)
    med = np.round(np.median(thetas[:, i]), 3)
    print(f"{val}, mean: {ci_array[0]}, median: {med}, 95% CI: ({ci_array[1]}, {ci_array[2]})")
