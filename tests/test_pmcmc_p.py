import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymc3 as pm3
import arviz as az

sys.path.append('.')

from gillespie_algo import sir_simulate
from pmcmc import *
from helpers import *

t = np.linspace(0, 14, num=200)
y0 = (4800, 20, 0)
beta, gamma, pobs = 2, 1, .1
probs = pobs
dataset = sir_simulate_discrete(y0, t, beta, gamma)
# data2 = np.array([[0, 0, 0]])
# prob_obs = 0.1
# for t in range(dataset.shape[0]):
#     sus = np.random.binomial(dataset.iloc[t, 1], prob_obs)
#     inf = np.random.binomial(dataset.iloc[t, 2], prob_obs)
#     rec = np.random.binomial(dataset.iloc[t, 3], prob_obs)
#     data2 = np.append(data2, np.array([[sus, inf, rec]]), axis=0)
# data2 = data2[1:]
# np.savetxt("data/simulated_datasets/sir_underreported.csv", data2, delimiter=", ")
data2 = np.loadtxt("data/simulated_datasets/sir_underreported.csv", delimiter=", ")

results_directory = "pmcmc_p/test1/"
results_directory = "data/" + results_directory
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
thetas2 = thetas[100:, :]
thetas3 = thetas2[::20]
thetas_unique = np.unique(thetas3, axis=0)
theta_proposal = thetas[-1].tolist()
# sigma = np.cov(thetas_unique.T, ddof=0)
sigma = np.array([
    [8.56210710e-03, 4.96880438e-03, -2.94152350e-05],
    [4.96880438e-03, 3.20130528e-03, -1.73813239e-05],
    [-2.94152350e-05, -1.73813239e-05, 2.68921978e-06]
    ])
h = 5


# thetas, likelihoods, sampled_trajs = particle_mcmc(
#     data2,
#     "sir",
#     theta_proposal,
#     h,
#     sigma=sigma,
#     n_chains=5000,
#     observations=False,
#     probs=None,
#     n_particles=100,
#     n_population=4820,
#     mu=20,
#     jobs=-1,
# )

i=1
directory = f"pmcmc_p/run{i}/"

results_directory = "data/" + directory
graphs_directory = "graphs/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# # # save results
# np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
# np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
# np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")

# load results
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
likelihoods = np.loadtxt(results_directory + "likelihoods.csv", delimiter=",")
sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

## burn-in
burn_in = 100

thetas2 = thetas[burn_in:, :]
likelihoods2 = likelihoods[burn_in:]
sampled_trajs2 = sampled_trajs[:, burn_in:, :]

## apply thinning
thinning = 10

thetas3 = thetas2[::thinning]
likelihoods3 = likelihoods2[::thinning]
sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

## calculate posterior variance
thetas_unique = np.unique(thetas3, axis=0)
sigma = np.cov(thetas_unique.T, ddof=0)

## plot thetas
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, 0],
    color="blue",
    linewidth=1,
    label="beta",
)
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, 1],
    color="orange",
    linewidth=1,
    label="gamma",
)
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, 2],
    color="green",
    linewidth=1,
    label="p",
)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma.png", bbox_inches='tight')
plt.show()

## plot beta
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, 0],
    color="blue",
    linewidth=1,
    label="beta",
)
plt.plot(
    range(len(likelihoods3)),
    [beta] * len(likelihoods3),
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
    range(len(likelihoods3)),
    thetas3[:, 1],
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

## plot p
plt.plot(
    range(len(likelihoods3)),
    thetas3[:, 2],
    color="green",
    linewidth=1,
    label="p",
)
plt.plot(
    range(len(likelihoods3)),
    [probs] * len(likelihoods3),
    color="red",
    linestyle="--",
    linewidth=.5,
    label="true p",
)
plt.legend()
plt.savefig(graphs_directory + "probs.png", bbox_inches='tight')
plt.show()

## plot thetas - scatterplot
plt.scatter(thetas3[:, 0], thetas3[:, 1])
plt.scatter(beta, gamma, color="red")
plt.savefig(graphs_directory + "beta_gamma2.png", bbox_inches='tight')
plt.show()

## plot thetas - histogram
sns.kdeplot(thetas3[:, 0], shade=True, label="beta")
sns.kdeplot(thetas3[:, 1], shade=True, label="gamma")
sns.kdeplot(thetas3[:, 2], shade=True, label="p")
plt.axvline(beta)
plt.axvline(gamma)
plt.axvline(probs)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma3.png", bbox_inches='tight')
plt.show()

# histogram - betagamma
sns.kdeplot(thetas3[:, 0], shade=True, label="beta")
sns.kdeplot(thetas3[:, 1], shade=True, label="gamma")
plt.axvline(beta)
plt.axvline(gamma)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma_hist.png", bbox_inches='tight')
plt.show()

# histogram - probs
sns.kdeplot(thetas3[:, 2], shade=True, label="p")
plt.axvline(probs)
plt.legend()
plt.savefig(graphs_directory + "probs_hist.png", bbox_inches='tight')
plt.show()


## plot likelihoods
plt.plot(range(len(likelihoods3)), likelihoods3, color="blue", linewidth=1)
plt.savefig(graphs_directory + "likelihoods.png", bbox_inches='tight')
plt.show()

## plot likelihood*prior
res = likelihoods3 * multivariate_normal.cdf(thetas3, np.array(theta_proposal), h*sigma)
plt.plot(range(len(res)), res, color="blue", linewidth=1)
plt.savefig(graphs_directory + "likelihoods2.png", bbox_inches='tight')
plt.show()

## plot trajectories
lines1 = plt.plot(
    range(len(data2)), sampled_trajs3[:, :, 0], "orange", linewidth=1
)
for i in range(1, sampled_trajs3.shape[2]):
    plt.plot(
        range(len(data2)), sampled_trajs3[:, :, i], "orange", linewidth=1
    )

lines2 = plt.plot(range(len(data2)), dataset.iloc[:, 1], "b", linewidth=2)
lines3 = plt.plot(range(len(data2)), dataset.iloc[:, 2], "g", linewidth=2)
lines4 = plt.plot(range(len(data2)), dataset.iloc[:, 3], "c", linewidth=2)
lines5 = plt.plot(range(len(data2)), data2[:, 0], "b--", linewidth=2)
lines6 = plt.plot(range(len(data2)), data2[:, 1], "g--", linewidth=2)
lines7 = plt.plot(range(len(data2)), data2[:, 2], "c--", linewidth=2)
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
    ["particle trajectories", "hidden susceptible", "hidden infected",
    "hidden removed", "observed susceptible", "observed infected",
    "observed removed"],
)
plt.savefig(graphs_directory + "trajectories.png", bbox_inches='tight')
plt.show()

## plot trajectories - ci
means = [np.mean(sampled_trajs3[:, :, i], axis=1) for i in range(3)]
percentiles = [np.percentile(sampled_trajs3[:, :, i], [5, 95], axis=1) for i in range(3)]
lines1 = plt.plot(
    range(len(data2)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data2)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(1, 4):
    plt.plot(
        range(len(data2)), means[i], "red", linewidth=1.5
    )
    plt.fill_between(range(len(data2)), percentiles[i][0], percentiles[i][1], color='orange', alpha=.8)
lines2 = plt.plot(range(len(data2)), dataset.iloc[:, 1], "b", linewidth=2)
lines3 = plt.plot(range(len(data2)), dataset.iloc[:, 2], "g", linewidth=2)
lines4 = plt.plot(range(len(data2)), dataset.iloc[:, 3], "c", linewidth=2)
lines5 = plt.plot(range(len(data2)), data2[:, 0], "b--", linewidth=2)
lines6 = plt.plot(range(len(data2)), data2[:, 1], "g--", linewidth=2)
lines7 = plt.plot(range(len(data2)), data2[:, 2], "c--", linewidth=2)
plt.legend(
    lines1[:1] + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
    ["particle trajectories", "hidden susceptible", "hidden infected",
    "hidden removed", "observed susceptible", "observed infected",
    "observed removed"],
)
plt.savefig(graphs_directory + "trajectories2.png", bbox_inches='tight')
plt.show()


## test acceptance rate
acceptance_rate = 100*len(np.unique(thetas[:, 0]))/len(thetas[:, 0])
acceptance_rate2 = 100*len(np.unique(thetas3[:, 0]))/len(thetas3[:, 0])

print(acceptance_rate)
print(acceptance_rate2)

for i, val in enumerate(["beta", "gamma", "pobs"]):
    ci_array = np.round(mean_credible_interval(thetas3[:, i]), 3)
    med = np.round(np.median(thetas3[:, i]), 3)
    print(f"{val}, mean: {ci_array[0]}, median: {med}, 95% CI: ({ci_array[1]}, {ci_array[2]})")


# test convergence
thetas1 = np.loadtxt("data/pmcmc_p/run1/thetas.csv", delimiter=",")
thetas2 = np.loadtxt("data/pmcmc_p/run2/thetas.csv", delimiter=",")
thetas3 = np.loadtxt("data/pmcmc_p/run3/thetas.csv", delimiter=",")

grt = gelman_rubin_test((thetas1, thetas2, thetas3))
ess = az.ess(
    az.convert_to_dataset(
        thetas1[None, :, :],
    )
)
ess = np.array(ess["x"])
length_used = len(thetas1)

print(f"Gelman-Rubin: {grt}, ESS: {ess} from {length_used} samples")

# combine plots
thetas1 = thetas1[burn_in:, :][::thinning]
thetas2 = thetas2[burn_in:, :][::thinning]
thetas3 = thetas3[burn_in:, :][::thinning]


colors = ["blue", "orange", "green"]
for j, val in enumerate(["beta", "gamma", "pobs"]):
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(
            range(len(thetas3)),
            globals()[f"thetas{i+1}"][:, j],
            color=colors[j],
            linewidth=1,
            label=val,
        )
        plt.plot(
            range(len(thetas3)),
            [globals()[val]] * len(thetas3),
            color="red",
            linestyle="--",
            linewidth=.5,
            label=f"true {val}",
        )
        plt.legend()
    plt.savefig(f"{graphs_directory}{val}_3chains.png", bbox_inches='tight')
    plt.show()

for j, val in enumerate(["beta", "gamma", "pobs"]):
    for i in range(3):
        plt.subplot(3, 1, i+1)
        sns.kdeplot(globals()[f"thetas{i+1}"][:, j], shade=True, label=val)
        plt.axvline(globals()[val], color="red", linestyle="--")
        plt.legend()
    plt.savefig(f"{graphs_directory}hist_{val}_3chains.png", bbox_inches='tight')
    plt.show()
