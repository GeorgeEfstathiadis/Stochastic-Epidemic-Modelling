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
beta, gamma = 2, 1

dataset = sir_simulate_discrete(y0, t, beta, gamma)
data2 = np.loadtxt("data/simulated_datasets/sir_underreported.csv", delimiter=", ")


# for i, prob in enumerate(probs_levels):
#     data2 = np.array([[0, 0, 0]])
#     for t in range(dataset.shape[0]):
#         sus = np.random.binomial(dataset.iloc[t, 1], prob)
#         inf = np.random.binomial(dataset.iloc[t, 2], prob)
#         rec = np.random.binomial(dataset.iloc[t, 3], prob)
#         data2 = np.append(data2, np.array([[sus, inf, rec]]), axis=0)
#     data2 = data2[1:].astype(int)
#     np.savetxt(f"data/simulated_datasets/sir_under_{prob}.csv", data2, delimiter=", ")


probs_levels = [0.005, 0.01, 0.025, 0.05, 0.075]
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

# for prob in probs_levels:
#     data2 = np.loadtxt(f"data/simulated_datasets/sir_under_{prob}.csv", delimiter=", ")
#     thetas, likelihoods, sampled_trajs = particle_mcmc(
#         data2,
#         ModelType.SIR,
#         theta_proposal,
#         h,
#         sigma=sigma,
#         n_chains=5000,
#         observations=False,
#         probs=prob,
#         n_particles=100,
#         n_population=4820,
#         mu=20,
#         jobs=-1,
#     )
#     directory = f"under_level/under_{prob}/"

#     results_directory = "data/" + directory

#     if not os.path.exists(results_directory):
#         os.makedirs(results_directory)

#     # # save results
#     np.savetxt(results_directory + "thetas.csv", thetas, delimiter=",")
#     np.savetxt(results_directory + "likelihoods.csv", likelihoods, delimiter=",")
#     np.savetxt(results_directory + "sampled_trajs_susceptible.csv", sampled_trajs[:, :, 0], delimiter=",")
#     np.savetxt(results_directory + "sampled_trajs_infected.csv", sampled_trajs[:, :, 1], delimiter=",")
#     np.savetxt(results_directory + "sampled_trajs_recovered.csv", sampled_trajs[:, :, 2], delimiter=",")


# visualise
probs_levels = [0.01, 0.025, 0.05, 0.075]
prob = probs_levels[0]
directory = f"under_level2/under_{prob}/"
data2 = np.loadtxt(f"data/simulated_datasets/sir_under_{prob}.csv", delimiter=", ")

results_directory = "data/" + directory
graphs_directory = "graphs/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

# load results
thetas = np.loadtxt(results_directory + "thetas.csv", delimiter=",")
likelihoods = np.loadtxt(results_directory + "likelihoods.csv", delimiter=",")
sampled_trajs_susceptible = np.loadtxt(results_directory + "sampled_trajs_susceptible.csv", delimiter=",")
sampled_trajs_infected = np.loadtxt(results_directory + "sampled_trajs_infected.csv", delimiter=",")
sampled_trajs_recovered = np.loadtxt(results_directory + "sampled_trajs_recovered.csv", delimiter=",")
sampled_trajs = np.stack((sampled_trajs_susceptible, sampled_trajs_infected, sampled_trajs_recovered), axis=-1)

## burn-in
burn_in = 1000

thetas2 = thetas[burn_in:, :]
likelihoods2 = likelihoods[burn_in:]
sampled_trajs2 = sampled_trajs[:, burn_in:, :]

## apply thinning
thinning = 1

thetas3 = thetas2[::thinning]
likelihoods3 = likelihoods2[::thinning]
sampled_trajs3 = sampled_trajs2[:, ::thinning, :]

for i, val in enumerate(["beta", "gamma"]):
    ci_array = np.round(mean_credible_interval(thetas3[:, i]), 3)
    med = np.round(np.median(thetas3[:, i]), 3)
    pmse = round(posterior_mse(globals()[val], thetas3[:, i]), 6)
    print(f"{val}, mean: {ci_array[0]}, median: {med}, 95% CI: ({ci_array[1]}, {ci_array[2]}), PMSE: {pmse}")


## calculate posterior variance
thetas_unique = np.unique(thetas3, axis=0)
sigma = np.cov(thetas_unique.T, ddof=0)

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

## plot thetas - histogram
sns.kdeplot(thetas3[:, 0], shade=True, label="beta")
sns.kdeplot(thetas3[:, 1], shade=True, label="gamma")
plt.axvline(beta)
plt.axvline(gamma)
plt.legend()
plt.savefig(graphs_directory + "beta_gamma3.png", bbox_inches='tight')
plt.show()

## plot likelihood*prior
res = likelihoods3 * multivariate_normal.cdf(thetas3, np.array(theta_proposal), h*sigma)
plt.plot(range(len(res)), res, color="blue", linewidth=1)
plt.savefig(graphs_directory + "likelihoods2.png", bbox_inches='tight')
plt.show()

## plot trajectories - ci
means = [np.mean(sampled_trajs3[:, :, i], axis=1) for i in range(3)]
percentiles = [np.percentile(sampled_trajs3[:, :, i], [5, 95], axis=1) for i in range(3)]
lines1 = plt.plot(
    range(len(data2)), means[0], "red", linewidth=1.5
)
plt.fill_between(range(len(data2)), percentiles[0][0], percentiles[0][1], color='orange', alpha=.8)
for i in range(3):
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
acceptance_rate = 100*len(np.unique(thetas2[:, 0]))/len(thetas2[:, 0])
acceptance_rate2 = 100*len(np.unique(thetas3[:, 0]))/len(thetas3[:, 0])

print(acceptance_rate)
print(acceptance_rate2)

