import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

sys.path.append('.')

from gillespie_algo import *
from pmcmc import *

population = np.array([[2000, 30, 0], [3000, 40, 0]])
beta = np.array([[5, 2], [1, 3]])
gamma = .5
max_time = 14
t = np.linspace(0, max_time, num=200)

data = sir_subgroups_simulate_discrete(population, t, beta, gamma)
data2 = np.loadtxt('data\simulated_datasets\sir_subgrps.csv', delimiter=', ')

directory = "likelihood_map/sbgrps/run1/"

results_directory = "data/" + directory
graphs_directory = "graphs/" + directory

if not os.path.exists(results_directory):
    os.makedirs(results_directory)
if not os.path.exists(graphs_directory):
    os.makedirs(graphs_directory)

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

# simulate likelihood boolean map
bools = likelihoods > np.mean(likelihoods)

thetas3 = thetas[bools, :]

# simulate fake prob
theta_proposal = [5, 2, 1, 3, .5]
std = np.array([[ 9.92932933e-03, -1.55148010e-03, -2.88252820e-03,
         1.06775266e-03,  4.30805482e-05],
       [-1.55148010e-03,  7.58934319e-03,  1.18080491e-04,
        -1.97263662e-03,  1.37145500e-04],
       [-2.88252820e-03,  1.18080491e-04,  5.94141566e-03,
        -2.74992441e-04,  1.52805272e-05],
       [ 1.06775266e-03, -1.97263662e-03, -2.74992441e-04,
         3.16009591e-03,  3.39485677e-06],
       [ 4.30805482e-05,  1.37145500e-04,  1.52805272e-05,
         3.39485677e-06,  4.93981158e-05]])
h = 1

thetas3 = np.zeros([*thetas.shape])
thetas3[0] = thetas[0]
acceptances = 1
for i in tqdm(range(1, len(thetas))):
    if "e" in str(likelihoods[i]):
        constant = int(str(likelihoods[i]).split("e-")[-1])//2
    else:
        constant = 1
    prob = (
        1e1 ** constant
        * multivariate_normal.cdf(thetas[i], theta_proposal, h*std)
        * multivariate_normal.cdf(thetas3[i - 1], thetas[i], h*std)
        * likelihoods[i]
    )
    prob /= (
        1e1 ** constant
        * multivariate_normal.cdf(theta_proposal, thetas[i], h*std)
        * multivariate_normal.cdf(thetas[i], thetas3[i-1], h*std)
        * likelihoods[i - 1]
    )

    prob = min(1, prob)

    if np.random.uniform() < prob:
        acceptances += 1
        thetas3[i] = thetas[i]
    else:
        thetas3[i] = thetas3[i - 1]
