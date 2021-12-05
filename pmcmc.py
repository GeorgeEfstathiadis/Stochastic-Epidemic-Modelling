import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import binom, norm, multivariate_normal
from tqdm import tqdm

from gillespie_algo import *


def differential_sir(n_sir, t, beta, gamma):
    """Differential sir formulas"""

    N = sum(n_sir)
    dS_dt = -beta * n_sir[0] * n_sir[1] / N
    dI_dt = ((beta * n_sir[0] / N) - gamma) * n_sir[1]
    dR_dt = gamma * n_sir[1]
    return dS_dt, dI_dt, dR_dt


def differential_seir(n_seir, t, beta, alpha, gamma):
    """Differential seir formulas"""

    N = sum(n_seir)
    dS_dt = -beta * n_seir[0] * n_seir[2] / N
    dE_dt = beta * n_seir[0] * n_seir[2] / N - alpha * n_seir[1]
    dI_dt = alpha * n_seir[1] - gamma * n_seir[2]
    dR_dt = gamma * n_seir[2]
    return dS_dt, dE_dt, dI_dt, dR_dt


def differential_sir_subroups(n_sir, t, beta, gamma):
    """Differential sir formulas"""
    
    subrgoups = len(beta)
    N = sum(n_sir)
    dS_dt = []
    dI_dt = []
    dR_dt = []
    for i in range(subrgoups):
        dS_dt.append(-n_sir[i * 3] * sum(beta[i] * n_sir[[1 + 3 * j for j in range(subrgoups)]]) / N)
        dI_dt.append(n_sir[i * 3] * sum(beta[i] * n_sir[[1 + 3 * j for j in range(subrgoups)]]) / N - gamma * n_sir[i * 3 + 1])
        dR_dt.append(gamma * n_sir[i * 3 + 1])
    results = [[dS_dt[i], dI_dt[i], dR_dt[i]] for i in range(subrgoups)]
    results = [item for sublist in results for item in sublist]
    return tuple(results)


def sir_simulate_discrete(y0, t, beta, gamma):
    solution = odeint(differential_sir, y0, t, args=(beta, gamma))
    data = [[row[i] for row in solution] for i in range(3)]
    data2 = pd.DataFrame(
        {
            "time": t,
            "susceptible": data[0],
            "infected": data[1],
            "removed": data[2],
        }
    )

    data3 = data2.copy()
    data3["time"] = np.ceil(data3["time"]).astype(int)
    times = data3["time"].copy()
    indexes = []
    for i in range(times.iloc[-1] + 1):
        indexes.append(len(times) - 1 - list(times)[::-1].index(i))
    data3 = data3.iloc[indexes].reset_index()
    return data3.iloc[:, 1:]


def seir_simulate_discrete(y0, t, beta, alpha, gamma):
    solution = odeint(differential_seir, y0, t, args=(beta, alpha, gamma))
    data = [[row[i] for row in solution] for i in range(4)]
    data2 = pd.DataFrame(
        {
            "time": t,
            "susceptible": data[0],
            "exposed": data[1],
            "infected": data[2],
            "removed": data[3],
        }
    )

    data3 = data2.copy()
    data3["time"] = np.ceil(data3["time"]).astype(int)
    times = data3["time"].copy()
    indexes = []
    for i in range(times.iloc[-1] + 1):
        indexes.append(len(times) - 1 - list(times)[::-1].index(i))
    data3 = data3.iloc[indexes].reset_index()
    return data3.iloc[:, 1:]


def sir_subgroups_simulate_discrete(y0, t, beta, gamma):
    y0 = [i for item in y0.tolist() for i in item]
    solution = odeint(differential_sir_subroups, y0, t, args=(beta.tolist(), gamma))
    data = pd.DataFrame(solution)
    columns = [[f"susceptible{grp}", f"infected{grp}", f"removed{grp}"] for grp in range(len(beta))]
    data.columns = [item for sublist in columns for item in sublist]
    data["time"] = t
    data3 = data.copy()
    data3["time"] = np.ceil(data3["time"]).astype(int)
    times = data3["time"].copy()
    indexes = []
    for i in range(times.iloc[-1] + 1):
        indexes.append(len(times) - 1 - list(times)[::-1].index(i))
    data3 = data3.iloc[indexes].reset_index()
    return data3.iloc[:, 1:]


def particle_filter(
    Y,
    theta_proposal,
    observations=False,
    probs=.1,
    n_particles=1000,
    n_population=4820,
    mu=20,
    jobs=4,
):
    zetas = np.zeros(len(Y))
    zetas_small = np.zeros((len(Y), n_particles, Y.shape[1]))
    weights = np.zeros((len(Y), n_particles))
    normalised_weights = np.zeros((len(Y), n_particles))
    hidden_process = np.zeros((len(Y), n_particles, Y.shape[1]))
    ancestry_matrix = np.zeros((len(Y), n_particles))

    zetas[0] = 1

    if Y.shape[1] == 3:
        zetas_small[0, :, 1] = np.random.poisson(mu, n_particles)
        zetas_small[0, :, 0] = n_population - zetas_small[0, :, 1]
        zetas_small[0, :, 2] = 0
    elif Y.shape[1] == 4:
        zetas_small[0, :, 2] = np.random.poisson(mu, n_particles)
        zetas_small[0, :, 0] = n_population - zetas_small[0, :, 2]
        zetas_small[0, :, 1] = 0
        zetas_small[0, :, 3] = 0
    if isinstance(mu, list):
        for i in range(len(mu)):
            zetas_small[0, :, i*3 + 1] = np.random.poisson(mu[i], n_particles)
            zetas_small[0, :, i*3 + 0] = n_population[i] - zetas_small[0, :, i*3 + 1]
            zetas_small[0, :, i*3 + 2] = 0
    hidden_process[0, :, :] = zetas_small[0, :, :]

    for p in range(1, len(Y)):
        if not observations:
            weights[p, :] = np.min(np.array([binom.pmf(Y[p - 1, i], zetas_small[p - 1, :, i], probs) for i in range(Y.shape[1])]), axis=0)
        else:
            weights[p, :] = np.min(np.array([binom.pmf(Y[p-1, i], zetas_small[p-1, :, i] / probs, probs) for i in range(Y.shape[1])]), axis=0)

        zetas[p] = zetas[p - 1] * np.mean(weights[p, :])

        normalised_weights[p, :] = np.array(weights[p, :]) / sum(weights[p, :])

        try:
            likely_particles = np.random.choice(
                range(n_particles), n_particles, p=normalised_weights[p, :]
            )
        except ValueError:
            return None, None, None
        ancestry_matrix[p, :] = likely_particles

        if isinstance(mu, int):
            model_state = [[hidden_process[p-1, j, i] for i in range(Y.shape[1])] for j in range(n_particles)]
        elif isinstance(mu, list):
            times = Y.shape[1] / len(mu)
            model_state = [np.array([[hidden_process[p-1, j, h*3 + i] for i in range(len(mu))] for h in range(int(times))]) for j in range(n_particles)]

        if jobs != 1:
            if Y.shape[1] == 3:
                simulated_pops = Parallel(n_jobs=jobs)(
                    delayed(sir_simulate)(
                        model_state[j],
                        theta_proposal,
                        1,
                        True,
                    )
                    for j in range(len(model_state))
                )
            elif Y.shape[1] == 4:
                simulated_pops = Parallel(n_jobs=jobs)(
                    delayed(seir_simulate)(
                        model_state[j],
                        theta_proposal,
                        1,
                        True,
                    )
                    for j in range(len(model_state))
                )
            elif isinstance(mu, list):
                simulated_pops = Parallel(n_jobs=jobs)(
                    delayed(sir_subgroups_simulate)(
                        model_state[j],
                        theta_proposal[0],
                        theta_proposal[1],
                        1,
                        True,
                    )
                    for j in range(len(model_state))
                )
        else:
            if Y.shape[1] == 3:
                simulated_pops = [
                    sir_simulate(
                        model_state[j],
                        theta_proposal,
                        1,
                        True,
                    )
                    for j in range(len(model_state))
                ]
            if Y.shape[1] == 4:
                simulated_pops = [
                    seir_simulate(
                        model_state[j],
                        theta_proposal[0],
                        theta_proposal[1],
                        1,
                        True,
                    )
                    for j in range(len(model_state))
                ]

        if isinstance(mu, int):
            hidden_process[p, :, :] = np.array(simulated_pops)[:, :hidden_process.shape[2]]
        elif isinstance(mu, list):
            hidden_process[p, :, :] = np.array([[item for sublist in x for item in sublist] for x in simulated_pops])[:, :hidden_process.shape[2]]

        zetas_small[p, :, :] = hidden_process[p, :, :]

    return zetas, hidden_process, ancestry_matrix


def particle_path_sampler(hidden_process, ancestry_matrix):
    n_particles = hidden_process.shape[1]
    time_steps = hidden_process.shape[0]

    trajectory = np.zeros((time_steps, hidden_process.shape[2]))
    chosen_path = np.random.randint(0, n_particles)
    trajectory[-1, :] = hidden_process[-1, chosen_path, :]

    for p in range(time_steps - 2, -1, -1):
        chosen_path = int(ancestry_matrix[p, chosen_path])
        trajectory[p, :] = hidden_process[p, chosen_path, :]

    return trajectory


def particle_mcmc(
    Y,
    parameters,
    h,
    sigma = None,
    n_chains=1000,
    observations=False,
    probs=.1,
    n_particles=1000,
    n_population=4820,
    mu=20,
    jobs=4,
):

    thetas = np.zeros((n_chains, len(parameters)))
    likelihoods = np.zeros(n_chains)
    sampled_trajs = np.zeros((Y.shape[0], n_chains, Y.shape[1]))
    std = sigma
    if sigma is None:
        std = np.eye(len(parameters))        
    while True:
        theta_proposal = abs(
            np.random.multivariate_normal(
                np.array(parameters), h*std
            )
        )

        probs2 = probs
        if probs is None:
            probs2 = min(theta_proposal[-1], 1)
            probs2 = max(probs2, 0)
            theta_proposal = theta_proposal[:-1]

        if isinstance(mu, list):
            gamma = theta_proposal[-1]
            beta = np.zeros((len(mu), len(mu)))
            for p in range(len(mu)**2):
                beta[p // len(mu), p % len(mu)] = theta_proposal[p]
            theta_proposal2 = (beta, gamma)
        else:
            theta_proposal2 = theta_proposal

        zetas, hidden_process, ancestry_matrix = particle_filter(
            Y,
            theta_proposal2,
            observations,
            probs2,
            n_particles,
            n_population,
            mu,
            jobs,
        )
        if zetas is not None:
            break
    trajectory = particle_path_sampler(hidden_process, ancestry_matrix)

    if probs is None:
        theta_proposal = np.append(theta_proposal, probs2)

    thetas[0, :] = theta_proposal
    likelihoods[0] = zetas[-1]
    sampled_trajs[:, 0, :] = trajectory

    outer_bar = tqdm(total=n_chains-1, desc='Chains', position=1)
    thetas_bar = tqdm(total=0, bar_format='{desc}', position=2)

    acceptances = 1

    for i in range(1, n_chains):
        theta_proposal = abs(
            np.random.multivariate_normal(
                thetas[i-1], h*std
            )
        )

        probs2 = probs
        if probs is None:
            probs2 = min(theta_proposal[-1], 1)
            probs2 = max(probs2, 0)
            theta_proposal = theta_proposal[:-1]

        if isinstance(mu, list):
            gamma = theta_proposal[-1]
            beta = np.zeros((len(mu), len(mu)))
            for p in range(len(mu)**2):
                beta[p // len(mu), p % len(mu)] = theta_proposal[p]
            theta_proposal2 = (beta, gamma)
        else:
            theta_proposal2 = theta_proposal

        zetas, hidden_process, ancestry_matrix = particle_filter(
            Y,
            theta_proposal2,
            observations,
            probs2,
            n_particles,
            n_population,
            mu,
            jobs,
        )
        if zetas is None:
            thetas[i, :] = thetas[i - 1, :]
            likelihoods[i] = likelihoods[i - 1]
            sampled_trajs[:, i, :] = sampled_trajs[:, i - 1, :]
            continue
        
        trajectory = particle_path_sampler(hidden_process, ancestry_matrix)

        if probs is None:
            theta_proposal = np.append(theta_proposal, probs2)

        if "e" in str(zetas[-1]):
            constant = int(str(zetas[-1]).split("e-")[-1])//2
        else:
            constant = 1
        prob = (
            1e1 ** constant
            * multivariate_normal.cdf(theta_proposal, np.array(parameters), h*std)
            * multivariate_normal.cdf(thetas[i - 1], theta_proposal, h*std)
            * zetas[-1]
        )
        prob /= (
            1e1 ** constant
            * multivariate_normal.cdf(np.array(parameters), theta_proposal, h*std)
            * multivariate_normal.cdf(theta_proposal, thetas[i-1], h*std)
            * likelihoods[i - 1]
        )

        prob = min(1, prob)

        if np.random.uniform() < prob:
            acceptances += 1
            thetas[i, :] = theta_proposal
            likelihoods[i] = zetas[-1]
            sampled_trajs[:, i, :] = trajectory
        else:
            thetas[i, :] = thetas[i - 1]
            likelihoods[i] = likelihoods[i - 1]
            sampled_trajs[:, i, :] = sampled_trajs[:, i - 1, :]
        
        outer_bar.update(1)
        thetas_bar.set_description_str(f'accepted_theta: {thetas[i]}, zeta: {zetas[-1]}, acceptance_ratio: {100*acceptances/(i+1)}%')

    return thetas, likelihoods, sampled_trajs
