"""Module for gillespie algorithm
for stochastic epimic models
"""
from math import log
from random import random
import numpy as np


def sir_simulate(
    population: list, theta_proposal: np.ndarray, max_time: float, last_values_only: bool
) -> dict:
    """Create generator class of stochastic model

    Args:
        population: list of susceptible, infected and
            recovered numbers
        theta_proposal: model parameters
        max_time: maximum time for generating trajectories
        last_values_only: if True, return only last values
    Returns:
        conditions: dictionary of trajectories
    """
    beta, gamma = theta_proposal.tolist()
    s0, i0, r0 = population

    # initial species counts and sojourn times
    conditions = {
        "s": [s0],
        "i": [i0],
        "r": [r0],
        "time": [0.0],
    }

    N = s0 + i0 + r0
    # propensity functions
    propensities = {
        0: lambda d: beta * d["s"][-1] * d["i"][-1] / N,
        1: lambda d: gamma * d["i"][-1],
    }

    # change in species for each propensity
    stoichiometry = {
        0: {"s": -1, "i": 1, "r": 0},
        1: {"s": 0, "i": -1, "r": 1},
    }

    while conditions["i"][-1] > 0:

        evaluated_reactions = np.array([propensities[0](conditions), propensities[1](conditions)])

        # # draw r1, r2 from uniform distribution
        # r1, r2 = np.random.uniform(0, 1, 2)
        
        # # evaluate reaction time
        # tau = np.log(1 / r1) / sum(evaluated_reactions)

        # # evaluate reaction
        # evaluated_reactions_cumulative = np.cumsum(evaluated_reactions)
        # j = np.where(evaluated_reactions_cumulative - r2 * sum(evaluated_reactions) > 0)[0][0]

        tau = np.random.exponential(1/sum(evaluated_reactions))
        j = np.random.choice(a=2, p=evaluated_reactions/sum(evaluated_reactions))

        if conditions["time"][-1] + tau > max_time:
            break

        conditions["time"].append(conditions["time"][-1] + tau)
        for pop in ["s", "i", "r"]:
            conditions[pop].append(conditions[pop][-1] + stoichiometry[j][pop])
    
    if last_values_only:
        return conditions["s"][-1], conditions["i"][-1], conditions["r"][-1]
    
    return conditions


def seir_simulate(
    population: list, theta_proposal: np.ndarray, max_time: float, last_values_only: bool
) -> dict:
    """Create generator class of stochastic model

    Args:
        population: list of susceptible, infected and
            recovered numbers
        theta_proposal: model parameters
        max_time: maximum time for generating trajectories
        last_values_only: if True, return only last values
    Returns:
        conditions: dictionary of trajectories
    """
    beta, alpha, gamma = theta_proposal.tolist()
    s0, e0, i0, r0 = population

    # initial species counts and sojourn times
    conditions = {
        "s": [s0],
        "e": [e0],
        "i": [i0],
        "r": [r0],
        "time": [0.0],
    }

    N = s0 + e0 + i0 + r0
    # propensity functions
    propensities = {
        0: lambda d: beta * d["s"][-1] * d["i"][-1] / N,
        1: lambda d: alpha * d["e"][-1],
        2: lambda d: gamma * d["i"][-1],
    }

    # change in species for each propensity
    stoichiometry = {
        0: {"s": -1, "e": 1, "i": 0, "r": 0},
        1: {"s": 0, "e": -1, "i": 1, "r": 0},
        2: {"s": 0, "e": 0, "i": -1, "r": 1},
    }

    while conditions["e"][-1] > 0 or conditions["i"][-1] > 0:

        evaluated_reactions = np.array([propensities[0](conditions), propensities[1](conditions), propensities[2](conditions)])

        # # draw r1, r2 from uniform distribution
        # r1, r2 = np.random.uniform(0, 1, 2)
        
        # # evaluate reaction time
        # tau = np.log(1 / r1) / sum(evaluated_reactions)

        # # evaluate reaction
        # evaluated_reactions_cumulative = np.cumsum(evaluated_reactions)
        # j = np.where(evaluated_reactions_cumulative - r2 * sum(evaluated_reactions) > 0)[0][0]

        tau = np.random.exponential(1/sum(evaluated_reactions))
        j = np.random.choice(a=3, p=evaluated_reactions/sum(evaluated_reactions))

        if conditions["time"][-1] + tau > max_time:
            break

        conditions["time"].append(conditions["time"][-1] + tau)
        for pop in ["s", "e", "i", "r"]:
            conditions[pop].append(conditions[pop][-1] + stoichiometry[j][pop])
    
    if last_values_only:
        return conditions["s"][-1], conditions["e"][-1], conditions["i"][-1], conditions["r"][-1]
    
    return conditions
