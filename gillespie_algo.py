"""Module for gillespie algorithm
for stochastic epimic models
"""
from math import log
from random import random
import numpy as np
from typing import Union, List, Dict, Tuple


def sir_simulate(
    population: List[int], theta_proposal: np.ndarray, max_time: float, last_values_only: bool
) -> Union[Dict[str, float], Tuple[float, float, float]]:
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
    population: List[int], theta_proposal: np.ndarray, max_time: float, last_values_only: bool
) -> Union[Dict[str, float], Tuple[float, float, float, float]]:
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

def sir_subgroups_simulate(
    population: np.ndarray, betas_proposal: np.ndarray, gamma_proposal: float, max_time: float,
    last_values_only: bool,
) -> Union[Dict[str, float], List[List[int]]]:
    """Create generator class of stochastic model with subgroups

    Args:
        population: matrix of susceptible, infected and
            recovered numbers per subgroup
        betas_proposal: infection rates
        gamma_proposal: recovery rate
        max_time: maximum time for generating trajectories
        last_values_only: if True, return only last values
    Returns:
        dictionary of trajectories
            or tuple of last values
    """
    subgroups = population.shape[0]

    # initial species counts and sojourn times
    conditions = {
        "time": [0.0],
    }
    
    for pop in range(subgroups):
        for i, compartment in enumerate(["s", "i", "r"]):
            conditions[f"{compartment}_{pop}"] = [population[pop, i]]

    N = [sum(population[pop, :]) for pop in range(subgroups)]
    # propensity functions and change in species for each propensity
    propensities = {}
    stoichiometry = {}
    for pop in range(subgroups):
        for pop2 in range(subgroups):
            propensities[f"s_{pop}_{pop2}"] = lambda d, pop=pop, pop2=pop2: betas_proposal[pop, pop2] * d[f"s_{pop2}"][-1] * d[f"i_{pop}"][-1] / sum(N)
            stoichiometry[f"s_{pop}_{pop2}"] = {f"s_{pop2}": -1, f"i_{pop2}": 1}
        propensities[f"i_{pop}"] = lambda d, pop=pop: gamma_proposal * d[f"i_{pop}"][-1]
        stoichiometry[f"i_{pop}"] = {f"i_{pop}": -1, f"r_{pop}": 1}

    for key in list(stoichiometry.keys()):
        for key2 in list(conditions.keys())[1:]:
            if key2 not in stoichiometry[key]:
                stoichiometry[key][key2] = 0
        
    infected_pop = sum([conditions[f"i_{grp}"][-1] for grp in range(subgroups)])
    while infected_pop > 0:
        evaluated_reactions = {}
        for key in propensities.keys():
            evaluated_reactions[key] = propensities[key](conditions)

        # # draw r1, r2 from uniform distribution
        # r1, r2 = np.random.uniform(0, 1, 2)
        
        # # evaluate reaction time
        # tau = np.log(1 / r1) / sum(evaluated_reactions)

        # # evaluate reaction
        # evaluated_reactions_cumulative = np.cumsum(evaluated_reactions)
        # j = np.where(evaluated_reactions_cumulative - r2 * sum(evaluated_reactions) > 0)[0][0]

        tau = np.random.exponential(1/sum(list(evaluated_reactions.values())))
        j = np.random.choice(
            a=len(list(evaluated_reactions.values())),
            p=list(evaluated_reactions.values())/sum(list(evaluated_reactions.values()))
        )
        j2 = list(evaluated_reactions.keys())[j]

        if conditions["time"][-1] + tau > max_time:
            break

        conditions["time"].append(conditions["time"][-1] + tau)
        for pop in list(conditions.keys())[1:]:
            conditions[pop].append(conditions[pop][-1] + stoichiometry[j2][pop])

        infected_pop = sum([conditions[f"i_{grp}"][-1] for grp in range(subgroups)])
    
    if last_values_only:
        conditions2 = []
        for grp in range(subgroups):
            compartments = []
            for comp in ["s", "i", "r"]:
                compartments.append(conditions[f"{comp}_{grp}"][-1])
            conditions2.append(compartments)
        return conditions2
    
    return conditions
