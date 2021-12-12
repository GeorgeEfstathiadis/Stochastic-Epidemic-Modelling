"""Module for running simple ABC algorithm"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from gillespie_algo import sir_simulate


# Distance function - Absolute difference
def distance_function(I_1, I_2, R_1, R_2):
    """Distance function for ABC"""
    distance = (np.mean(abs(I_1 - I_2)) + np.mean(abs(R_1 - R_2))) / 2
    return distance


# The ABC method with uniform prior, returns samples from the posterior
def abc_algo(observed_data, no_of_samples, threshold, priors):
    """ABC algorithm function"""
    # The observed data
    # initialise Posterior array
    posterior_distr = {"beta": [], "gamma": []}
    # trials
    max_time = observed_data.shape[0]
    # loop through to get the samples.
    outer_bar = tqdm(total=no_of_samples, desc='NoSample', position=1)
    thetas_bar = tqdm(total=0, bar_format='{desc}', position=2)
    acceptances = 0
    trials = 0
    for _ in range(0, no_of_samples):
        distance = threshold + 1
        # While the distance is greater than the threshold
        # continue to sample beta, gamma
        while distance > threshold:
            trials += 1
            # sample beta, gamma from the prior
            beta = np.random.uniform(priors['beta'][0], priors['beta'][1])
            gamma = np.random.uniform(priors['gamma'][0], priors['gamma'][1])
            # generate the sim data
            n_start = observed_data[0, :].astype(int)
            sim_traj = sir_simulate(
                list(n_start),
                np.array([beta, gamma]),
                max_time,
                False,
            )

            sim_traj["i"] = np.array(sim_traj["i"])[
                np.array(sim_traj["time"]) <= max_time
            ]
            sim_traj["r"] = np.array(sim_traj["r"])[
                np.array(sim_traj["time"]) <= max_time
            ]

            if len(sim_traj["time"]) == len(sim_traj["i"]) + 1:
                sim_traj["time"] = sim_traj["time"][:-1]

            simulated_trajectories = np.transpose(
                np.array([
                    np.ceil(sim_traj["time"]),
                    sim_traj["s"],
                    sim_traj["i"],
                    sim_traj["r"],
                ]))
            
            simulated_trajectories2 = np.array([[0, 0, 0, 0]])
            nrows = len(simulated_trajectories)
            for j in range(nrows):
                if j == nrows - 1:
                    simulated_trajectories2 = np.append(
                        simulated_trajectories2,
                        [list(simulated_trajectories[j, :])],
                        axis = 0,
                    )
                elif simulated_trajectories[j, 0] not in np.unique(
                    simulated_trajectories[(j + 1): nrows, 0]
                ):
                    simulated_trajectories2 = np.append(
                        simulated_trajectories2,
                        [list(simulated_trajectories[j, :])],
                        axis = 0,
                    )
                
            simulated_trajectories2 = simulated_trajectories2[1:]
            for j in range(1, max_time):
                if j not in np.unique(simulated_trajectories2[:, 0]):
                    row_to_add = simulated_trajectories2[j - 1, :].copy()
                    row_to_add[0] = j
                    simulated_trajectories2 = np.insert(
                        simulated_trajectories2, j, [list(row_to_add)], axis = 0
                    )
            # calcalute the distance from Y
            simulated_trajectories2 = simulated_trajectories2[:observed_data.shape[0]]
            distance = distance_function(
                simulated_trajectories2[:, 2],
                observed_data[:, 1],
                simulated_trajectories2[:, 3],
                observed_data[:, 2],
            )
        posterior_distr["beta"].append(beta)
        posterior_distr["gamma"].append(gamma)
        if acceptances == 0:
            trajectories = simulated_trajectories2[None, :, :]
        else:
            trajectories = np.append(trajectories, simulated_trajectories2[None, :, :], axis = 0)
        acceptances += 1
        outer_bar.update(1)
        thetas_bar.set_description_str(f'accepted_theta: {[beta, gamma]}, distance: {distance}, acceptance_ratio: {100*acceptances/trials}%')
    return posterior_distr, trajectories
