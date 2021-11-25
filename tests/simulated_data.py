"""Module to run simulations for real world data"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from abc_algo import abc_algo
from gillespie_algo import sir_simulate

# numerical solution using an ordinary differential equation solversir
t = np.linspace(0, 14, num=200)
y0 = (480, 20, 0)
beta, gamma = 2, 1


def differential_sir(n_sir, t, beta, gamma):
    """Differential sir formulas"""
    N = sum(n_sir)
    dS_dt = -beta * n_sir[0] * n_sir[1] / N
    dI_dt = ((beta * n_sir[0] / N) - gamma) * n_sir[1]
    dR_dt = gamma * n_sir[1]
    return dS_dt, dI_dt, dR_dt


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

priors = {
    "beta": [0, 4],
    "gamma": [0, 4],
}

data3 = data2.copy()
data3["time"] = np.ceil(data3["time"]).astype(int)
times = data3["time"].copy()
indexes = []
for i in range(times.iloc[-1] + 1):
    indexes.append(len(times) - 1 - list(times)[::-1].index(i))
data3 = data3.iloc[indexes].reset_index()

posterior_distribution = abc_algo(data3, 100, 10, priors)
with open("posterior_dist.json", "w") as fp:
    json.dump(posterior_distribution, fp, indent=4)

## Review Results

# plot posteriors
plt.hist(posterior_distribution["beta"], label="beta")
plt.hist(posterior_distribution["gamma"], label="gamma")
plt.legend()
plt.show()

# plot pairs of parameters
plt.scatter(posterior_distribution["beta"], posterior_distribution["gamma"])
plt.scatter(
    np.mean(posterior_distribution["beta"]),
    np.mean(posterior_distribution["gamma"]),
)
plt.xlabel("beta")
plt.ylabel("gamma")
plt.show()

# plot simulation vs data
generator = sir_simulate(
    [data3["susceptible"][0], data3["infected"][0], data3["removed"][0]],
    np.mean(posterior_distribution["beta"]),
    np.mean(posterior_distribution["gamma"]),
    data3.shape[0],
)

plt.figure(figsize=(10, 10), dpi=100)

axes_s = plt.subplot(311)
axes_s.set_ylabel("susceptible individuals")

axes_i = plt.subplot(312)
axes_i.set_ylabel("infected individuals")

axes_r = plt.subplot(313)
axes_r.set_ylabel("recovered individuals")
axes_r.set_xlabel("time (arbitrary units)")

# simulate and plot 30 trajectories
TRAJECTORIES = 0
for trajectory in generator.direct():
    axes_s.plot(
        trajectory["time"], trajectory["s"], color="orange", linewidth=1 / 2
    )
    axes_i.plot(
        trajectory["time"], trajectory["i"], color="orange", linewidth=1 / 2
    )
    axes_r.plot(
        trajectory["time"], trajectory["r"], color="orange", linewidth=1 / 2
    )
    TRAJECTORIES += 1
    if TRAJECTORIES == 30:
        break

# plot data
axes_s.plot(data2["time"], data2["susceptible"], color="black")
axes_i.plot(data2["time"], data2["infected"], color="black")
axes_r.plot(data2["time"], data2["removed"], color="black")

plt.show()
