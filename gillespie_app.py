"""Streamlit app for gillespie algorithm"""
import streamlit as st
from matplotlib import pyplot
from numpy import linspace
from scipy.integrate import odeint

from gillespie_algo import sir_simulate


beta = st.sidebar.number_input("beta", 0.0, 100.0, 1.0, 0.1)
gamma = st.sidebar.number_input("gamma", 0.0, 100.0, 1.0, 0.1)
s = st.sidebar.slider("S", 100, 11079, 11068, 1)
i = st.sidebar.slider("I", 1, 100, 11, 1)
t_end = st.sidebar.number_input("t", 0, 100, 31, 1)


# instantiate the SSA container with model
epidemic_generator = sir_simulate([s, i, 0], beta, gamma, t_end)

# make a nice, big figure
fig = pyplot.figure(figsize=(10, 10), dpi=500)

# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(311)
axes_s.set_xlim(0, t_end)
axes_s.set_ylabel("susceptible individuals")

axes_i = pyplot.subplot(312)
axes_i.set_xlim(0, t_end)
axes_i.set_ylabel("infected individuals")

axes_r = pyplot.subplot(313)
axes_r.set_xlim(0, t_end)
axes_r.set_ylabel("recovered individuals")
axes_r.set_xlabel("time (arbitrary units)")

# simulate and plot 30 trajectories
TRAJECTORIES = 0
for trajectory in epidemic_generator.direct():
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

# numerical solution using an ordinary differential equation solversir
t = linspace(0, t_end, num=200)
y0 = (s, i, 0)


def differential_sir(n_sir, beta, gamma):
    """Differential sir formulas"""
    N = sum(n_sir)
    dS_dt = -beta * n_sir[0] * n_sir[1] / N
    dI_dt = ((beta * n_sir[0] / N) - gamma) * n_sir[1]
    dR_dt = gamma * n_sir[1]
    return dS_dt, dI_dt, dR_dt


solution = odeint(differential_sir, y0, t, args=(beta, gamma))
solution = [[row[i] for row in solution] for i in range(3)]

# plot numerical solution
axes_s.plot(t, solution[0], color="black")
axes_i.plot(t, solution[1], color="black")
axes_r.plot(t, solution[2], color="black")

st.pyplot(fig)
