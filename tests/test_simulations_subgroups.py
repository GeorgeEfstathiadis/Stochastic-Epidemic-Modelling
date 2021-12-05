import os
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import odeint
from scipy.stats import binom, norm
from tqdm import tqdm

sys.path.append('.')

from gillespie_algo import *
from pmcmc import *

conditions = sir_subgroups_simulate(
    np.array([[1000, 0, 0], [1500, 20, 0], [2000, 0, 0]]),
    np.array([[1, 10, 10], [10, 6, 4], [1, 4, 5]]),
    2, 14, False
    )

plt.plot(conditions['time'], conditions['s_0'], label='s_0')
plt.plot(conditions['time'], conditions['i_0'], label='i_0')
plt.plot(conditions['time'], conditions['r_0'], label='r_0')
plt.plot(conditions['time'], conditions['s_1'], label='s_1')
plt.plot(conditions['time'], conditions['i_1'], label='i_1')
plt.plot(conditions['time'], conditions['r_1'], label='r_1')
plt.plot(conditions['time'], conditions['s_2'], label='s_2')
plt.plot(conditions['time'], conditions['i_2'], label='i_2')
plt.plot(conditions['time'], conditions['r_2'], label='r_2')
plt.legend()
plt.show()

conditions2 = sir_subgroups_simulate(
    np.array([[1000, 15, 0], [1500, 20, 0], [2000, 25, 0]]),
    np.array([[1, 10, 10], [10, 6, 4], [1, 4, 5]]),
    2, 14, True
    )