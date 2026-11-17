import os
import sys
import time
import numpy as np
import networkx as nx

from scipy.stats import multinomial

def sample_occupation(stationary, n, size = 10000, rng = None):

    if rng is None:

            rng = np.random.default_rng()

    samples = rng.multinomial(n, stationary, size = size)

    return samples