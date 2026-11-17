import random
import numpy as np

from itertools import count
from collections import deque
from copy import deepcopy

def fast_non_dominated_sort_vectorized(candidates):
    #candidates - list of tuples

    n = len(candidates)
    m = len(candidates[0])

    scores = np.array(candidates).T

    at_least_as_good = np.ones((n, n), dtype = int)
    better_than = np.zeros((n, n), dtype = int)

    # dominance = np.ones((n, n), dtype = int)

    for score in scores:

        s_r, s_c = np.meshgrid(score, score, indexing = 'ij')

        at_least_as_good *= (s_r <= s_c)
        better_than = better_than | (s_r < s_c)

    dominance = at_least_as_good & better_than

    domination_counter = {i: k for i, k in enumerate(dominance.sum(axis = 0))}

    dominated = [[] for _ in range(n)]

    for i, j in np.argwhere(dominance == 1):

        dominated[i].append(j)

    del dominance

    rank = {k: 0 for k, v in domination_counter.items() if v == 0}
    front = list(rank.keys())

    idx = 0

    while front:

        next_front = []

        for i in front:
            for j in dominated[i]:

                domination_counter[j] -= 1

                if domination_counter[j] == 0:

                    rank[j] = idx + 1
                    next_front.append(j)


        idx += 1

        front = next_front

    return rank

def fast_non_dominated_sort(candidates):
    #candidates - list of tuples

    dims = len(candidates[0])
    
    dominated = [[] for i in range(len(candidates))]
    domination_counter = {i: 0 for i in range(len(candidates))}

    kk = 0

    for i, incumbent in enumerate(candidates):

        dominated[i] = set()

        for j, challenger in enumerate(candidates):

            kk += 1

            better = [incumbent[k] < challenger[k] for k in range(dims)]
            not_worse = [incumbent[k] <= challenger[k] for k in range(dims)]

            if np.any(better) and np.all(not_worse):

                dominated[i].add(j)
                domination_counter[j] += 1

    # print(domination_counter)
    # print(dominated)

    rank = {k: 0 for k, v in domination_counter.items() if v == 0}
    front = list(rank.keys())

    idx = 0

    while front:

        next_front = []

        for i in front:
            for j in dominated[i]:

                domination_counter[j] -= 1

                if domination_counter[j] == 0:

                    rank[j] = idx + 1
                    next_front.append(j)


        idx += 1

        front = next_front

    return rank

def crowding_distance_assignment(candidates):

    n = len(candidates)
    m = len(candidates[0])

    distances = {i: 0 for i in range(n)}

    for i in range(m):

        indices = np.argsort([c[i] for c in candidates])

        denominator = (candidates[indices[-1]][i] - candidates[indices[0]][i])

        if denominator == 0:

            continue

        distances[indices[0]] = np.inf
        distances[indices[-1]] = np.inf

        for j in range(1, n - 1):

            pred = indices[j - 1]
            curr = indices[j]
            succ = indices[j + 1]


            distances[curr] += (
                (candidates[succ][i] - candidates[pred][i]) /
                denominator
            ) ** 2


    return list(np.flip(np.argsort(list(distances.values()))))

def crowding_distance_generator(candidates):

    to_str = lambda c: ''.join(str(s) for s in c)

    reverse = {to_str(v): i for i, v in enumerate(candidates)}

    for _ in range(len(candidates)):

        indices = crowding_distance_assignment(candidates)

        best = indices[-1]
        val = to_str(candidates[best])
        
        candidates = candidates[:best] + candidates[best + 1:]

        yield reverse[val]

def n_highest_spread(candidates, n):

    ordered_indices = list(crowding_distance_generator(candidates))

    return ordered_indices[len(candidates) - n:]

def dense_rank_single_objective(candidates):
    '''
    Single-objective drop-in for fast_non_dominated_sort_vectorized.

    candidates - list of 1-length sequences (one fitness value each)

    At m = 1, Pareto dominance (at-least-as-good-everywhere, strictly-better-
    somewhere) collapses to a plain '<' comparison, so ranking is just the
    dense rank of each value in sorted order (ties share a rank). This
    produces identical rank values to fast_non_dominated_sort_vectorized at
    m = 1, in O(n log n) instead of O(n^2).
    '''

    values = np.array([c[0] for c in candidates])

    _, inverse = np.unique(values, return_inverse = True)

    return {i: int(r) for i, r in enumerate(inverse)}

def truncate_by_rank(population, population_size, rng = None):
    '''
    Single-objective drop-in for the rank/crowding-distance front-filling
    loop in Network.cull_population.

    Since ties in single-objective rank correspond to identical fitness
    values, any tie-break among them yields the same fitness distribution,
    so a stable sort on .rank followed by truncation is sufficient - no
    crowding distance or front-by-front bookkeeping needed.
    '''

    if rng is None:

        rng = np.random.default_rng()

    order = np.argsort([s.rank for s in population], kind = 'stable')

    next_generation = [population[i] for i in order[:population_size]]

    if len(next_generation) < population_size:

        remaining = population_size - len(next_generation)

        next_generation.extend(
            [deepcopy(s) for s in rng.choice(next_generation, size = remaining)]
            )

    return next_generation