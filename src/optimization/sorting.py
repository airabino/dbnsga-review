import random
import numpy as np

from itertools import count
from collections import deque

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

    # print(ordered_indices[len(candidates) - n:])

    # out = [candidates[i] for i in ordered_indices[len(candidates) - n:]]

    return ordered_indices[len(candidates) - n:]

