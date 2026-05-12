import os
import sys
import time
import json
import numpy as np
import networkx as nx

from heapq import heappush, heappop
from itertools import count, batched, permutations
from copy import copy, deepcopy

from .solution import Solution
from .sorting import fast_non_dominated_sort
from .sorting import fast_non_dominated_sort_vectorized
from .sorting import crowding_distance_assignment

from ..progress_bar import ProgressBar
from ..graph import cypher

def _time(func):

    def inner(*args, **kwargs):

        t0 = time.time()

        out =  func(*args, **kwargs)

        print(f'{time.time() - t0:.4f}')

        return out

    return inner

def population_to_dict(population):
    '''
    Strip information from populations
    '''

    output = {}

    for idx, solution in enumerate(population):

        data = {
            'rank': deepcopy(solution.rank),
            'age': deepcopy(solution.age),
            'feasible': deepcopy(solution.feasible),
            'compliant': deepcopy(solution.compliant),
            'fitness': deepcopy(solution.fitness),
            'successors': deepcopy(solution.successors),
            'equipment': deepcopy(solution.equipment),
            'tours': deepcopy(solution.tours),
            'supply_events': deepcopy(solution.supply_events),
            'vehicles': {d: [vi.type for vi in v] for d, v in solution.vehicles.items()},
            'ports': {d: [vi.type for vi in v] for d, v in solution.ports.items()}
        }

        output[idx] = data

    return output

class Network():

    def __init__(self, **kwargs):

        # A graph defining the route start locations and depots of the network
        self.locations = kwargs.get('locations', nx.DiGraph())

        # A graph defining the trips and possible trip-trip sequences of the network
        self.trips = kwargs.get('trips', nx.DiGraph())

        # A set of Vehicle objects
        self.vehicle_types = kwargs.get('vehicle_types', {})

        # A set of Station objects
        self.port_types = kwargs.get('port_types', {})

        # Objective objects used for evaluation
        self.objectives = kwargs.get('objectives', {})

        # Constraint objects used for evaluation
        self.constraints = kwargs.get('constraints', {})

        self.depots = (
            [k for k, v in self.locations._node.items() if v['type'] == 'depot']
            )

    def statistics(self, population):

        output = {}

        for idx, solution in enumerate(population):

            output[idx] = solution.statistics()

        return output

    def data(self, population):

        output = {}

        for idx, solution in enumerate(population):

            output[idx] = solution.data()

        return output

    def to_json(self, data, path):

        with open(path, 'w') as file:

            json.dump(data, file, indent = 4)

    def optimize(self, rng = None, initial_population = None, **kwargs):

        # Seeded random number generation
        if rng is None:

            rng = np.random.default_rng()

        # optimization hyper-parameters
        max_iter = kwargs.get('max_iter', 100)
        min_iter = kwargs.get('min_iter', 0)
        initial_population_size = kwargs.get('initial_population_size', 300)
        population_size = kwargs.get('population_size', 100)
        max_workers = kwargs.get('max_workers', 1)
        crossover_probability = kwargs.get('crossover_probability', 0.5)
        mutation_probability = kwargs.get('mutation_probability', 0)
        survival_threshold = kwargs.get('survival_threshold', 0)
        corner_depth = kwargs.get('corner_depth', 0)
        retries = kwargs.get('retries', 0)

        parallel = max_workers > 1

        # Settings
        store = kwargs.get('store', True)

        # Arguments for ProgressBar
        progress_bar_kw = kwargs.get('progress_bar_kw', {})

        # Initial population
        if initial_population is None:
            #Randomly generate initial population

            population = self.initial_population(
                population_size = initial_population_size,
                rng = rng,
                max_workers = max_workers,
            )

        else:
            # Clone initial population

            population_size = len(initial_population)

            # Evaluation of initial population is done to reset some effects
            # of vehicle assignment
            initial_population = self.evaluate_population(initial_population)

            population = []

            for solution in initial_population:

                population.append(
                    Solution(
                        self,
                        successors = deepcopy(solution.successors),
                        equipment = deepcopy(solution.equipment),
                        )
                    )

        # Applying operators to initial population
        population = self.evaluate_population(population, max_workers = max_workers)

        candidates = [list(s.fitness.values()) for s in population]
        order = crowding_distance_assignment(candidates)

        population = [population[o] for o in order][:population_size]
        population = self.rank_population(population)

        generations = {0: self.statistics(population)}

        retries_used = 0

        print('')
        print('Optimizing')
        print('')

        seeds = rng.integers(0, 1000000000000, size = (max_iter))

        # Running NSGA
        for idx in ProgressBar(range(max_iter), **progress_bar_kw):

            rng = np.random.default_rng(seeds[idx])

            for solution in population:

                solution.age += 1

            # Build the next population
            offspring = self.crossover_mutation(
                population,
                crossover_probability = crossover_probability,
                mutation_probability = mutation_probability,
                rng = rng
                )

            offspring = self.evaluate_population(offspring, max_workers = max_workers)
            next_population = population + offspring
            next_population = self.rank_population(next_population)
            population = self.cull_population(
                next_population, population_size, corner_depth = corner_depth
                )

            if store:

                generations[idx + 1] = self.statistics(population)

            remaining_offspring = len([p for p in population if p.age == 0])

            if idx >= min_iter:

                if remaining_offspring <= survival_threshold:

                    retries_used += 1

                if retries_used > retries:

                    break

        return population, generations

    def initial_population(self, population_size = 100, rng = None, **kwargs):

        print('Generating Initial Population\n')

        if rng is None:

            rng = np.random.default_rng()

        population = []
        
        for i in ProgressBar(range(population_size)):

            population.append(Solution(self, **kwargs).generate(self, rng))

        return population

    def evaluate_population(self, population, **kwargs):

        max_workers = kwargs.get('max_workers', 10)

        for solution in population:

            solution.solve_and_evaluate(self)

        return population

    def rank_population(self, population, **kwargs):

        # Selecting only feasible solutions
        feasible_solutions = [s for s in population if s.feasible]
        viable_solutions = [s for s in feasible_solutions if s.compliant]
        non_viable_solutions = [s for s in feasible_solutions if not s.compliant]

        # Determining rank for viable solutions
        if len(viable_solutions) > 0:

            rank = fast_non_dominated_sort_vectorized(
                [list(s.fitness.values()) for s in viable_solutions]
                )

            for idx, solution in enumerate(viable_solutions):

                solution.rank = rank[idx]

            min_non_viable_rank = max(list(rank.values())) + 1

        else:

            min_non_viable_rank = 0

        # Determining rank for non-viable solutions (strictly higher than all
        # viable solutions).
        if len(non_viable_solutions) > 0:

            

            rank = fast_non_dominated_sort_vectorized(
                [list(s.fitness.values()) for s in non_viable_solutions]
                )

            for idx, solution in enumerate(non_viable_solutions):

                solution.rank = rank[idx] + min_non_viable_rank

        # print(len(viable_solutions), len(non_viable_solutions))

        population = viable_solutions + non_viable_solutions

        return population

    def cull_population(self, population, population_size, rng = None, **kwargs):

        corner_depth = kwargs.get('corner_depth', 0)

        if rng is None:

            rng = np.random.default_rng()

        # Determining candidate rank
        rank = {i: s.rank for i, s in enumerate(population)}

        candidates = [list(s.fitness.values()) for s in population]
        order = crowding_distance_assignment(candidates)

        # Keeping coner points
        if corner_depth > 0:

            candidates_array = np.array(candidates)
            indices = np.argsort(candidates_array, axis = 0)

            min_indices = indices[:corner_depth].flatten()
            max_indices = indices[-corner_depth:].flatten()

            corner_points = list(set([*min_indices, *max_indices]))

            for corner_point in corner_points:

                rank[corner_point] = 0

        # Building the mating pool
        fronts = np.unique(list(rank.values()))

        next_generation = []
        current_next_generation_size = 0

        # Adding the candidates of each rank in order
        for front in fronts:

            # Determining how many candidates need to be added to the mating pool
            remaining = population_size - current_next_generation_size

            if remaining <= 0:

                break

            # Finding the indices of the front
            indices_in_front = [i for i, k in rank.items() if k == front]
            front_size = len(indices_in_front)

            # If the number of needed solutions is greater than the number of candidates
            # in the current front then add all candidates in the current front to the
            # mating pool. If fewer are needed do a crowing distance sort and add the
            # most diverse solutions in the current front until the mating pool is filled
            if front_size <= remaining:

                next_generation.extend([population[i] for i in indices_in_front])

            else:

                order_in_front = [o for o in order if o in indices_in_front][:remaining]

                next_generation.extend([population[i] for i in order_in_front])

            current_next_generation_size = len(next_generation)

        if current_next_generation_size < population_size:

            remaining = population_size - current_next_generation_size

            next_generation.extend(
                [deepcopy(s) for s in rng.choice(next_generation, size = remaining)]
                )

        return next_generation

    def crossover_mutation(self, mating_pool, **kwargs):

        crossover_probability = kwargs.get('crossover_probability', 0.5)
        mutation_probability = kwargs.get('mutation_probability', 0.)
        rng = kwargs.get('rng', None)

        # Seeded random number generation
        if rng is None:

            rng = np.random.default_rng()

        # rng = np.random.default_rng(134)
        # Shuffling the mating pool
        indices = rng.choice(
            list(range(0, len(mating_pool))), size = len(mating_pool), replace = False
            )
        mating_pool = [mating_pool[i] for i in indices]

        # Building offspring
        offspring = []

        for parent_1, parent_2 in batched(mating_pool, 2):

            offspring.append(
                parent_1.mate_mutate(
                    self, parent_2,
                    crossover_probability = crossover_probability,
                    mutation_probability = mutation_probability,
                    rng = rng
                    )
                )

            offspring.append(
                parent_2.mate_mutate(
                    self, parent_1,
                    crossover_probability = crossover_probability,
                    mutation_probability = mutation_probability,
                    rng = rng
                    )
                )

        return offspring