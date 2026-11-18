import os
import sys
import time
import json
import numpy as np
import networkx as nx

from heapq import heappush
from heapq import heappop
from itertools import count
from itertools import batched
from itertools import permutations
from copy import copy
from copy import deepcopy
from tqdm import tqdm
from shutil import get_terminal_size

from .solution import Solution
from .sorting import fast_non_dominated_sort
from .sorting import fast_non_dominated_sort_vectorized
from .sorting import crowding_distance_assignment
from .sorting import dense_rank_single_objective
from .sorting import truncate_by_rank

from ..graph import cypher

TQDM_KWARGS = {
    'file': sys.stdout,
    'ncols': get_terminal_size()[0],
}

class NpEncoder(json.JSONEncoder):
    '''
    Encoder to allow for numpy types to be converted to default types for
    JSON serialization. For use with json.dump(s)/load(s).
    '''
    def default(self, obj):

        if isinstance(obj, np.integer):

            return int(obj)

        if isinstance(obj, np.floating):

            return float(obj)

        if isinstance(obj, np.ndarray):

            return obj.tolist()

        if isinstance(obj, np.bool_):

            return int(obj)

        return super(NpEncoder, self).default(obj)

def _time(func):

    def inner(*args, **kwargs):

        t0 = time.time()

        out =  func(*args, **kwargs)

        print(f'{time.time() - t0:.4f}')

        return out

    return inner

class _tqdm(tqdm):

    def __init__(self, *args, **kwargs):

        self.color = '\033[1m\033[38;5;34m\033[48;5;0m'
        self.reset = '\r\033[0m\r'
        self.format = "{l_bar}{bar}{r_bar}"
        self.disable = kwargs.get('disable', False)

        kwargs["bar_format"] = f"{self.color}{self.format}{self.reset}"

        super().__init__(*args, **kwargs)

    def close(self):

        if self.disable:
            return
            
        super().close()

        self.fp.write(self.reset)
        self.fp.flush()

def population_to_dict(population):
    '''
    Strip information from populations
    '''

    output = {}

    for idx, solution in enumerate(population):

        data = {
            'rank': deepcopy(solution.rank),
            'age': deepcopy(solution.age),
            'compliant': deepcopy(solution.compliant),
            'fitness': deepcopy(solution.fitness),
            'successors': deepcopy(solution.successors),
            'depot': deepcopy(solution.depot),
            'equipment': deepcopy(solution.equipment),
            'supply_type': deepcopy(solution.supply_type),
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
        self.vehicle_type_names = list(self.vehicle_types.keys())
        self.vehicle_type_probabilities = (
            {k: v.probability for k, v in self.vehicle_types.items()}
            )
        self.trip_vehicle_types = [
             [(k, self.vehicle_type_probabilities[k]) \
             for k in t.get('vehicle_types', self.vehicle_type_names)] \
             for _, t in self.trips.nodes(data=True)
            ]

        # A set of Station objects
        self.supply_types = kwargs.get('supply_types', {})
        self.supply_type_names = list(self.supply_types.keys())
        self.supply_type_probabilities = (
            {k: v.probability for k, v in self.supply_types.items()}
            )
        self.trip_supply_types = [
             list(dict.fromkeys([(s, self.supply_type_probabilities[s]) \
             for k in t.get('vehicle_types', self.vehicle_type_names) \
             for s in self.vehicle_types[k].supply_types])) \
             for _, t in self.trips.nodes(data=True)
            ]

        # Depots of the network
        self.depots = (
            [k for k, v in self.locations._node.items() if v['type'] == 'depot']
            )
        self.depot_probabilities = (
            {k: self.locations._node[k].get('probability', 1.) for k in self.depots}
            )

        # Objective objects used for evaluation
        self.objectives = kwargs.get('objectives', {})

        if not self.objectives:

            raise ValueError(
                'Network requires at least one objective; got an empty '
                '`objectives` dict.'
                )

        # When there is exactly one objective, rank_population/cull_population
        # dispatch to a cheaper single-objective sort/select path (see sorting.py)
        # instead of the full NSGA-II dominance/crowding machinery.
        self.single_objective = len(self.objectives) == 1

        # Constraint objects used for evaluation
        self.constraints = kwargs.get('constraints', {})

        # When True, voting results are written to chromosome
        self.lamarckian = kwargs.get('lamarckian', False)

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

            json.dump(data, file, indent = 4, cls = NpEncoder)

    def cold_start(self, population_size=0, rng=None, disable_pb=False):
        '''
        Generate an initial population
        '''

        # Seeded random number generator
        if rng is None:

            rng = np.random.default_rng()

        population = []

        print('Generating Initial Population\n')
        
        for i in _tqdm(range(population_size), disable=disable_pb, **TQDM_KWARGS):

            population.append(Solution(self).generate(self, rng))

        return population

    def hot_start(self, data=[], disable_pb=False):
        '''
        Reconstruct an initial population from data. Data must be a list of
        dictionaries containing the following fields:
        - successors
        - vehicle_type
        - port_type
        - depot
        '''

        population = []

        print('Recovering Initial Population\n')

        for solution in _tqdm(data, disable=disable_pb, **TQDM_KWARGS):

            population.append(
                Solution(
                    self,
                    successors=deepcopy(solution.successors),
                    vehicle_type=deepcopy(solution.vehicle_type),
                    supply_type=deepcopy(solution.supply_type),
                    depot=deepcopy(solution.depot),
                    )
                )

        return population

    def optimize(self, population=[], rng=None, **kwargs):

        # Seeded random number generation
        if rng is None:

            rng = np.random.default_rng()

        # optimization hyper-parameters
        max_iter = kwargs.get('max_iter', 100)
        min_iter = kwargs.get('min_iter', 0)
        population_size = kwargs.get('population_size', 100)
        crossover_probability = kwargs.get('crossover_probability', 0.5)
        mutation_probability = kwargs.get('mutation_probability', 0)
        survival_threshold = kwargs.get('survival_threshold', 0)
        retries = kwargs.get('retries', 0)

        # Settings
        store = kwargs.get('store', True)
        store_interval = kwargs.get('store_interval', 1)

        # Arguments for ProgressBar
        disable_pb = kwargs.get('disable_pb', False)

        # Applying operators to initial population
        population = self.evaluate_population(population)

        if self.single_objective:

            # No trade-off front to diversity-trim across - rank then truncate
            # directly, reusing the same fast path cull_population dispatches to.
            population = self.rank_population(population)
            population = self.cull_population(population, population_size, rng = rng)

        else:

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
        for idx in _tqdm(range(max_iter), disable=disable_pb, **TQDM_KWARGS):

            t0 = time.time()

            rng = np.random.default_rng(seeds[idx])

            for solution in population:

                solution.age += 1

            # Build the next population
            offspring = self.cross_population(
                population, crossover_probability=crossover_probability, rng=rng
                )
            offspring = self.mutate_population(
                population, mutation_probability=mutation_probability, rng=rng
                )

            offspring = self.evaluate_population(offspring,)
            next_population = population + offspring
            next_population = self.rank_population(next_population)
            population = self.cull_population(next_population, population_size)

            if store:
                if (idx + 1) % store_interval == 0:

                    generations[idx + 1] = self.statistics(population)

            remaining_offspring = len([p for p in population if p.age == 0])

            if idx >= min_iter:

                if remaining_offspring <= survival_threshold:

                    retries_used += 1

                if retries_used > retries:

                    break

        return population, generations

    def evaluate_population(self, population, **kwargs):

        for solution in population:

            solution.solve(self)
            solution.evaluate(self)

        return population

    def rank_population(self, population, **kwargs):

        sort = dense_rank_single_objective if self.single_objective \
            else fast_non_dominated_sort_vectorized

        unique_penalties = np.unique([s.penalty for s in population])

        min_rank = 0

        for penalty in unique_penalties:

            solutions = [s for s in population if s.penalty == penalty]
            rank = sort([list(s.fitness.values()) for s in solutions])

            for idx, solution in enumerate(solutions):

                solution.rank = rank[idx] + min_rank

            min_rank = max(list(rank.values())) + 1

        return population

    def cull_population(self, population, population_size, rng = None, **kwargs):

        if rng is None:

            rng = np.random.default_rng()

        if self.single_objective:

            # No trade-off fronts to preserve diversity across, and ranks are
            # typically near-unique for a continuous objective - the front-by-
            # front loop below degrades toward O(n^2) in that case. A stable
            # sort on rank is equivalent (same fitness distribution kept) and
            # is O(n log n).
            return truncate_by_rank(population, population_size, rng = rng)

        # Determining candidate rank
        rank = {i: s.rank for i, s in enumerate(population)}

        candidates = [list(s.fitness.values()) for s in population]
        order = crowding_distance_assignment(candidates)

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

    def cross_population(self, population, crossover_probability=0.5, rng=None, **kwargs):

        # Seeded random number generation
        if rng is None:

            rng = np.random.default_rng()

        # Shuffling the mating pool
        indices = rng.choice(
            list(range(0, len(population))), size = len(population), replace = False
            )
        population = [population[i] for i in indices]

        # Building offspring
        offspring = []

        for parent_1, parent_2 in batched(population, 2):

            offspring.append(
                parent_1.cross(
                    self, parent_2, crossover_probability=crossover_probability, rng=rng
                    )
                )

            offspring.append(
                parent_2.cross(
                    self, parent_1, crossover_probability=crossover_probability, rng=rng
                    )
                )

        return offspring

    def mutate_population(self, population, mutation_probability=0., rng=None):

        # Seeded random number generation
        if rng is None:

            rng = np.random.default_rng()

        for solution in population:

            solution.mutate(self, mutation_probability=mutation_probability)

        return population