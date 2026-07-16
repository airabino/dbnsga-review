import os
import sys
import time
import numpy as np
import networkx as nx

from itertools import count, pairwise
from collections import defaultdict, Counter
from heapq import heappush, heappop
from copy import copy, deepcopy
from scipy.stats import mode

def pick_one_from_each(data, rng):

    data = [d if len(d) > 0 else [None] for d in data]

    lengths = np.array([len(d) for d in data])
    ratios = 1 / lengths

    rn = rng.uniform(0, 1, size = len(data))

    indices = (rn // ratios).astype(int)

    return [d[indices[i]] for i, d in enumerate(data)]

class Solution():

    idx = 0

    def __init__(self, network, **kwargs):

        self.iidx = self.idx
        self.increment()

        self.compliant = True # Are all constraints met?

        self.vehicles = defaultdict(list)
        self.ports = defaultdict(list)

        self.successors = kwargs.get('successors', None)
        self.equipment = kwargs.get('equipment', None)
        self.depot = kwargs.get('depot', None)
        self.supply_type = kwargs.get('supply_type', None)

        self.fitness = {}

        self.rank = None

        self.age = 0

        self.depots = (
            [k for k, v in network.locations._node.items() if v['type'] == 'depot']
            )

        self.depot_probabilities = np.array(
            [network.locations[d].get('probability', 1.) for d in self.depots]
            )
        self.depot_probabilities /= self.depot_probabilities.sum()

        self.vehicle_types = list(network.vehicle_types.keys())

        self.vehicle_type_probabilities = np.array(
            [getattr(v, 'probability', 1.) for v in network.vehicle_types.values()]
            )
        self.vehicle_type_probabilities /= self.vehicle_type_probabilities.sum()

    @classmethod
    def increment(self):

        self.idx += 1

    def statistics(self):

        output = {
            'rank': deepcopy(self.rank),
            'age': deepcopy(self.age),
            'fitness': deepcopy(self.fitness),
            'compliant': deepcopy(self.compliant),
        }

        return output

    def data(self):

        output = {
            'successors': deepcopy(self.successors),
            'equipment': deepcopy(self.equipment),
            'depot': deepcopy(self.depot),
            'supply_type': deepcopy(self.supply_type),
        }

        return output

    def from_data(self, network, data):

        self.successors = data['successors']
        self.equipment = data['equipment']
        self.depot = data['depot']
        self.supply_type = data['supply_type']

        self.solve(network)
        self.evaluate(network)

        return self

    def mate_mutate(self, network, partner, **kwargs):

        child = self.mate(network, partner, **kwargs)
        child.mutate(network, **kwargs)

        return child

    def mate(self, network, partner, **kwargs):

        # Inputs
        crossover_probability = kwargs.get('crossover_probability', 0.5)
        rng = kwargs.get('rng', None)

        if rng is None:

            rng = np.random.default_rng()

        # Copying successors structure for child
        successors = {
            k: v for k, v in self.successors.items() if not v == 'depot'
        }
        predecessors = {v: k for k, v in successors.items()}
        
        # Random selection of partner genes to contribute to child
        partner_successors = {
            k: v for k, v in partner.successors.items() if not v == 'depot'
        }

        rn_crossover = rng.uniform(0, 1, size = len(partner_successors))

        contribution = {}
        inverse_contribution = {}

        for idx, (k, v) in enumerate(partner_successors.items()):

            # Crossover
            if rn_crossover[idx] <= crossover_probability:

                contribution[k] = v
                inverse_contribution[v] = k

        # Integrating the crossover
        for source, target in contribution.items():

            conflict_target = successors.pop(source, None)
            _ = predecessors.pop(conflict_target, None)

            conflict_source = predecessors.pop(target, None)
            _ = successors.pop(conflict_source, None)

            # Adding the contribution value
            successors[source] = target

            # Updating predecessors
            predecessors[target] = source

            if conflict_target in network.trips._adj.get(conflict_source, []):

                successors[conflict_source] = conflict_target
                predecessors[conflict_target] = conflict_source

        # Adding depots as successors where successor trips are not defined
        for idx, source in enumerate(network.trips.nodes()):
            if not source in successors:

                successors[source] = 'depot'

        # Copying equipment for child
        depot = {k: v for k, v in self.depot.items()}
        equipment = {k: v for k, v in self.equipment.items()}
        supply_type = {k: v for k, v in self.supply_type.items()}

        rn = rng.uniform(0, 1, size = len(self.equipment))

        for idx, (k, v) in enumerate(partner.equipment.items()):
            if rn[idx] <= crossover_probability:
                
                equipment[k] = v
                depot[k] = partner.depot[k]
                supply_type[k] = partner.supply_type[k]

        child = Solution(network)

        child.successors = successors
        child.equipment = equipment
        child.depot = depot
        child.supply_type = supply_type

        return child

    def mutate(self, network, **kwargs):

        # print('b', sum([v is None for v in self.successors.values()]))

        # Inputs
        mutation_probability = kwargs.get('mutation_probability', 0.)
        rng = kwargs.get('rng', None)

        if rng is None:

            rng = np.random.default_rng()

        predecessors = {v: k for k, v in self.successors.items()}

        rn_mutation = rng.uniform(0, 1, size = len(self.successors))

        selected_sources = [
            s for i, s in enumerate(network.trips.nodes()) \
            if rn_mutation[i] <= mutation_probability
            ]

        possible_targets = (
            [list(network.trips.successors(s)) for s in selected_sources]
            )
        selected_targets = pick_one_from_each(possible_targets, rng)
        
        mutated_genes = {
            s: selected_targets[i] for i, s in enumerate(selected_sources)
            }

        # Integrating the mutated genes
        for source, target in mutated_genes.items():

            if target is None:

                continue

            conflict_target = self.successors.pop(source, None)
            _ = predecessors.pop(conflict_target, None)

            conflict_source = predecessors.pop(target, None)
            _ = self.successors.pop(conflict_source, None)

            # Adding the contribution value
            self.successors[source] = target

            # Updating predecessors
            predecessors[target] = source

            if conflict_target in network.trips._adj.get(conflict_source, []):

                # print(conflict_target, conflict_source)

                self.successors[conflict_source] = conflict_target
                predecessors[conflict_target] = conflict_source

        # print('c', sum([v is None for v in self.successors.values()]))

        # Adding depots as successors where successor trips are not defined
        for idx, source in enumerate(network.trips.nodes()):
            if not source in self.successors:

                self.successors[source] = 'depot'

        # Mutating depot
        # selected_depot = rng.choice(
        #     rng.permutation(list(self.depot.values())), size = len(network.trips.nodes),
        #     )

        selected_depot = rng.choice(
            self.depots, size = len(network.trips.nodes),
            p = self.depot_probabilities
            )

        # Mutating equipment
        # selected = rng.choice(
        #     rng.permutation(list(self.equipment.values())), size = len(network.trips.nodes),
        #     )
        
        selected = rng.choice(
            self.vehicle_types, size = len(network.trips.nodes),
            p = self.vehicle_type_probabilities
            )

        # Mutating supply type
        supply_type_choices = (
            [network.vehicle_types[v].supply_types for v in selected]
            )

        supply_type_selections = pick_one_from_each(supply_type_choices, rng)

        rn = rng.uniform(0, 1, size = len(self.equipment))

        for idx, source in enumerate(self.equipment.keys()):

            if rn[idx] <= mutation_probability:

                self.equipment[source] = selected[idx]
                self.depot[source] = selected_depot[idx]
                self.supply_type[source] = supply_type_selections[idx]

    def generate_successors(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        sources = set()
        targets = set()
        successors = {}

        keys = list(network.trips.nodes)
        rng.shuffle(keys)

        for source in keys:

            _adj = network.trips._adj[source]

            if source in sources:

                continue

            potential_targets = [t for t in _adj.keys() if not t in targets] + ['depot']

            target = rng.choice(potential_targets)

            successors[source] = target

            sources.add(source)
            targets.add(target)

        return successors

    def generate_equipment(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        p = rng.uniform(0, 1, size = (len(network.vehicle_types)))
        p /= p.sum()


        vehicle_types = list(network.vehicle_types.keys())
        selected = rng.choice(vehicle_types, size = len(network.trips.nodes), p = p)

        equipment = {s: selected[i] for i, s in enumerate(network.trips.nodes)}

        return equipment

    def generate_depot(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        p = rng.uniform(0, 1, size = (len(self.depots)))
        p /= p.sum()

        selected = rng.choice(self.depots, size = len(network.trips.nodes), p = p)

        depot = {s: selected[i] for i, s in enumerate(network.trips.nodes)}

        return depot

    def generate_supply(self, network, equipment, rng = None):
        '''
        For each peice of equipment, pick one of the available supply types
        '''

        if rng is None:

            rng = np.random.default_rng()

        supply_type_choices = (
            [network.vehicle_types[v].supply_types for v in equipment.values()]
            )

        supply_type_selections = pick_one_from_each(supply_type_choices, rng)

        supply_type = {
            k: supply_type_selections[i] for i, k in enumerate(equipment.keys())
            }

        return supply_type

    def generate(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        self.successors = self.generate_successors(network, rng = rng)
        self.equipment = self.generate_equipment(network, rng = rng)
        self.depot = self.generate_depot(network, rng = rng)
        self.supply_type = self.generate_supply(network, self.equipment, rng = rng)

        return self

    def solve_and_evaluate(self, network, **kwargs):

        self.solve(network, **kwargs)
        self.evaluate(network)

    def evaluate(self, network):

        for k, objective in network.objectives.items():

            self.fitness[k] = objective.evaluate_solution(network, self)

        self.compliant = True

        for k, constraint in network.constraints.items():

            self.compliant = bool(
                self.compliant * constraint.evaluate_solution(network, self)
                )

        # print(len(self.tours))

    def solve(self, network, **kwargs):

        self.vehicles.clear()
        self.ports.clear()
        self.tours = None
        self.supply_events = None

        # Recovering the tours
        tours = self._tours()

        # a = np.unique(list(self.successors.keys()))
        # b = np.unique([t for tour in tours for t in tour['trips']])

        # c = np.setdiff1d(a, b)

        # print('a', sum([v is None for v in self.successors.values()]))

        # predecessors = defaultdict(list)

        # for k, v in self.successors.items():

        #     predecessors[v].append(k)

        # print([(ci, self.successors[ci]) for ci in c])

        tours = self._voting(tours)

        tours = self._tour_information(network, tours)

        tours = self._vehicle_assignment(network, tours)

        supply_events = self._supply_events(tours)

        supply_events = self._port_assignment(network, supply_events)

        self.tours = tours
        self.supply_events = supply_events

    def _port_assignment(self, network, supply_events, **kwargs):

        # Finding the earliest event time
        earliest_event_start = min([s['earliest_start'] for s in supply_events])

        # Creating heaps for buses at depots
        port_type_heaps = {
            d: {v: [] for v in network.port_types.keys()} for d in self.depots
        }

        # Loading events in to the heap by latest finish time
        events_heap = []
        c = count()

        completed_events = []

        for event in supply_events:

            heappush(events_heap, (event['latest_finish'], next(c), event))

        while events_heap:

            # Getting the next event
            latest_finish, _, event = heappop(events_heap)

            port_type = event['supply_type']

            port_type_heap = (
                port_type_heaps[event['depot']][port_type]
                )

            build_flag = True

            # Checking for available port
            if port_type_heap:

                next_available_time, _, next_available_port = heappop(port_type_heap)

                resupply_time = next_available_port.resupply_time(
                    event['vehicle'], event['energy']
                    )

                finish_time = next_available_time + resupply_time

                if finish_time <= latest_finish:

                    event['port'] = next_available_port
                    build_flag = False

                    event['start'] = max([next_available_time, event['earliest_start']])
                    event['finish'] = event['start'] + resupply_time

                else:

                    heappush(
                        port_type_heap, (next_available_time, next(c), next_available_port)
                        )

            if build_flag:

                # Create a new instance
                event['port'] = network.port_types[port_type]
                self.ports[event['depot']].append(event['port'])

                resupply_time = event['port'].resupply_time(
                    event['vehicle'], event['energy']
                    )

                finish_time = event['earliest_start'] + resupply_time

                event['start'] = event['earliest_start']
                event['finish'] = finish_time

            # Adding the port to the heap
            heappush(port_type_heap, (event['finish'], next(c), event['port']))

            completed_events.append(event)

        return completed_events

    def _supply_events(self, tours, **kwargs):

        vehicle_tours = defaultdict(list)

        for idx, tour in enumerate(tours):

            vehicle_tours[tour['vehicle_idx']].append(idx)

        # Finding the earliest tour start time
        schedule_start = min([t['start'] for t in tours])

        # Recovering supply events

        supply_events = []

        for vehicle, tour_indices in vehicle_tours.items():
            for tour_0_idx, tour_1_idx in pairwise(tour_indices):

                event = {
                    'vehicle': tours[tour_0_idx]['vehicle'],
                    'vehicle_idx': tours[tour_0_idx]['vehicle_idx'],
                    'supply_type': tours[tour_0_idx]['supply_type'],
                    'depot': tours[tour_0_idx]['depot'],
                    'earliest_start': tours[tour_0_idx]['finish'],
                    'latest_finish': tours[tour_1_idx]['start'],
                    'energy': tours[tour_0_idx]['energy'],
                }

                supply_events.append(event)

            event = {
                'vehicle': tours[tour_indices[-1]]['vehicle'],
                'vehicle_idx': tours[tour_indices[-1]]['vehicle_idx'],
                'supply_type': tours[tour_indices[-1]]['supply_type'],
                'depot': tours[tour_indices[-1]]['depot'],
                'earliest_start': tours[tour_indices[-1]]['finish'],
                'latest_finish': 24 * 3600 + schedule_start,
                'energy': tours[tour_indices[-1]]['energy'],
                }

            supply_events.append(event)

        return supply_events

    def _vehicle_assignment(self, network, tours, **kwargs):

        # Finding the earliest tour start time
        schedule_start = min([t['start'] for t in tours])

        # Creating heaps for buses at depots
        vehicle_type_heaps = {
            d: {v: [] for v in network.vehicle_types.keys()} for d in self.depots
        }

        # Loading tours in to the tour heap by start time
        tours_heap = []
        c = count()
        k = count()

        completed_tours = []

        for tour in tours:

            heappush(tours_heap, (tour['start'], next(c), tour))

        while tours_heap:

            # Getting the next tour to depart
            departure_time, _, tour = heappop(tours_heap)

            # Assigning a vehicle
            vehicle_type = tour['vehicle_type']
            tour_depot = tour['depot']

            vehicle_type_heap = vehicle_type_heaps[tour_depot][vehicle_type]

            # Checking for available vehicle
            if vehicle_type_heap:

                next_available_time = vehicle_type_heap[0][0]

                if next_available_time <= departure_time:

                    _, _, tour['vehicle_idx'], tour['vehicle'] = heappop(
                        vehicle_type_heap
                        )

                else:

                    # Create a new instance
                    tour['vehicle'] = network.vehicle_types[vehicle_type]
                    tour['vehicle_idx'] = next(k)
                    self.vehicles[tour_depot].append(tour['vehicle'])  

            else:

                # Create a new instance
                tour['vehicle'] = network.vehicle_types[vehicle_type]
                tour['vehicle_idx'] = next(k)
                self.vehicles[tour_depot].append(tour['vehicle'])

            # Tour energy
            tour['energy'] = tour['vehicle'].energy(tour)

            # Tour feasibility
            tour_feasible = np.all(
                [c.evaluate_tour(network, tour) \
                for c in network.constraints.values()]
                )

            if not tour_feasible:
                # Split the tour and add the halves to the heap

                split_idx = len(tour['trips']) // 2

                self.successors[tour['trips'][split_idx - 1]] = 'depot'

                new_tours = [
                    {
                        'trips': tour['trips'][:split_idx],
                        'vehicle_type': tour['vehicle_type'],
                        'supply_type': tour['supply_type'],
                    },
                    {
                        'trips': tour['trips'][split_idx:],
                        'vehicle_type': tour['vehicle_type'],
                        'supply_type': tour['supply_type'],
                    }
                ]

                new_tours = self._tour_information(network, new_tours)

                for new_tour in new_tours:

                    heappush(tours_heap, (new_tour['start'], next(c), new_tour))

                # Add vehicle back to heap
                heappush(
                    vehicle_type_heap,
                    (
                        departure_time, next(c), tour['vehicle_idx'], tour['vehicle']
                        )
                    )

                continue

            # Adding minimum resupply time
            supply_type = tour['supply_type']
            resupply_time = network.port_types[supply_type].resupply_time(
                tour['vehicle'], tour['energy']
                )

            return_time = tour['finish'] + resupply_time

            # Adding the vehicle to the heap
            heappush(
                vehicle_type_heap,
                (return_time, next(c), tour['vehicle_idx'], tour['vehicle'])
                )

            completed_tours.append(tour)

        return completed_tours

    def _voting(self, tours, **kwargs):

        for idx, tour in enumerate(tours):

            # Voting on depot
            trip_depots = [self.depot[t] for t in tour['trips']]
            selected_depot = Counter(trip_depots).most_common(1)[0][0]

            # Voting on vehicle type
            trip_vehicle_types = [self.equipment[t] for t in tour['trips']]
            selected_vehicle_type = Counter(trip_vehicle_types).most_common(1)[0][0]

            # Voting on supply type
            trip_supply_types = (
                [self.supply_type[t] for t in tour['trips'] \
                if self.equipment[t] == selected_vehicle_type]
                )
            selected_supply_type = Counter(trip_supply_types).most_common(1)[0][0]

            for trip in tour['trips']:

                self.depot[trip] = selected_depot
                self.equipment[trip] = selected_vehicle_type
                self.supply_type[trip] = selected_supply_type

            tour['depot'] = selected_depot
            tour['vehicle_type'] = selected_vehicle_type
            tour['supply_type'] = selected_supply_type

        return tours

    def _tours(self, **kwargs):

        predecessors = defaultdict(list)

        for k, v in self.successors.items():

            predecessors[v].append(k)

        tours = []

        heap = []
        c = count()

        heappush(heap, (0, next(c), 'depot', ['depot']))

        while heap:

            tour_length, _, source, tour_trips = heappop(heap)

            if not predecessors[source]:

                tours.append({'trips': tour_trips[:-1]})

            else:

                for target in predecessors[source]:

                    heappush(
                        heap, (tour_length + 1, next(c), target, [target] + tour_trips)
                        )

        return tours

    def _tour_information(self, network, tours, **kwargs):

        trips = network.trips
        locations = network.locations

        for tour in tours:

            tour_trips = tour['trips']
            depot = tour['depot']

            first_trip_start = trips._node[tour_trips[0]]['start']
            first_trip_location = trips._node[tour_trips[0]]['start_location']
            tour_start = (
                first_trip_start -
                locations._adj[depot][first_trip_location]['duration']
            )

            duration = 0
            distance = 0

            # First_depot_leg
            s_location = depot
            t_location = trips._node[tour_trips[0]]['start_location']

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            duration += trips._node[tour_trips[0]].get('duration', 0)
            distance += trips._node[tour_trips[0]].get('distance', 0)

            # Trips
            for s, t in pairwise(tour_trips):

                s_location = trips._node[s].get('finish_location', depot)
                t_location = trips._node[t].get('start_location', depot)

                duration += locations._adj[s_location][t_location].get('duration', 0)
                distance += locations._adj[s_location][t_location].get('distance', 0)

                duration += trips._node[t].get('duration', 0)
                distance += trips._node[t].get('distance', 0)

            # Second_depot_leg
            s_location = trips._node[tour_trips[-1]]['finish_location']
            t_location = depot

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            tour['start'] = tour_start
            tour['finish'] = tour_start + duration
            tour['distance'] = distance
            tour['duration'] = duration

        return tours