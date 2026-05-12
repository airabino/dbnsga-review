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

class Solution():

    idx = 0

    def __init__(self, network, **kwargs):

        self.iidx = self.idx
        self.increment()

        # self.network = network

        self.feasible = True # Can the energy balance be solved?
        self.compliant = True # Are all constraints met?

        self.vehicles = defaultdict(list)
        self.ports = defaultdict(list)

        self.successors = kwargs.get('successors', None)
        self.equipment = kwargs.get('equipment', None)
        self.supply_type = kwargs.get('supply_type', None)

        self.fitness = {}

        self.rank = None

        self.age = 0

        self.depots = (
            [k for k, v in network.locations._node.items() if v['type'] == 'depot']
            ) #Unneccesary

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
            'supply_type': deepcopy(self.supply_type),
            'vehicles': {d: [vi.type for vi in v] for d, v in self.vehicles.items()},
            'ports': {d: [vi.type for vi in v] for d, v in self.ports.items()},
            'rank': deepcopy(self.rank),
            'age': deepcopy(self.age),
            'fitness': deepcopy(self.fitness),
            'compliant': deepcopy(self.compliant),
        }

        return output

    def from_data(self, network, data):

        self.successors = data['successors']
        self.equipment = data['equipment']
        self.supply_type = data['supply_type']

        self.solve(network)
        self.evaluate(network)

        return self

    def mate_mutate(self, network, partner, **kwargs):
        '''
        Contains the Mate and Mutate operators
        '''

        # Inputs
        crossover_probability = kwargs.get('crossover_probability', 0.5)
        mutation_probability = kwargs.get('mutation_probability', 0.)
        rng = kwargs.get('rng', None)

        if rng is None:

            rng = np.random.default_rng()

        # print('a', rng.bit_generator.state['state']['state'])

        # Copying successors structure for child
        successors = {
            k: v for k, v in self.successors.items() if not v in self.depots
        }
        predecessors = {v: k for k, v in successors.items()}

        # Random selection of partner genes to contribute to child
        rn_crossover = rng.uniform(0, 1, size = len(partner.successors))

        contribution = {}
        inverse_contribution = {}

        for idx, (k, v) in enumerate(partner.successors.items()):

            # Crossover
            if (rn_crossover[idx] <= crossover_probability) and (not v in self.depots):

                contribution[k] = v
                inverse_contribution[v] = k

        # Random mutation of the crossover genes
        rn_mutation = rng.uniform(0, 1, size = len(partner.successors))

        for idx, (k, v) in enumerate(partner.successors.items()):

            #Mutation
            if rn_mutation[idx] <= mutation_probability:

                potential_targets = list(
                    set(list(network.trips.successors(k)))
                )

                potential_targets.sort()

                if not potential_targets:

                    continue

                target = rng.choice(potential_targets)

                if k in contribution:

                    _ = inverse_contribution.pop(contribution[k])

                if target in inverse_contribution:

                    conflict_key = inverse_contribution[target]
                    _ = contribution.pop(conflict_key)

                contribution[k] = target
                inverse_contribution[target] = k

        # Integrating the crossover / mutation
        for source, target in contribution.items():

            # If the target is not in predecessors it means that source: target
            # can be added without needing to resolve a conflict. If target is
            # in predecessors a conflict needs to be resolved.
            if target in predecessors:

                if source in successors:

                    # Identifying the conflict key
                    conflict_key = predecessors[target]

                    # Identifying and popping the conflict value
                    conflict_value = successors.pop(source)

                    # Adding the contribution value
                    successors[source] = target
                    predecessors[target] = source

                    if conflict_value in network.trips._adj[conflict_key]:

                        # Swapping values if valid
                        successors[conflict_key] = conflict_value
                        predecessors[conflict_value] = conflict_key

                    else:

                        # Otherwise drop the conflicts
                        _ = successors.pop(conflict_key)
                        _ = predecessors.pop(conflict_value)

                else:

                    # Identifying the conflict key
                    conflict_key = predecessors[target]

                    _ = successors.pop(conflict_key)

                    # Adding the contribution value
                    successors[source] = target

                    # Updating predecessors
                    predecessors[target] = source

            else:

                # Adding the contribution value
                successors[source] = target

                # Updating predecessors
                predecessors[target] = source

        # Adding depots as successors where successor trips are not defined
        # rn_depots = rng.choice(self.depots, size = len(self.successors))
        for idx, source in enumerate(self.successors.keys()):
            if not source in successors:

                # successors[source] = rn_depots[idx]
                rn_depot = rng.choice(
                    network.trips._node[source]['depots'],
                    )
                successors[source] = rn_depot

        # Copying equipment for child
        rn = rng.uniform(0, 1, size = len(self.equipment))
        equipment = {k: v for k, v in self.equipment.items()}
        supply_type = {k: v for k, v in self.supply_type.items()}

        for idx, (k, v) in enumerate(partner.equipment.items()):
            if rn[idx] <= crossover_probability:

                equipment[k] = v
                supply_type[k] = partner.supply_type[k]

        # Mutating equipment
        rn = rng.uniform(0, 1, size = len(self.equipment))

        p = np.array(
            [getattr(v, 'probability', 1.) for v in network.vehicle_types.values()]
            )
        p /= p.sum()

        vehicle_types = list(network.vehicle_types.keys())
        selected = rng.choice(vehicle_types, size = len(network.trips.nodes), p = p)

        for idx, source in enumerate(equipment.keys()):

            if rn[idx] <= mutation_probability:

                equipment[source] = selected[idx]

                rn_supply_type = rng.choice(
                    network.vehicle_types[equipment[source]].supply_types
                )

                supply_type[source] = rn_supply_type

        child = Solution(network)

        child.successors = successors
        child.equipment = equipment
        child.supply_type = supply_type

        return child

    def mate_mutate1(self, network, partner, **kwargs):
        '''
        Contains the Mate and Mutate operators
        '''

        # Inputs
        crossover_probability = kwargs.get('crossover_probability', 0.5)
        mutation_probability = kwargs.get('mutation_probability', 0.)
        rng = kwargs.get('rng', None)

        if rng is None:

            rng = np.random.default_rng()

        # print('a', rng.bit_generator.state['state']['state'])

        # Copying successors structure for child
        successors = {
            k: v for k, v in self.successors.items()
        }
        predecessors = {v: k for k, v in successors.items() if not v in self.depots}

        # Random selection of partner genes to contribute to child
        rn_crossover = rng.uniform(0, 1, size = len(partner.successors))

        contribution = {}
        inverse_contribution = {}

        for idx, (k, v) in enumerate(partner.successors.items()):

            # Crossover
            if (rn_crossover[idx] <= crossover_probability):

                contribution[k] = v
                inverse_contribution[v] = k

        # Random mutation of the crossover genes
        rn_mutation = rng.uniform(0, 1, size = len(partner.successors))

        for idx, (k, v) in enumerate(partner.successors.items()):

            #Mutation
            if rn_mutation[idx] <= mutation_probability:

                potential_targets = list(
                    set(list(network.trips.successors(k)))
                )

                potential_targets.sort()

                if not potential_targets:

                    continue

                target = rng.choice(potential_targets)

                if k in contribution:

                    _ = inverse_contribution.pop(contribution[k], None)

                if target in inverse_contribution:

                    conflict_key = inverse_contribution[target]
                    _ = contribution.pop(conflict_key)

                contribution[k] = target
                inverse_contribution[target] = k

        # Integrating the crossover / mutation
        for source, target in contribution.items():

            if target in self.depots:

                # Adding the contribution value
                successors[source] = target

                continue

            # If the target is not in predecessors it means that source: target
            # can be added without needing to resolve a conflict. If target is
            # in predecessors a conflict needs to be resolved.
            if target in predecessors:

                if source in successors:

                    # Identifying the conflict key
                    conflict_key = predecessors[target]

                    # Identifying and popping the conflict value
                    conflict_value = successors.pop(source)

                    # Adding the contribution value
                    successors[source] = target
                    predecessors[target] = source

                    if conflict_value in network.trips._adj[conflict_key]:

                        # Swapping values if valid
                        successors[conflict_key] = conflict_value
                        predecessors[conflict_value] = conflict_key

                    else:

                        # Otherwise drop the conflicts
                        _ = successors.pop(conflict_key)
                        _ = predecessors.pop(conflict_value, None)

                else:

                    # Identifying the conflict key
                    conflict_key = predecessors[target]

                    _ = successors.pop(conflict_key)

                    # Adding the contribution value
                    successors[source] = target

                    # Updating predecessors
                    predecessors[target] = source

            else:

                # Adding the contribution value
                successors[source] = target

                # Updating predecessors
                predecessors[target] = source

        # Adding depots as successors where successor trips are not defined
        # rn_depots = rng.choice(self.depots, size = len(self.successors))
        for idx, source in enumerate(self.successors.keys()):
            if not source in successors:

                # successors[source] = rn_depots[idx]
                rn_depot = rng.choice(
                    network.trips._node[source]['depots'],
                    )
                successors[source] = rn_depot

        # Copying equipment for child
        rn = rng.uniform(0, 1, size = len(self.equipment))
        equipment = {k: v for k, v in self.equipment.items()}
        supply_type = {k: v for k, v in self.supply_type.items()}

        for idx, (k, v) in enumerate(partner.equipment.items()):
            if rn[idx] <= crossover_probability:

                equipment[k] = v
                supply_type[k] = partner.supply_type[k]

        # Mutating equipment
        rn = rng.uniform(0, 1, size = len(self.equipment))

        p = np.array(
            [getattr(v, 'probability', 1.) for v in network.vehicle_types.values()]
            )
        p /= p.sum()

        vehicle_types = list(network.vehicle_types.keys())
        selected = rng.choice(vehicle_types, size = len(network.trips.nodes), p = p)

        for idx, source in enumerate(equipment.keys()):

            if rn[idx] <= mutation_probability:

                equipment[source] = selected[idx]

                rn_supply_type = rng.choice(
                    network.vehicle_types[equipment[source]].supply_types
                )

                supply_type[source] = rn_supply_type

        child = Solution(network)

        child.successors = successors
        child.equipment = equipment
        child.supply_type = supply_type

        return child

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

            potential_targets = (
                # [t for t in _adj.keys() if not t in targets] + self.depots
                [t for t in _adj.keys() if not t in targets] +
                network.trips._node[source]['depots']
            )

            target = rng.choice(potential_targets)

            successors[source] = target

            sources.add(source)
            targets.add(target)

        return successors

    def generate_equipment(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        p = np.array(
            [getattr(v, 'probability', 1.) for v in network.vehicle_types.values()]
            )
        p /= p.sum()

        vehicle_types = list(network.vehicle_types.keys())
        selected = rng.choice(vehicle_types, size = len(network.trips.nodes), p = p)

        equipment = {s: selected[i] for i, s in enumerate(network.trips.nodes)}

        return equipment

    def generate_supply(self, network, equipment, rng = None):
        '''
        For each peice of equipment, pick one of the available supply types
        '''

        if rng is None:

            rng = np.random.default_rng()

        supply_type = {}

        for trip, vehicle_type in equipment.items():

            # print(vehicle_type, network.vehicle_types[vehicle_type].supply_types)

            supply_type[trip] = rng.choice(
                network.vehicle_types[vehicle_type].supply_types
                )

        return supply_type

    def generate(self, network, rng = None):

        if rng is None:

            rng = np.random.default_rng()

        self.successors = self.generate_successors(network, rng = rng)
        self.equipment = self.generate_equipment(network, rng = rng)
        self.supply_type = self.generate_supply(network, self.equipment, rng = rng)

        return self

    def solve_and_evaluate(self, network, **kwargs):

        self.solve(network, **kwargs)

        if self.feasible:

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

        tours = self._tour_information(network, tours)

        tours = self._tour_vehicle_type_assignment(tours)

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
                    'depot': tours[tour_0_idx]['trips'][-1],
                    'earliest_start': tours[tour_0_idx]['finish'],
                    'latest_finish': tours[tour_1_idx]['start'],
                    'energy': tours[tour_0_idx]['energy'],
                }

                supply_events.append(event)

            event = {
                'vehicle': tours[tour_indices[-1]]['vehicle'],
                'vehicle_idx': tours[tour_indices[-1]]['vehicle_idx'],
                'supply_type': tours[tour_indices[-1]]['supply_type'],
                'depot': tours[tour_indices[-1]]['trips'][-1],
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

                self.successors[tour['trips'][split_idx - 1]] = tour['trips'][-1]

                # print('s', tour['trips'][split_idx], split_idx)
                # print(tour['trips'])

                new_tours = [
                    {
                        'trips': tour['trips'][:split_idx] + [tour['trips'][-1]],
                        'vehicle_type': tour['vehicle_type'],
                        'supply_type': tour['supply_type'],
                    },
                    {
                        'trips': [tour['trips'][0]] + tour['trips'][split_idx:],
                        'vehicle_type': tour['vehicle_type'],
                        'supply_type': tour['supply_type'],
                    }
                ]

                # print('')
                # print(new_tours)

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

    def _tour_vehicle_type_assignment(self, tours, **kwargs):

        for idx, tour in enumerate(tours):

            trip_types = [self.equipment[t] for t in tour['trips'][1:-1]]

            selected = Counter(trip_types).most_common(1)[0][0]

            supply_types = (
                [self.supply_type[t] for t in tour['trips'][1:-1] \
                if self.equipment[t] == selected]
                )

            selected_supply = Counter(supply_types).most_common(1)[0][0]

            for trip in tour['trips'][1:-1]:

                self.equipment[trip] = selected
                self.supply_type[trip] = selected_supply

            tour['vehicle_type'] = selected
            tour['supply_type'] = selected_supply

        return tours

    def _tours(self, **kwargs):

        predecessors = defaultdict(list)

        for k, v in self.successors.items():

            predecessors[v].append(k)

        origins = self.depots

        tours = []

        heap = []
        c = count()

        for origin in origins:

            heappush(heap, (0, next(c), origin, [origin]))

        while heap:

            tour_length, _, source, tour_trips = heappop(heap)

            if not predecessors[source]:

                tours.append({'trips': [tour_trips[-1]] + tour_trips})

            else:

                for target in predecessors[source]:

                    heappush(
                        heap, (tour_length + 1, next(c), target, [target] + tour_trips)
                        )

        return tours

    def _tour_information_old(self, network, tours, **kwargs):

        trips = network.trips
        locations = network.locations

        for tour in tours:

            tour_trips = tour['trips']
            depot = tour['trips'][0]

            first_trip_start = trips._node[tour_trips[1]]['start']
            first_trip_location = trips._node[tour_trips[1]]['location']
            tour_start = (
                first_trip_start -
                locations._adj[depot][first_trip_location]['duration']
            )

            duration = 0
            distance = 0

            # First_depot_leg
            s_location = tour_trips[0]
            t_location = trips._node[tour_trips[1]]['location']

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            duration += trips._node[tour_trips[1]].get('duration', 0)
            distance += trips._node[tour_trips[1]].get('distance', 0)

            # Trips
            for s, t in pairwise(tour_trips[1:-1]):

                s_location = trips._node[s].get('location', depot)
                t_location = trips._node[t].get('location', depot)

                duration += locations._adj[s_location][t_location].get('duration', 0)
                distance += locations._adj[s_location][t_location].get('distance', 0)

                duration += trips._node[t].get('duration', 0)
                distance += trips._node[t].get('distance', 0)

            # Second_depot_leg
            s_location = trips._node[tour_trips[-2]]['location']
            t_location = tour_trips[-1]

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            tour['start'] = tour_start
            tour['finish'] = tour_start + duration
            tour['distance'] = distance
            tour['duration'] = duration
            tour['depot'] = tour['trips'][0]

        return tours

    def _tour_information(self, network, tours, **kwargs):

        trips = network.trips
        locations = network.locations

        for tour in tours:

            tour_trips = tour['trips']
            depot = tour['trips'][0]

            first_trip_start = trips._node[tour_trips[1]]['start']
            first_trip_location = trips._node[tour_trips[1]]['start_location']
            tour_start = (
                first_trip_start -
                locations._adj[depot][first_trip_location]['duration']
            )

            duration = 0
            distance = 0

            # First_depot_leg
            s_location = tour_trips[0]
            t_location = trips._node[tour_trips[1]]['start_location']

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            duration += trips._node[tour_trips[1]].get('duration', 0)
            distance += trips._node[tour_trips[1]].get('distance', 0)

            # Trips
            for s, t in pairwise(tour_trips[1:-1]):

                s_location = trips._node[s].get('finish_location', depot)
                t_location = trips._node[t].get('start_location', depot)

                duration += locations._adj[s_location][t_location].get('duration', 0)
                distance += locations._adj[s_location][t_location].get('distance', 0)

                duration += trips._node[t].get('duration', 0)
                distance += trips._node[t].get('distance', 0)

            # Second_depot_leg
            s_location = trips._node[tour_trips[-2]]['finish_location']
            t_location = tour_trips[-1]

            duration += locations._adj[s_location][t_location].get('duration', 0)
            distance += locations._adj[s_location][t_location].get('distance', 0)

            tour['start'] = tour_start
            tour['finish'] = tour_start + duration
            tour['distance'] = distance
            tour['duration'] = duration
            tour['depot'] = tour['trips'][0]

        return tours