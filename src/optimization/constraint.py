import numpy as np

from collections import defaultdict

class Constraint():
    '''
    An objective is used to evaluate a solution
    '''

    def __init__(self, **kwargs):

        self.rigid = kwargs.get('rigid', True)

    def evaluate_tour(self, network, tour):

        return True

    def evaluate_solution(self, network, solution):

        return True

class Routes_Filled_Portion(Constraint):

    def __init__(self, **kwargs):

        self.routes = kwargs.get('routes', [])
        self.included = kwargs.get('included', [])
        self.portion = kwargs.get('portion', 0)

    def evaluate_solution(self, network, solution):

        unfilled = defaultdict(int)
        total = defaultdict(int)

        for tour in solution.tours:
            for trip in tour['trips'][1:-1]:

                route = network.trips._node[trip]['route']

                # print(route)

                if route in self.routes or (self.routes == []):

                    total[route] += 1

                    if tour['vehicle_type'] in self.included:

                        unfilled[route] += 1
        
        # portions = {k: 1 - unfilled.get(k, 0) / v for k, v in total.items()}
        threshold = 1 - self.portion
        portions = [unfilled.get(k, 0) / v < threshold for k, v in total.items()]

        compliant = np.all(portions)
        
        return compliant

    def describe(self, network, solution):

        unfilled = defaultdict(int)
        total = defaultdict(int)

        for tour in solution.tours:
            for trip in tour['trips'][1:-1]:

                route = network.trips._node[trip]['route']

                if route in self.routes or (self.routes == []):

                    total[route] += 1

                    if tour['vehicle_type'] in self.included:

                        unfilled[route] += 1
        
        portion = 1 - sum(list(unfilled.values())) / sum(list(total.values()))
        
        return portion


class Fleet_Portion(Constraint):

    def __init__(self, **kwargs):

        self.included = kwargs.get('included', [])
        self.portion = kwargs.get('portion', [])

    def evaluate_solution(self, network, solution):

        n_included = 0
        n_total = 0

        for depot, vehicles in solution.vehicles.items():

            for vehicle in vehicles:

                n_total += 1

                if vehicle.type in self.included:

                    n_included += 1

        compliant = n_included / n_total >= self.portion

        return compliant

class Port_Limit(Constraint):

    def __init__(self, **kwargs):

        self.sizes = kwargs.get('sizes', {})
        self.included = kwargs.get('included', [])

    def evaluate_solution(self, network, solution):

        compliant = True

        for depot, ports in solution.ports.items():

            included_ports = [p for p in ports if p.type in self.included]

            if len(included_ports) > self.sizes.get(depot, np.inf):

                compliant = False

        return compliant

class Vehicle_Limit(Constraint):

    def __init__(self, **kwargs):

        self.sizes = kwargs.get('sizes', {})
        self.included = kwargs.get('included', [])

    def evaluate_solution(self, network, solution):

        compliant = True

        for depot, vehicles in solution.vehicles.items():

            included_vehicles = [v for v in vehicles if v.type in self.included]

            if len(included_vehicles) > self.sizes.get(depot, np.inf):

                compliant = False

        return compliant

class Total_Vehicle_Limit(Constraint):

    def __init__(self, **kwargs):

        self.size = kwargs.get('size', np.inf)
        self.included = kwargs.get('included', [])

    def evaluate_solution(self, network, solution):

        compliant = True

        total_vehicles = 0.

        for depot, vehicles in solution.vehicles.items():

            included_vehicles = [v for v in vehicles if v.type in self.included]

            total_vehicles += len(included_vehicles)

        if total_vehicles > self.size:

            compliant = False

        return compliant

class Total_Port_Limit(Constraint):

    def __init__(self, **kwargs):

        self.size = kwargs.get('size', np.inf)
        self.included = kwargs.get('included', [])

    def evaluate_solution(self, network, solution):

        compliant = True

        total_vehicles = 0.

        for depot, ports in solution.ports.items():

            included_ports = [v for v in ports if v.type in self.included]

            total_ports += len(included_ports)

        if total_ports > self.size:

            compliant = False

        return compliant

class Energy(Constraint):

    def evaluate_tour(self, network, tour):

        return tour['energy'] <= tour['vehicle'].capacity