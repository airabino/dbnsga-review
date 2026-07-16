import numpy as np

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

    def evaluate_solution(self, network, solution):

        compliant = True

        for depot, ports in solution.ports.items():

            if len(ports) > self.sizes.get(depot, np.inf):

                compliant = False

        return compliant

class Lot_Size_Limit(Constraint):

    def __init__(self, **kwargs):

        self.sizes = kwargs.get('sizes', {})

    def evaluate_solution(self, network, solution):

        compliant = True

        for depot, vehicles in solution.vehicles.items():

            # print(depot, len(vehicles))

            if len(vehicles) > self.sizes.get(depot, np.inf):

                compliant = False

        # print(compliant)

        return compliant


class Energy(Constraint):

    def evaluate_tour(self, network, tour):

        return tour['energy'] <= tour['vehicle'].capacity