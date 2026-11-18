import numpy as np

class Constraint():
    '''
    An objective is used to evaluate a solution
    '''

    def __init__(self, penalty=1, **kwargs):

        self.penalty = penalty

    def evaluate(self, network, solution):

        return 0

class Fleet_Portion(Constraint):

    def __init__(self, included=[], portion=0, penalty=1, **kwargs):

        self.included = included
        self.portion = portion
        self.penalty = penalty

    def evaluate(self, network, solution):

        n_included = 0
        n_total = 0

        for depot, vehicles in solution.vehicles.items():

            for vehicle in vehicles:

                n_total += 1

                if vehicle.type in self.included:

                    n_included += 1

        compliant = n_included / n_total >= self.portion

        return self.penalty * (not compliant)

class Port_Limit(Constraint):

    def __init__(self, sizes={}, penalty=1, **kwargs):

        self.sizes = sizes
        self.penalty = penalty

    def evaluate(self, network, solution):

        penalty = 0

        for depot, ports in solution.ports.items():

            if len(ports) > self.sizes.get(depot, np.inf):

                penalty += self.penalty

        return penalty

class Lot_Size_Limit(Constraint):

    def __init__(self, sizes={}, penalty=1, **kwargs):

        self.sizes = sizes
        self.penalty = penalty

    def evaluate(self, network, solution):

        penalty = 0

        for depot, vehicles in solution.vehicles.items():

            if len(vehicles) > self.sizes.get(depot, np.inf):

                penalty += self.penalty

        return penalty