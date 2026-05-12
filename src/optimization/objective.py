import numpy as np

from scipy.stats import norm

class Objective():
    '''
    An objective is used to evaluate a solution
    '''

    def __init__(self, **kwargs):

        self.minimize = kwargs.get('minimize', True)

    def evaluate_solution(self, network, solution):

        cost = 0.

        return cost

class Routes_Filled():
    '''
    An objective is used to evaluate a solution
    '''

    def __init__(self, **kwargs):

        self.included = kwargs.get('included', [])

    def evaluate_solution(self, network, solution):

        filled_trips = 0
        total_trips = 0

        for tour in solution.tours:

            # print(tour['vehicle'].type)

            total_trips += len(tour['trips']) - 2

            if not tour['vehicle'].type in self.included:

                filled_trips += len(tour['trips']) - 2

        return 1 - filled_trips / total_trips

class Total_Cost(Objective):

    def __init__(self, **kwargs):

        self.schedules_per_period = kwargs.get('schedules_per_period', 1)
        self.discount_rate = kwargs.get('discount_rate', 0)

        self.operational = Operational_Cost(**kwargs)
        self.capital = Capital_Cost(**kwargs)

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.capital.evaluate_solution(network, solution)
        cost += self.operational.evaluate_solution(network, solution)

        return cost

    def describe(self, network, solution, cost = {}):

        cost['capital'] = self.capital.describe(network, solution)
        cost['operational'] = self.operational.describe(network, solution)

        return cost

class Operational_Cost(Objective):
    '''
    Operational costs scale with operational time - i.e. time not in depot.
    The amount of operational time per investment period is computed as follows:

    1. From the state occupation stationary distribution, the portion of total
    tiem that is operational time is computed
    2. Operational time portion is multiplied by the total_time_per_period
    attribute to produce total operational time
    3. Total operational time is multiplied by the operational_cost attribute
    of the vehicle
    '''

    def __init__(self, **kwargs):

        self.schedules_per_period = kwargs.get('schedules_per_period', 1)
        self.discount_rate = kwargs.get('discount_rate', 0)

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.vehicles(network, solution)
        cost += self.stations(network, solution)

        return cost

    def describe(self, network, solution, cost = {}):

        cost['vehicles'] = self.vehicles(network, solution)
        cost['stations'] = self.stations(network, solution)

        return cost

    def stations(self, network, solution):
        '''
        Station operational  costs scale with energy dispensed
        '''

        cost = 0.

        discount = 1 + self.discount_rate

        for event in solution.supply_events:

            port = event['port']

            period_cost = (
                self.schedules_per_period * event['energy'] * port.operational_cost
                )

            cost += sum(
                [period_cost / discount ** p for p in list(range(port.service_periods))]
            )

        return cost

    def vehicles(self, network, solution):

        '''
        Time costs - costs like driver pay which scale with time not
        spent in depots
        '''

        cost = 0.

        discount = 1 + self.discount_rate

        for tour in solution.tours:

            vehicle = tour['vehicle']

            period_cost = (
                self.schedules_per_period * tour['duration'] * vehicle.operational_cost
                )

            cost += sum(
                [period_cost / discount ** p for p \
                in list(range(vehicle.service_periods))]
            )

        return cost

class Capital_Cost(Objective):

    def __init__(self, **kwargs):

        self.discount_rate = kwargs.get('discount_rate', 0)

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.vehicles(network, solution)
        cost += self.stations(network, solution)

        return cost

    def describe(self, network, solution, cost = {}):

        cost['vehicles'] = self.vehicles(network, solution)
        cost['stations'] = self.stations(network, solution)

        return cost

    def vehicles(self, network, solution):

        cost = 0.

        discount = 1 + self.discount_rate

        for depot, vehicles in solution.vehicles.items():

            depot_types = []

            for vehicle in vehicles:

                initial_cost = vehicle.unit_cost

                if not vehicle.type in depot_types:

                    initial_cost += vehicle.fixed_cost
                    depot_types.append(vehicle.type)

                cash_flows = [initial_cost]

                for period in range(1, vehicle.service_periods):

                    cash_flow = vehicle.annual_cost

                    cash_flows.append(cash_flow / discount ** period)

                cash_flows[-1] += (
                    vehicle.disposal_cost / discount ** vehicle.service_periods
                    )

                cost += sum(cash_flows)

        return cost

    def stations(self, network, solution):

        cost = 0.

        discount = 1 + self.discount_rate

        for depot, ports in solution.ports.items():

            depot_types = []

            for port in ports:

                initial_cost = port.unit_cost

                if not port.type in depot_types:

                    initial_cost += port.fixed_cost
                    depot_types.append(port.type)

                cash_flows = [initial_cost]

                for period in range(1, port.service_periods):

                    cash_flow = port.annual_cost

                    cash_flows.append(cash_flow / discount ** period)

                cash_flows[-1] += (
                    port.disposal_cost / discount ** port.service_periods
                    )

                cost += sum(cash_flows)

        return cost

class Initial_Cost(Objective):

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.vehicles(network, solution)
        cost += self.stations(network, solution)

        return cost

    def describe(self, network, solution, cost = {}):

        cost['vehicles'] = self.vehicles(network, solution)
        cost['stations'] = self.stations(network, solution)

        return cost

    def vehicles(self, network, solution):

        cost = 0.

        for depot, vehicles in solution.vehicles.items():

            depot_types = []

            for vehicle in vehicles:

                initial_cost = vehicle.unit_cost

                if not vehicle.type in depot_types:

                    initial_cost += vehicle.fixed_cost
                    depot_types.append(vehicle.type)

                cost += initial_cost

        return cost

    def stations(self, network, solution):

        cost = 0.

        for depot, ports in solution.ports.items():

            depot_types = []

            for port in ports:

                initial_cost = port.unit_cost

                if not port.type in depot_types:

                    initial_cost += port.fixed_cost
                    depot_types.append(port.type)

                cost += initial_cost

        return cost

class Emissions_Cost(Objective):
    '''
    Operational costs scale with operational time - i.e. time not in depot.
    The amount of operational time per investment period is computed as follows:

    1. From the state occupation stationary distribution, the portion of total
    tiem that is operational time is computed
    2. Operational time portion is multiplied by the total_time_per_period
    attribute to produce total operational time
    3. Total operational time is multiplied by the operational_cost attribute
    of the vehicle
    '''

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.stations(network, solution)

        return cost

    def stations(self, network, solution):

        '''
        Time costs - costs like driver pay which scale with time not
        spent in depots
        '''

        cost = 0.

        for event in solution.supply_events:

            port = event['port']

            cost += event['energy'] * port.emissions

        return cost

class Daily_Cost(Objective):
    '''
    Operational costs scale with operational time - i.e. time not in depot.
    The amount of operational time per investment period is computed as follows:

    1. From the state occupation stationary distribution, the portion of total
    tiem that is operational time is computed
    2. Operational time portion is multiplied by the total_time_per_period
    attribute to produce total operational time
    3. Total operational time is multiplied by the operational_cost attribute
    of the vehicle
    '''

    def __init__(self, **kwargs):

        self.emissions_cost = kwargs.get('emissions_cost', 0.)

    def evaluate_solution(self, network, solution):

        cost = 0.

        cost += self.vehicles(network, solution)
        cost += self.stations(network, solution)

        # print(cost)

        return cost

    def describe(self, network, solution, cost = {}):

        cost['vehicles'] = self.vehicles(network, solution)
        cost['stations'] = self.stations(network, solution)

        return cost

    def stations(self, network, solution):
        '''
        Station operational  costs scale with energy dispensed
        '''

        cost = 0.

        for event in solution.supply_events:

            port = event['port']

            cost += event['energy'] * port.operational_cost
            cost += event['energy'] * port.emissions * self.emissions_cost

        return cost

    def vehicles(self, network, solution):

        '''
        Time costs - costs like driver pay which scale with time not
        spent in depots
        '''

        cost = 0.

        for tour in solution.tours:

            vehicle = tour['vehicle']

            cost += tour['duration'] * vehicle.operational_cost

        return cost