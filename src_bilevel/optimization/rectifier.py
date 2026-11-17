import numpy as np

class Rectifier():
    '''
    An objective is used to evaluate a solution
    '''

    def __init__(self, **kwargs):

        self.rigid = kwargs.get('rigid', True)

    def rectify(self, network, solution):

        return solution

class Single_Routes(Rectifier):
    '''
    An objective is used to evaluate a solution
    '''
    def rectify(self, network, solution):

        for k, transition in solution.transition.items():

            m, n = transition.shape

            for i in range(m):
                for j in range(n):

                    if (i == 0) or (j == 0) or (i == j):

                        continue

                    transition[i, j] = 0
                    
            transition = transition.T
            transition /= transition.sum(axis = 0)
            transition = transition.T

        return solution

class Bus_Count_Ranges(Rectifier):

    def __init__(self, **kwargs):

        self.bounds = kwargs.get('bounds', (0, np.inf))

    def rectify(self, network, solution):

        fleet_total = max([1, sum(list(solution.fleet_counts.values()))])
        
        if (fleet_total < self.bounds[0]):

            fleet_counts = list(solution.fleet_counts.values())
            fleet_counts_sum = 0

            for idx, fleet_count in enumerate(fleet_counts):

                if idx == (len(fleet_counts) - 1):

                    fleet_counts[idx] = self.bounds[0] - fleet_counts_sum

                else:

                    fleet_counts[idx] = round(fleet_count * self.bounds[0] / fleet_total)

                    fleet_counts_sum += fleet_counts[idx]

        elif (fleet_total > self.bounds[1]):

            fleet_counts = list(solution.fleet_counts.values())
            fleet_counts_sum = 0

            for idx, fleet_count in enumerate(fleet_counts):

                if idx == (len(fleet_counts) - 1):

                    fleet_counts[idx] = self.bounds[1] - fleet_counts_sum

                else:

                    fleet_counts[idx] = round(fleet_count * self.bounds[1] / fleet_total)

                    fleet_counts_sum += fleet_counts[idx]

        else:

            return solution

        solution.fleet_counts = {
            k: fleet_counts[i] for i, k in enumerate(solution.fleet_counts.keys())
            }

        return solution

            

