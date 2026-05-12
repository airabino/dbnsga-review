import os
import sys
import time
import numpy as np
import networkx as nx

class Vehicle_Type():

	def __init__(self, **kwargs):

		self.type = kwargs.get('type', 'vehicle')

		# Supply type - vehicles may have multiple
		self.supply_types = kwargs.get('supply_types', [])

		# Maximum amount of energy storage
		self.capacity = kwargs.get('capacity', 1)

		# Maximum rate of energy resupply
		self.resupply_rate = kwargs.get('resupply_rate', 1)

		# Energy consumption rate
		self.consumption = kwargs.get('consumption', 1)

		# Maximum rate of energy resupply
		self.emissions = kwargs.get('emissions', 0)

		# Cost parameters (scaling with service life)
		# Cost to acquire the vehicle
		self.fixed_cost = kwargs.get('fixed_cost', 0)
		self.unit_cost = kwargs.get('unit_cost', 0)
		self.annual_cost = kwargs.get('annual_cost', 0)
		self.disposal_cost = kwargs.get('disposal_cost', 0)
		self.service_periods = kwargs.get('service_periods', 1)

		# Operational cost (scaling with time operated)
		self.operational_cost = kwargs.get('operational_cost', 0)

		# Energy consumption rate
		self.consumption = kwargs.get('consumption', 1)

	def energy(self, tour, **kwargs):

		energy_consumed = self.consumption * tour['distance']

		return energy_consumed