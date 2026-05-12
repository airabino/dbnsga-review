import os
import sys
import time
import numpy as np
import networkx as nx

class Port_Type():

	def __init__(self, **kwargs):

		self.type = kwargs.get('type', 'port')

		# Supply type
		self.supply_type = kwargs.get('supply_type', '')

		# Maximum rate of energy resupply
		self.resupply_rate = kwargs.get('resupply_rate', 1)

		# Excess time
		self.operating_time = kwargs.get('operating_time', 0)

		# Efficiency of resupply equipment
		self.efficiency = kwargs.get('efficiency', 1)

		# Maximum rate of energy resupply
		self.emissions = kwargs.get('emissions', 0)

		# Cost parameters (scaling with service life)
		# Cost to acquire the station
		self.fixed_cost = kwargs.get('fixed_cost', 0)
		self.unit_cost = kwargs.get('unit_cost', 0)
		self.annual_cost = kwargs.get('annual_cost', 0)
		self.disposal_cost = kwargs.get('disposal_cost', 0)
		self.service_periods = kwargs.get('service_periods', 1)

		# Operational cost (scaling with energy dispensed)
		self.operational_cost = kwargs.get('operational_cost', 0)

	def resupply_time(self, vehicle, energy):

		power = min([vehicle.resupply_rate, self.resupply_rate])

		event_time = energy / power + self.operating_time

		return event_time