"""
rust_bridge.py — Converts the Python Network data model into the flat array
format expected by fleet_opt_core.RustNetwork.

This module is the only place that knows about both the Python data model
(NetworkX graphs, Vehicle_Type / Port_Type objects, objective classes) and
the Rust interface. The existing Network class and example scripts are
not modified.

Usage:
    from src.rust_bridge import build_rust_network
    rust_net, index_maps = build_rust_network(network)
    print(rust_net.status())
"""

import numpy as np
import fleet_opt_core

from .optimization import Solution

# ---------------------------------------------------------------------------
# Objective class name → Rust ObjectiveKind string
# ---------------------------------------------------------------------------
_OBJECTIVE_KIND_MAP = {
    'Initial_Cost':    'InitialCost',
    'Daily_Cost':      'DailyCost',
    'Capital_Cost':    'Capital',
    'Operational_Cost':'Operational',
    'Emissions_Cost':  'Emissions',
}

def _objective_params(obj):
    """Extract numeric parameters from a Python objective instance."""
    params = {}
    if hasattr(obj, 'discount_rate'):
        params['discount_rate'] = float(obj.discount_rate)
    if hasattr(obj, 'schedules_per_period'):
        params['schedules_per_period'] = float(obj.schedules_per_period)
    if hasattr(obj, 'emissions_cost'):
        params['emissions_cost'] = float(obj.emissions_cost)
    return params

def build_rust_network(network):
    """
    Convert a Python Network instance into a fleet_opt_core.RustNetwork.

    Performs all NetworkX → flat-array conversions on the Python side so
    the Rust constructor receives only numpy arrays and Python lists.
    The Rust extension owns all data after this call returns.

    Parameters
    ----------
    network : src.optimization.network.Network

    Returns
    -------
    rust_net : fleet_opt_core.RustNetwork
    index_maps : dict
        Bidirectional mappings between string IDs and integer indices,
        needed to reconstruct Python Solution objects from Rust output
        in Step 5.
    """
    trips_graph     = network.trips
    locations_graph = network.locations

    # -----------------------------------------------------------------------
    # 1. Location index mapping
    #    Assign a stable integer index to each location node.
    #    Depots must be at indices matching network.depots ordering so that
    #    Rust's depot_sentinel encoding (n_trips + depot_idx) is consistent.
    # -----------------------------------------------------------------------
    location_nodes = list(locations_graph.nodes)
    loc_to_idx = {loc: i for i, loc in enumerate(location_nodes)}
    n_locations = len(location_nodes)

    # -----------------------------------------------------------------------
    # 2. Location duration / distance matrices (n_locations × n_locations)
    #    Replicates: locations._adj[s][t].get('duration', 0)
    #    Missing edges default to 0.0, matching Python behaviour.
    # -----------------------------------------------------------------------
    duration_matrix = np.zeros((n_locations, n_locations), dtype=np.float64)
    distance_matrix = np.zeros((n_locations, n_locations), dtype=np.float64)

    for s, neighbors in locations_graph._adj.items():
        i = loc_to_idx[s]
        for t, attrs in neighbors.items():
            j = loc_to_idx[t]
            duration_matrix[i, j] = attrs.get('duration', 0.0)
            distance_matrix[i, j] = attrs.get('distance', 0.0)

    # -----------------------------------------------------------------------
    # 3. Trip index mapping and attribute arrays
    # -----------------------------------------------------------------------
    trip_nodes  = list(trips_graph.nodes)
    trip_to_idx = {trip: i for i, trip in enumerate(trip_nodes)}
    n_trips     = len(trip_nodes)

    trip_starts                 = np.zeros(n_trips, dtype=np.float64)
    trip_finishes               = np.zeros(n_trips, dtype=np.float64)
    trip_distances              = np.zeros(n_trips, dtype=np.float64)
    trip_durations              = np.zeros(n_trips, dtype=np.float64)
    trip_start_location_indices = np.zeros(n_trips, dtype=np.int64)
    trip_end_location_indices   = np.zeros(n_trips, dtype=np.int64)

    for trip_id, idx in trip_to_idx.items():
        attrs = trips_graph._node[trip_id]
        trip_starts[idx]                 = float(attrs.get('start',    0.0))
        trip_finishes[idx]               = float(attrs.get('finish',   0.0))
        trip_distances[idx]              = float(attrs.get('distance', 0.0))
        trip_durations[idx]              = float(attrs.get('duration', 0.0))
        trip_start_location_indices[idx] = loc_to_idx[attrs['start_location']]
        trip_end_location_indices[idx]   = loc_to_idx[attrs['finish_location']]

    # -----------------------------------------------------------------------
    # 4. Trip adjacency edge lists
    #    Extract all directed edges from the trips DiGraph as parallel
    #    integer-index arrays (source, target).
    # -----------------------------------------------------------------------
    adj_source_list = []
    adj_target_list = []
    for s, neighbors in trips_graph._adj.items():
        src_idx = trip_to_idx[s]
        for t in neighbors:
            adj_source_list.append(src_idx)
            adj_target_list.append(trip_to_idx[t])

    adj_sources = np.array(adj_source_list, dtype=np.int64)
    adj_targets = np.array(adj_target_list, dtype=np.int64)

    # -----------------------------------------------------------------------
    # 5. Depot location indices and sampling probabilities
    #    network.depots is a list of location node IDs that are depots.
    #    The order here defines depot_idx (0, 1, ...) used in Rust.
    #    depot_probabilities mirrors Python's generate_depot() weighting.
    # -----------------------------------------------------------------------
    depot_location_indices = [loc_to_idx[loc] for loc in network.depots]

    depot_prob_raw = np.array(
        [locations_graph._node[d].get('probability', 1.0) for d in network.depots],
        dtype=np.float64,
    )
    depot_probabilities = depot_prob_raw / depot_prob_raw.sum()

    # -----------------------------------------------------------------------
    # 6. Vehicle type arrays
    # -----------------------------------------------------------------------
    vehicle_type_names = list(network.vehicle_types.keys())
    vt_name_to_idx     = {name: i for i, name in enumerate(vehicle_type_names)}

    # Port type index mapping is needed to build vt_supply_type_indices
    port_type_names = list(network.port_types.keys())
    pt_name_to_idx  = {name: i for i, name in enumerate(port_type_names)}

    n_vt = len(vehicle_type_names)
    vt_capacities        = np.zeros(n_vt, dtype=np.float64)
    vt_resupply_rates    = np.zeros(n_vt, dtype=np.float64)
    vt_consumptions      = np.zeros(n_vt, dtype=np.float64)
    vt_emissions         = np.zeros(n_vt, dtype=np.float64)
    vt_fixed_costs       = np.zeros(n_vt, dtype=np.float64)
    vt_unit_costs        = np.zeros(n_vt, dtype=np.float64)
    vt_annual_costs      = np.zeros(n_vt, dtype=np.float64)
    vt_disposal_costs    = np.zeros(n_vt, dtype=np.float64)
    vt_service_periods   = np.zeros(n_vt, dtype=np.int64)
    vt_operational_costs = np.zeros(n_vt, dtype=np.float64)
    vt_supply_type_indices = []  # list[list[int]]
    vt_prob_raw            = np.zeros(n_vt, dtype=np.float64)

    for name, i in vt_name_to_idx.items():
        vt = network.vehicle_types[name]
        vt_capacities[i]        = vt.capacity
        vt_resupply_rates[i]    = vt.resupply_rate
        vt_consumptions[i]      = vt.consumption
        vt_emissions[i]         = vt.emissions
        vt_fixed_costs[i]       = vt.fixed_cost
        vt_unit_costs[i]        = vt.unit_cost
        vt_annual_costs[i]      = vt.annual_cost
        vt_disposal_costs[i]    = vt.disposal_cost
        vt_service_periods[i]   = vt.service_periods
        vt_operational_costs[i] = vt.operational_cost
        vt_supply_type_indices.append(
            [pt_name_to_idx[pt] for pt in vt.supply_types]
        )
        vt_prob_raw[i] = float(getattr(vt, 'probability', 1.0))

    # Normalize to a proper probability distribution (mirrors Python's generate_equipment)
    vt_probabilities = vt_prob_raw / vt_prob_raw.sum()
    # print(vt_probabilities)

    # vt_probabilities = np.random.default_rng().uniform(0, 1, size = len(vt_probabilities))
    # vt_probabilities /= vt_probabilities.sum()

    # print(vt_probabilities)

    # -----------------------------------------------------------------------
    # 7. Port type arrays
    # -----------------------------------------------------------------------
    n_pt = len(port_type_names)
    pt_resupply_rates    = np.zeros(n_pt, dtype=np.float64)
    pt_operating_times   = np.zeros(n_pt, dtype=np.float64)
    pt_efficiencies      = np.zeros(n_pt, dtype=np.float64)
    pt_fixed_costs       = np.zeros(n_pt, dtype=np.float64)
    pt_unit_costs        = np.zeros(n_pt, dtype=np.float64)
    pt_annual_costs      = np.zeros(n_pt, dtype=np.float64)
    pt_disposal_costs    = np.zeros(n_pt, dtype=np.float64)
    pt_service_periods   = np.zeros(n_pt, dtype=np.int64)
    pt_operational_costs = np.zeros(n_pt, dtype=np.float64)
    pt_emissions         = np.zeros(n_pt, dtype=np.float64)

    for name, i in pt_name_to_idx.items():
        pt = network.port_types[name]
        pt_resupply_rates[i]    = pt.resupply_rate
        pt_operating_times[i]   = pt.operating_time
        pt_efficiencies[i]      = pt.efficiency
        pt_fixed_costs[i]       = pt.fixed_cost
        pt_unit_costs[i]        = pt.unit_cost
        pt_annual_costs[i]      = pt.annual_cost
        pt_disposal_costs[i]    = pt.disposal_cost
        pt_service_periods[i]   = pt.service_periods
        pt_operational_costs[i] = pt.operational_cost
        pt_emissions[i]         = getattr(pt, 'emissions', 0.0)

    # -----------------------------------------------------------------------
    # 8. Objective configuration
    # -----------------------------------------------------------------------
    objective_kinds       = []
    objective_params_list = []
    objective_names       = list(network.objectives.keys())

    for obj in network.objectives.values():
        class_name = type(obj).__name__
        kind = _OBJECTIVE_KIND_MAP.get(class_name)
        if kind is None:
            raise ValueError(
                f"Objective class '{class_name}' is not supported by the Rust core. "
                f"Supported: {list(_OBJECTIVE_KIND_MAP.keys())}"
            )
        objective_kinds.append(kind)
        objective_params_list.append(_objective_params(obj))

    # -----------------------------------------------------------------------
    # 9. Constraint configuration
    #    Extract Fleet_Portion parameters; Energy is handled implicitly
    #    inside the Rust solve pipeline (tour splitting).
    # -----------------------------------------------------------------------
    fleet_portion_included = []
    fleet_portion_min      = 0.0
    _NO_LIMIT              = 2**63 - 1  # maps to usize::MAX in Rust
    port_limits            = [_NO_LIMIT] * len(network.depots)
    lot_size_limits        = [_NO_LIMIT] * len(network.depots)

    _depot_to_idx = {name: idx for idx, name in enumerate(network.depots)}

    for constraint in network.constraints.values():
        if hasattr(constraint, 'included') and hasattr(constraint, 'portion'):
            fleet_portion_included = [
                vt_name_to_idx[name]
                for name in constraint.included
                if name in vt_name_to_idx
            ]
            fleet_portion_min = float(constraint.portion)
        elif type(constraint).__name__ == 'Port_Limit' and hasattr(constraint, 'sizes'):
            for depot_name, limit in constraint.sizes.items():
                if depot_name in _depot_to_idx:
                    port_limits[_depot_to_idx[depot_name]] = int(limit)
        elif type(constraint).__name__ == 'Lot_Size_Limit' and hasattr(constraint, 'sizes'):
            for depot_name, limit in constraint.sizes.items():
                if depot_name in _depot_to_idx:
                    lot_size_limits[_depot_to_idx[depot_name]] = int(limit)

    # -----------------------------------------------------------------------
    # 10. Construct RustNetwork (crosses PyO3 boundary once)
    # -----------------------------------------------------------------------
    rust_net = fleet_opt_core.RustNetwork(
        trip_starts, trip_finishes, trip_distances, trip_durations,
        trip_start_location_indices, trip_end_location_indices,
        adj_sources, adj_targets,
        duration_matrix, distance_matrix,
        depot_location_indices,
        depot_probabilities,
        vt_capacities, vt_resupply_rates, vt_consumptions, vt_emissions,
        vt_fixed_costs, vt_unit_costs, vt_annual_costs, vt_disposal_costs,
        vt_service_periods, vt_operational_costs, vt_supply_type_indices,
        vt_probabilities,
        pt_resupply_rates, pt_operating_times, pt_efficiencies,
        pt_fixed_costs, pt_unit_costs, pt_annual_costs, pt_disposal_costs,
        pt_service_periods, pt_operational_costs, pt_emissions,
        objective_kinds, objective_params_list, objective_names,
        fleet_portion_included, fleet_portion_min,
        port_limits, lot_size_limits,
    )

    # -----------------------------------------------------------------------
    # 11. Return index mappings for Step 5 result reconstruction
    #     When Rust returns the final population as integer arrays, these
    #     maps translate back to the string IDs Python expects.
    # -----------------------------------------------------------------------
    index_maps = {
        'trip_to_idx':    trip_to_idx,
        'idx_to_trip':    {v: k for k, v in trip_to_idx.items()},
        'vt_name_to_idx': vt_name_to_idx,
        'idx_to_vt_name': {v: k for k, v in vt_name_to_idx.items()},
        'pt_name_to_idx': pt_name_to_idx,
        'idx_to_pt_name': {v: k for k, v in pt_name_to_idx.items()},
        'loc_to_idx':     loc_to_idx,
        'idx_to_loc':     {v: k for k, v in loc_to_idx.items()},
        # depot_order[depot_idx] → depot name.
        # Rust returns start_depot and end_depot as integer index arrays;
        # translate with: depot_name = index_maps['depot_order'][depot_idx]
        'depot_order': network.depots,
        'n_trips': rust_net.n_trips(),
        'n_depots': rust_net.n_depots(),
        'obj_names': list(network.objectives.keys()),
        'vt_names': list(network.vehicle_types.keys()),
        'pt_names': list(network.port_types.keys()),
        'multi_depot': rust_net.n_depots() > 1,
    }

    return rust_net, index_maps

def rust_pop_to_python_format(rust_population, index_maps):
    """
    Convert Rust optimize() output into dicts accepted by Solution.from_data().

    Gene mapping:
      successors : int idx → trip_id string; values >= N_TRIPS → 'depot'
      equipment  : vt_idx  → vt_name string
      supply_type: pt_idx  → pt_name string
      depot      : start_depot_idx → depot_name  (Python single-depot gene)
    """

    idx_to_trip  = index_maps['idx_to_trip']
    idx_to_vt    = index_maps['idx_to_vt_name']
    idx_to_pt    = index_maps['idx_to_pt_name']
    depot_order  = index_maps['depot_order'] 

    output = {}
    for sol_idx, sol in enumerate(rust_population):
        successors = {
            idx_to_trip[i]: ('depot' if s >= index_maps['n_trips'] else idx_to_trip[s])
            for i, s in enumerate(sol['successors'])
        }
        equipment = {
            idx_to_trip[i]: idx_to_vt[v] for i, v in enumerate(sol['equipment'])
        }
        supply_type = {
            idx_to_trip[i]: idx_to_pt[v] for i, v in enumerate(sol['supply_type'])
        }
        # start_depot gene → Python depot name (used by _voting and _tour_information)
        depot = {
            idx_to_trip[i]: depot_order[d]
            for i, d in enumerate(sol['start_depot'])
        }
        fitness = {
            name: sol['fitness'][j] for j, name in enumerate(index_maps['obj_names'])
        }
        output[sol_idx] = {
            'rank':        sol['rank'],
            'age':         sol['age'],
            'feasible':    sol['feasible'],
            'compliant':   sol['compliant'],
            'fitness':     fitness,
            'successors':  successors,
            'equipment':   equipment,
            'supply_type': supply_type,
            'depot':       depot,
        }
    return output

def resolve_pareto_front(network, pop_dict, index_maps, max_solutions = 40, rank = 0):
    """
    Re-solve rank-0 compliant solutions through the Python pipeline to
    recover full tour / vehicle / port data. Returns solved Solution objects
    sorted by ascending first objective.

    from_data() calls solve+evaluate but leaves solution.rank=None (Solution.__init__
    default). rank_population() is called here to assign proper ranks so that
    population_to_dict() produces 'rank': 0 entries that downstream filters expect.
    """
    front = sorted(
        [v for v in pop_dict.values() if v['rank'] <= rank],
        key=lambda s: s['fitness'][index_maps['obj_names'][0]]
    )
    stride = max(1, len(front) // max_solutions)
    front  = front[::stride]
    solutions = [Solution(network).from_data(network, data) for data in front]
    network.rank_population(solutions)
    print(f'Re-solved {len(solutions)} Pareto-front solutions with Python.')
    return solutions