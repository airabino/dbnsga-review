// Suppress dead-code warnings for fields and methods defined in this step but
// not yet called — they will be used as Steps 4–6 are implemented.
#![allow(dead_code)]

// =============================================================================
// types.rs — Core data structures for the fleet optimization Rust core.
//
// Design principles:
//   - All Python dicts and NetworkX graph lookups are replaced with flat Vec
//     arrays and pre-computed matrices, eliminating Python object overhead.
//   - Successor encoding: indices 0..n_trips are trip indices;
//     indices n_trips..(n_trips + n_depots) are depot sentinels.
//   - NetworkData is immutable after construction and shared read-only across
//     all Rayon threads (satisfies Rust's Sync requirement without locks).
//   - Solution is Clone so offspring can be created from parents.
// =============================================================================

// -----------------------------------------------------------------------------
// Vehicle and port type definitions
// -----------------------------------------------------------------------------

/// Performance and cost parameters for one vehicle type (e.g. SRBEV, ICEV).
/// Replaces Python's Vehicle_Type class. All fields are plain f64 / usize —
/// no heap allocation per instance, full cache locality.
#[derive(Clone, Debug)]
pub struct VehicleType {
    /// J — maximum energy storage capacity.
    pub capacity: f64,
    /// J/s — maximum charge or fuel rate (vehicle-side limit).
    pub resupply_rate: f64,
    /// J/m — energy consumed per metre of travel.
    pub consumption: f64,
    /// g CO₂-eq/m — tailpipe emissions per metre (0 for BEV/FCEV).
    pub emissions: f64,
    /// $ — one-time setup cost applied once per depot per vehicle type.
    pub fixed_cost: f64,
    /// $ — per-vehicle acquisition cost.
    pub unit_cost: f64,
    /// $/year — annual maintenance cost per vehicle.
    pub annual_cost: f64,
    /// $ — end-of-life disposal cost per vehicle.
    pub disposal_cost: f64,
    /// Years of service life (used for NPV calculations).
    pub service_periods: usize,
    /// $/s — time-based operational cost (driver pay, etc.) per second of tour.
    pub operational_cost: f64,
    /// Indices into NetworkData::port_types for compatible supply types.
    /// Replaces Python's supply_types: List[str].
    pub supply_type_indices: Vec<usize>,
}

impl VehicleType {
    /// Energy consumed by a vehicle running a tour of the given distance.
    #[inline]
    pub fn energy(&self, distance: f64) -> f64 {
        self.consumption * distance
    }
}

/// Performance and cost parameters for one charging/fueling port type
/// (e.g. AC-PLUG, DC-PANTO, DIESEL-PUMP).
/// Replaces Python's Port_Type class.
#[derive(Clone, Debug)]
pub struct PortType {
    /// J/s — maximum power output (port-side limit).
    pub resupply_rate: f64,
    /// s — fixed time overhead per charging/fueling event (plug-in, safety checks, etc.).
    pub operating_time: f64,
    /// Dimensionless — energy transfer efficiency (1.0 = lossless).
    pub efficiency: f64,
    /// $ — one-time installation cost applied once per depot per port type.
    pub fixed_cost: f64,
    /// $ — per-port equipment cost.
    pub unit_cost: f64,
    /// $/year — annual operation and maintenance cost per port.
    pub annual_cost: f64,
    /// $ — end-of-life disposal cost per port.
    pub disposal_cost: f64,
    /// Years of service life.
    pub service_periods: usize,
    /// $/J — energy-based operational cost per joule dispensed.
    pub operational_cost: f64,
    /// g CO₂-eq/J — emissions per joule of energy dispensed (0 for zero-emission ports).
    pub emissions: f64,
}

impl PortType {
    /// Time (s) required to restore `energy` joules to a vehicle, given the
    /// vehicle's maximum resupply_rate. Effective power is the minimum of
    /// port and vehicle rates.
    #[inline]
    pub fn resupply_time(&self, vehicle_resupply_rate: f64, energy: f64) -> f64 {
        let power = vehicle_resupply_rate.min(self.resupply_rate);
        energy / power + self.operating_time
    }
}

// -----------------------------------------------------------------------------
// Trip representation
// -----------------------------------------------------------------------------

/// Attributes of a single scheduled trip, pre-extracted from the GTFS trip graph.
/// Replaces per-trip dict lookups on NetworkX node attributes.
#[derive(Clone, Debug)]
pub struct Trip {
    /// Index into the location graph for the trip's departure point.
    pub start_location_idx: usize,
    /// Index into the location graph for the trip's arrival point.
    pub end_location_idx: usize,
    /// Route distance in metres.
    pub distance: f64,
    /// Route duration in seconds.
    pub duration: f64,
    /// Scheduled departure time, seconds from midnight.
    pub start: f64,
    /// Scheduled arrival time, seconds from midnight.
    pub finish: f64,
}

// -----------------------------------------------------------------------------
// Solved-state structures (populated by Solution::solve)
// -----------------------------------------------------------------------------

/// A solved vehicle block: an ordered sequence of trips served by a single
/// physical vehicle. Equivalent to Python's tour dict.
///
/// tour.trips = [trip_A, ..., trip_Z] — real trip indices only, no sentinel brackets.
/// Depot assignment is determined by voting and stored in start_depot_idx / end_depot_idx.
#[derive(Clone, Debug, Default)]
pub struct Tour {
    /// Ordered sequence of real trip indices: [trip_A, ..., trip_Z].
    /// No depot sentinels — depot identity is in start_depot_idx / end_depot_idx.
    pub trips: Vec<usize>,
    /// Index into NetworkData::vehicle_types.
    pub vehicle_type_idx: usize,
    /// Index into NetworkData::port_types (the supply type for this tour).
    pub port_type_idx: usize,
    /// Unique vehicle instance identifier within this solution.
    pub vehicle_instance_idx: usize,
    /// Depot from which the vehicle departs (index into NetworkData::depot_location_indices).
    pub start_depot_idx: usize,
    /// Depot to which the vehicle returns (index into NetworkData::depot_location_indices).
    /// Equal to start_depot_idx when same_depot=true.
    pub end_depot_idx: usize,
    /// Wall-clock departure time (s from midnight), including deadhead from start depot.
    pub start: f64,
    /// Wall-clock return time (s from midnight), including deadhead to end depot.
    pub finish: f64,
    /// Total distance travelled including deadhead legs (m).
    pub distance: f64,
    /// Total duration including deadhead legs (s).
    pub duration: f64,
    /// Energy consumed over the tour (J).
    pub energy: f64,
}

/// A single charging or fueling event in the gap between two consecutive tours
/// of the same vehicle, or after the vehicle's final tour of the day.
/// Equivalent to Python's supply_event dict.
#[derive(Clone, Debug, Default)]
pub struct SupplyEvent {
    /// Unique vehicle instance identifier.
    pub vehicle_instance_idx: usize,
    /// Index into NetworkData::vehicle_types.
    pub vehicle_type_idx: usize,
    /// Index into NetworkData::port_types.
    pub port_type_idx: usize,
    /// Unique port instance identifier within this solution.
    pub port_instance_idx: usize,
    /// Index into NetworkData::depot_location_indices.
    pub depot_idx: usize,
    /// Earliest the event can start (= previous tour finish time).
    pub earliest_start: f64,
    /// Latest the event can finish (= next tour start time, or end of day).
    pub latest_finish: f64,
    /// Energy to restore (J) — equals the previous tour's energy consumption.
    pub energy: f64,
    /// Actual scheduled start time (set by _port_assignment).
    pub start: f64,
    /// Actual scheduled finish time (set by _port_assignment).
    pub finish: f64,
}

// -----------------------------------------------------------------------------
// Objective and constraint configuration
// -----------------------------------------------------------------------------

/// Objective function variant. Mirrors the Python objective class hierarchy
/// but as a data enum, enabling zero-cost dispatch in a match expression.
#[derive(Clone, Debug)]
pub enum ObjectiveKind {
    /// Sum of unit_cost for all vehicles and ports (no amortization).
    /// Mirrors Python's Initial_Cost.
    InitialCost,
    /// Daily operational cost: tour_duration × vehicle.operational_cost
    ///   + energy × port.operational_cost
    ///   + energy × port.emissions × emissions_cost.
    /// Mirrors Python's Daily_Cost.
    DailyCost {
        emissions_cost: f64,
    },
    /// NPV of acquisition + annual maintenance + disposal for all assets.
    /// Mirrors Python's Capital_Cost.
    Capital {
        discount_rate: f64,
    },
    /// NPV of time-based vehicle costs + energy-based port costs, scaled by
    /// schedules_per_period. Mirrors Python's Operational_Cost.
    Operational {
        schedules_per_period: f64,
        discount_rate: f64,
    },
    /// Sum of port.emissions × event.energy over all supply events.
    /// Mirrors Python's Emissions_Cost.
    Emissions,
}

/// Active constraint configuration for a run.
/// Mirrors Python's constraint dict (Energy is always active; Fleet_Portion is optional).
/// Energy feasibility is checked per-tour during vehicle assignment and handled
/// via tour splitting — it is not stored here.
#[derive(Clone, Debug)]
pub struct ConstraintConfig {
    /// Vehicle type indices that count toward the electrification mandate.
    /// Empty vec means Fleet_Portion constraint is disabled.
    pub fleet_portion_included: Vec<usize>,
    /// Minimum fraction of total fleet that must be of an included type.
    pub fleet_portion_min: f64,
    /// Maximum number of ports per depot (indexed by depot_idx).
    /// usize::MAX means no limit (mirrors Python's np.inf default).
    /// Empty vec means Port_Limit constraint is disabled.
    pub port_limits: Vec<usize>,
    /// Maximum number of vehicles per depot (indexed by depot_idx).
    /// usize::MAX means no limit (mirrors Python's np.inf default).
    /// Empty vec means Lot_Size_Limit constraint is disabled.
    pub lot_size_limits: Vec<usize>,
}

// -----------------------------------------------------------------------------
// Network (immutable problem definition, shared across Rayon threads)
// -----------------------------------------------------------------------------

/// The complete, immutable problem definition. Constructed once at the PyO3
/// boundary from Python numpy arrays and held for the duration of optimize().
///
/// All NetworkX graph data is flattened into Vec arrays here. The Rayon
/// parallel iterator holds a shared &NetworkData reference — this is safe
/// because NetworkData contains no interior mutability.
pub struct NetworkData {
    // --- Trip data ---
    /// One Trip struct per scheduled trip. Indexed by trip_idx (0..n_trips).
    pub trips: Vec<Trip>,
    pub n_trips: usize,
    /// Adjacency list for the trip successor graph (from the GTFS trips DiGraph).
    /// trip_adj[i] lists all trip indices that are valid successors of trip i.
    pub trip_adj: Vec<Vec<usize>>,

    // --- Location data ---
    /// Flat row-major travel duration matrix (seconds).
    /// Access: duration_matrix[from_loc * n_locations + to_loc]
    /// Replaces: network.locations._adj[s][t]['duration']
    pub duration_matrix: Vec<f64>,
    /// Flat row-major travel distance matrix (metres).
    /// Access: distance_matrix[from_loc * n_locations + to_loc]
    /// Replaces: network.locations._adj[s][t]['distance']
    pub distance_matrix: Vec<f64>,
    pub n_locations: usize,

    // --- Depot configuration ---
    /// Location indices (into the flat location array) that correspond to depots.
    /// Depot sentinel encoding: any successor value >= n_trips maps to the single
    /// sentinel slot n_trips, meaning "return to depot" — the actual depot is
    /// determined by voting on the start_depot / end_depot gene arrays.
    pub depot_location_indices: Vec<usize>,
    pub n_depots: usize,
    /// Normalised sampling probabilities for depot gene generation.
    /// Mirrors depot `probability` attribute (default 1.0, then normalised).
    pub depot_probabilities: Vec<f64>,

    // --- Vehicle and port types ---
    pub vehicle_types: Vec<VehicleType>,
    pub n_vehicle_types: usize,
    /// Normalized sampling probabilities for generate_solution equipment selection.
    /// Mirrors Python's Vehicle_Type.probability attribute (default 1.0, then normalized).
    pub vt_probabilities: Vec<f64>,
    pub port_types: Vec<PortType>,
    pub n_port_types: usize,

    // --- Objectives and constraints ---
    /// Ordered list of objectives. fitness[i] corresponds to objectives[i].
    pub objectives: Vec<ObjectiveKind>,
    /// User-facing objective names (e.g. "Capital Cost", "Operational Cost").
    /// Used to key fitness dicts in generation history output.
    pub objective_names: Vec<String>,
    pub constraints: ConstraintConfig,
}

impl NetworkData {
    // -------------------------------------------------------------------------
    // Successor encoding helpers
    // -------------------------------------------------------------------------

    /// Returns true if the encoded node is a depot sentinel rather than a trip.
    #[inline]
    pub fn is_depot(&self, node: usize) -> bool {
        node >= self.n_trips
    }

    /// Returns the depot index (0..n_depots) for a depot sentinel node.
    /// Panics in debug builds if called on a trip node.
    #[inline]
    pub fn depot_idx_of(&self, node: usize) -> usize {
        debug_assert!(self.is_depot(node), "depot_idx_of called on trip node {}", node);
        node - self.n_trips
    }

    /// Returns the single depot sentinel node index (always n_trips).
    /// The depot_idx argument is ignored — there is now exactly one sentinel value.
    #[inline]
    pub fn depot_sentinel(&self, _depot_idx: usize) -> usize {
        self.n_trips
    }

    // -------------------------------------------------------------------------
    // Location lookup helpers (replace NetworkX dict traversal)
    // -------------------------------------------------------------------------

    /// Returns the departure-side location index for any encoded node.
    /// For trips: the trip's start_location_idx.
    /// For depot sentinels: the depot's physical location (depots have one location).
    #[inline]
    pub fn start_location_of(&self, node: usize) -> usize {
        if node < self.n_trips {
            self.trips[node].start_location_idx
        } else {
            self.depot_location_indices[node - self.n_trips]
        }
    }

    /// Returns the arrival-side location index for any encoded node.
    /// For trips: the trip's end_location_idx.
    /// For depot sentinels: the depot's physical location (depots have one location).
    #[inline]
    pub fn end_location_of(&self, node: usize) -> usize {
        if node < self.n_trips {
            self.trips[node].end_location_idx
        } else {
            self.depot_location_indices[node - self.n_trips]
        }
    }

    /// Travel duration (s) between any two encoded nodes.
    /// Uses end_location_of(from_node) → start_location_of(to_node) so that
    /// deadhead costs correctly reflect a trip's arrival point and the next
    /// trip's departure point.
    /// Replaces: locations._adj[s_location][t_location].get('duration', 0)
    #[inline]
    pub fn duration_between(&self, from_node: usize, to_node: usize) -> f64 {
        let i = self.end_location_of(from_node);
        let j = self.start_location_of(to_node);
        self.duration_matrix[i * self.n_locations + j]
    }

    /// Travel distance (m) between any two encoded nodes.
    /// Uses end_location_of(from_node) → start_location_of(to_node).
    /// Replaces: locations._adj[s_location][t_location].get('distance', 0)
    #[inline]
    pub fn distance_between(&self, from_node: usize, to_node: usize) -> f64 {
        let i = self.end_location_of(from_node);
        let j = self.start_location_of(to_node);
        self.distance_matrix[i * self.n_locations + j]
    }
}

// Safety: NetworkData contains no interior mutability (no Cell, Mutex, etc.),
// so sharing a &NetworkData across threads is sound. Rust requires an explicit
// unsafe impl here because NetworkData contains raw Vecs that Rust cannot
// automatically verify are Sync in all cases.
unsafe impl Sync for NetworkData {}
unsafe impl Send for NetworkData {}

// -----------------------------------------------------------------------------
// Solution (the GA individual)
// -----------------------------------------------------------------------------

/// A single individual in the NSGA-II population.
///
/// Five gene arrays are the search variables:
///   successors[i]   — what node trip i hands off to (trip idx or single depot sentinel n_trips)
///   equipment[i]    — which vehicle type serves trip i
///   supply_type[i]  — which port type services the vehicle after trip i
///   start_depot[i]  — which depot the vehicle departs from for the tour containing trip i
///   end_depot[i]    — which depot the vehicle returns to after the tour containing trip i
///
/// All other fields are derived state, populated by solve() and evaluate().
/// Clone is derived so offspring can be produced without unsafe code.
#[derive(Clone, Debug)]
pub struct Solution {
    // --- Genes (the five co-optimized decision variables) ---

    /// successors[trip_idx] = successor node index.
    /// Values < n_trips: another trip. Values >= n_trips: single depot sentinel.
    pub successors: Vec<usize>,

    /// equipment[trip_idx] = index into NetworkData::vehicle_types.
    pub equipment: Vec<usize>,

    /// supply_type[trip_idx] = index into NetworkData::port_types.
    pub supply_type: Vec<usize>,

    /// start_depot[trip_idx] = index into NetworkData::depot_location_indices.
    /// Voting within each tour selects the plurality and writes back to all trips.
    pub start_depot: Vec<usize>,

    /// end_depot[trip_idx] = index into NetworkData::depot_location_indices.
    /// When same_depot=true, always equals start_depot after voting.
    pub end_depot: Vec<usize>,

    // --- Evaluation state ---

    /// True if the energy balance was solvable (no unresolvable tour split).
    pub feasible: bool,
    /// True if all constraints are satisfied.
    pub compliant: bool,
    /// Objective values in the order defined by NetworkData::objectives.
    pub fitness: Vec<f64>,
    /// Non-dominated front rank (0 = Pareto-optimal front).
    pub rank: usize,
    /// Number of generations this solution has survived.
    pub age: usize,

    // --- Solved state (populated by solve()) ---

    /// vehicles[depot_idx] = list of vehicle_type indices, one per vehicle instance.
    /// Length of inner Vec = number of physical vehicles at that depot.
    pub vehicles: Vec<Vec<usize>>,

    /// ports[depot_idx] = list of port_type indices, one per port instance.
    pub ports: Vec<Vec<usize>>,

    /// All scheduled tours for this solution.
    pub tours: Vec<Tour>,

    /// All charging/fueling events for this solution.
    pub supply_events: Vec<SupplyEvent>,
}

impl Solution {
    /// Allocate a blank Solution with correctly sized gene arrays.
    pub fn new(n_trips: usize, n_depots: usize, n_objectives: usize) -> Self {
        Solution {
            successors:   vec![0; n_trips],
            equipment:    vec![0; n_trips],
            supply_type:  vec![0; n_trips],
            start_depot:  vec![0; n_trips],
            end_depot:    vec![0; n_trips],
            feasible:     true,
            compliant:    true,
            fitness:      vec![f64::MAX; n_objectives],
            rank:         usize::MAX,
            age:          0,
            vehicles:     vec![Vec::new(); n_depots],
            ports:        vec![Vec::new(); n_depots],
            tours:        Vec::new(),
            supply_events: Vec::new(),
        }
    }

    /// Clear all derived (solved) state before re-running solve().
    /// Gene arrays (successors, equipment, supply_type, start_depot, end_depot) are preserved.
    pub fn clear_solved_state(&mut self) {
        self.feasible = true;
        self.compliant = true;
        for v in &mut self.vehicles {
            v.clear();
        }
        for p in &mut self.ports {
            p.clear();
        }
        self.tours.clear();
        self.supply_events.clear();
    }
}
