// =============================================================================
// network.rs — PyO3-exposed RustNetwork class.
//
// RustNetwork is the single object Python interacts with. Construction
// converts Python numpy arrays into the flat Rust NetworkData struct.
// optimize() (Step 5) will run the full NSGA-II loop entirely in Rust.
// =============================================================================

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use std::collections::HashMap;

use crate::types::*;
use crate::solve;
use crate::evaluate;
use crate::nsga2;

use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Python-facing wrapper around the immutable fleet optimization problem.
///
/// Python usage (after Step 3):
///   rust_net, index_maps = build_rust_network(network)
///   print(rust_net.status())
///
/// Python usage (after Step 5):
///   population, generations = rust_net.optimize(max_iter=300, population_size=150, ...)
#[pyclass]
pub struct RustNetwork {
    pub(crate) data: NetworkData,
}

#[pymethods]
impl RustNetwork {

    /// Build a RustNetwork from pre-computed numpy arrays.
    ///
    /// All NetworkX graph data must be pre-extracted on the Python side
    /// (see src/rust_bridge.py: build_rust_network). This constructor
    /// crosses the PyO3 boundary exactly once and owns all data thereafter.
    ///
    /// Parameters
    /// ----------
    /// Trip attributes (one f64 per trip, indexed by integer trip index):
    ///   trip_starts, trip_finishes, trip_distances, trip_durations,
    ///   trip_location_indices (i64 — index into location array)
    ///
    /// Trip adjacency (parallel edge arrays of i64 trip indices):
    ///   adj_sources, adj_targets
    ///
    /// Location matrices (n_locations × n_locations, row-major f64):
    ///   duration_matrix, distance_matrix
    ///
    /// depot_location_indices: list[int] — location indices that are depots
    ///
    /// Vehicle type parameters (one element per type, all f64 unless noted):
    ///   vt_capacities, vt_resupply_rates, vt_consumptions, vt_emissions,
    ///   vt_fixed_costs, vt_unit_costs, vt_annual_costs, vt_disposal_costs,
    ///   vt_service_periods (i64), vt_operational_costs,
    ///   vt_supply_type_indices: list[list[int]]
    ///
    /// Port type parameters (one element per type, all f64 unless noted):
    ///   pt_resupply_rates, pt_operating_times, pt_efficiencies,
    ///   pt_fixed_costs, pt_unit_costs, pt_annual_costs, pt_disposal_costs,
    ///   pt_service_periods (i64), pt_operational_costs
    ///
    /// Objective configuration:
    ///   objective_kinds: list[str]  — e.g. ["InitialCost", "DailyCost"]
    ///   objective_params: list[dict[str, float]]  — per-objective parameters
    ///
    /// Constraint configuration:
    ///   fleet_portion_included: list[int]  — vehicle type indices
    ///   fleet_portion_min: float
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // --- Trip attributes ---
        trip_starts:           PyReadonlyArray1<f64>,
        trip_finishes:         PyReadonlyArray1<f64>,
        trip_distances:        PyReadonlyArray1<f64>,
        trip_durations:        PyReadonlyArray1<f64>,
        trip_start_location_indices: PyReadonlyArray1<i64>,
        trip_end_location_indices:   PyReadonlyArray1<i64>,
        // --- Trip adjacency ---
        adj_sources: PyReadonlyArray1<i64>,
        adj_targets: PyReadonlyArray1<i64>,
        // --- Location matrices ---
        duration_matrix: PyReadonlyArray2<f64>,
        distance_matrix: PyReadonlyArray2<f64>,
        // --- Depot location indices and sampling weights ---
        depot_location_indices: Vec<usize>,
        depot_probabilities:    PyReadonlyArray1<f64>,
        // --- Vehicle type parameters ---
        vt_capacities:          PyReadonlyArray1<f64>,
        vt_resupply_rates:      PyReadonlyArray1<f64>,
        vt_consumptions:        PyReadonlyArray1<f64>,
        vt_emissions:           PyReadonlyArray1<f64>,
        vt_fixed_costs:         PyReadonlyArray1<f64>,
        vt_unit_costs:          PyReadonlyArray1<f64>,
        vt_annual_costs:        PyReadonlyArray1<f64>,
        vt_disposal_costs:      PyReadonlyArray1<f64>,
        vt_service_periods:     PyReadonlyArray1<i64>,
        vt_operational_costs:   PyReadonlyArray1<f64>,
        vt_supply_type_indices: Vec<Vec<usize>>,
        vt_probabilities:       PyReadonlyArray1<f64>,
        // --- Port type parameters ---
        pt_resupply_rates:    PyReadonlyArray1<f64>,
        pt_operating_times:   PyReadonlyArray1<f64>,
        pt_efficiencies:      PyReadonlyArray1<f64>,
        pt_fixed_costs:       PyReadonlyArray1<f64>,
        pt_unit_costs:        PyReadonlyArray1<f64>,
        pt_annual_costs:      PyReadonlyArray1<f64>,
        pt_disposal_costs:    PyReadonlyArray1<f64>,
        pt_service_periods:   PyReadonlyArray1<i64>,
        pt_operational_costs: PyReadonlyArray1<f64>,
        pt_emissions:         PyReadonlyArray1<f64>,
        // --- Objective and constraint configuration ---
        objective_kinds:  Vec<String>,
        objective_params: Vec<HashMap<String, f64>>,
        objective_names:  Vec<String>,
        fleet_portion_included: Vec<usize>,
        fleet_portion_min: f64,
        port_limits: Vec<usize>,
        lot_size_limits: Vec<usize>,
    ) -> PyResult<Self> {

        // -----------------------------------------------------------------
        // Trips
        // -----------------------------------------------------------------
        let starts      = trip_starts.as_slice()?;
        let finishes    = trip_finishes.as_slice()?;
        let dists       = trip_distances.as_slice()?;
        let durs        = trip_durations.as_slice()?;
        let start_locs  = trip_start_location_indices.as_slice()?;
        let end_locs    = trip_end_location_indices.as_slice()?;

        let n_trips = starts.len();

        let trips: Vec<Trip> = (0..n_trips)
            .map(|i| Trip {
                start_location_idx: start_locs[i] as usize,
                end_location_idx:   end_locs[i] as usize,
                distance:           dists[i],
                duration:           durs[i],
                start:              starts[i],
                finish:             finishes[i],
            })
            .collect();

        // -----------------------------------------------------------------
        // Trip adjacency list
        // Build Vec<Vec<usize>> from two parallel edge-index arrays.
        // trip_adj[i] = all valid successor trip indices for trip i.
        // -----------------------------------------------------------------
        let srcs = adj_sources.as_slice()?;
        let tgts = adj_targets.as_slice()?;

        let mut trip_adj: Vec<Vec<usize>> = vec![Vec::new(); n_trips];
        for (&s, &t) in srcs.iter().zip(tgts.iter()) {
            trip_adj[s as usize].push(t as usize);
        }

        // -----------------------------------------------------------------
        // Location matrices — flatten row-major 2D numpy arrays to Vec<f64>
        // Access pattern: matrix[i * n_locations + j]
        // -----------------------------------------------------------------
        let n_locations = duration_matrix.shape()[0];
        let duration_matrix_flat = duration_matrix.as_slice()?.to_vec();
        let distance_matrix_flat = distance_matrix.as_slice()?.to_vec();

        let n_depots = depot_location_indices.len();
        let depot_probabilities: Vec<f64> = depot_probabilities.as_slice()?.to_vec();

        // -----------------------------------------------------------------
        // Vehicle types
        // -----------------------------------------------------------------
        let vc  = vt_capacities.as_slice()?;
        let vr  = vt_resupply_rates.as_slice()?;
        let vco = vt_consumptions.as_slice()?;
        let ve  = vt_emissions.as_slice()?;
        let vfc = vt_fixed_costs.as_slice()?;
        let vuc = vt_unit_costs.as_slice()?;
        let vac = vt_annual_costs.as_slice()?;
        let vdc = vt_disposal_costs.as_slice()?;
        let vsp = vt_service_periods.as_slice()?;
        let voc = vt_operational_costs.as_slice()?;

        let n_vehicle_types = vc.len();

        let vehicle_types: Vec<VehicleType> = (0..n_vehicle_types)
            .map(|i| VehicleType {
                capacity:             vc[i],
                resupply_rate:        vr[i],
                consumption:          vco[i],
                emissions:            ve[i],
                fixed_cost:           vfc[i],
                unit_cost:            vuc[i],
                annual_cost:          vac[i],
                disposal_cost:        vdc[i],
                service_periods:      vsp[i] as usize,
                operational_cost:     voc[i],
                supply_type_indices:  vt_supply_type_indices[i].clone(),
            })
            .collect();

        let vt_probabilities: Vec<f64> = vt_probabilities.as_slice()?.to_vec();

        // -----------------------------------------------------------------
        // Port types
        // -----------------------------------------------------------------
        let pr  = pt_resupply_rates.as_slice()?;
        let pot = pt_operating_times.as_slice()?;
        let pe  = pt_efficiencies.as_slice()?;
        let pfc = pt_fixed_costs.as_slice()?;
        let puc = pt_unit_costs.as_slice()?;
        let pac = pt_annual_costs.as_slice()?;
        let pdc = pt_disposal_costs.as_slice()?;
        let psp = pt_service_periods.as_slice()?;
        let poc = pt_operational_costs.as_slice()?;
        let pem = pt_emissions.as_slice()?;

        let n_port_types = pr.len();

        let port_types: Vec<PortType> = (0..n_port_types)
            .map(|i| PortType {
                resupply_rate:    pr[i],
                operating_time:   pot[i],
                efficiency:       pe[i],
                fixed_cost:       pfc[i],
                unit_cost:        puc[i],
                annual_cost:      pac[i],
                disposal_cost:    pdc[i],
                service_periods:  psp[i] as usize,
                operational_cost: poc[i],
                emissions:        pem[i],
            })
            .collect();

        // -----------------------------------------------------------------
        // Objectives
        // -----------------------------------------------------------------
        let objectives: Vec<ObjectiveKind> = objective_kinds
            .iter()
            .zip(objective_params.iter())
            .map(|(kind, params)| parse_objective(kind, params))
            .collect::<PyResult<Vec<_>>>()?;

        // -----------------------------------------------------------------
        // Constraints
        // -----------------------------------------------------------------
        let constraints = ConstraintConfig {
            fleet_portion_included,
            fleet_portion_min,
            port_limits,
            lot_size_limits,
        };

        // -----------------------------------------------------------------
        // Assemble NetworkData
        // -----------------------------------------------------------------
        let data = NetworkData {
            trips,
            n_trips,
            trip_adj,
            duration_matrix: duration_matrix_flat,
            distance_matrix: distance_matrix_flat,
            n_locations,
            depot_location_indices,
            n_depots,
            depot_probabilities,
            vehicle_types,
            n_vehicle_types,
            vt_probabilities,
            port_types,
            n_port_types,
            objectives,
            objective_names,
            constraints,
        };

        Ok(RustNetwork { data })
    }

    // -----------------------------------------------------------------
    // Introspection helpers (used for verification after construction)
    // -----------------------------------------------------------------

    pub fn n_trips(&self) -> usize        { self.data.n_trips }
    pub fn n_locations(&self) -> usize    { self.data.n_locations }
    pub fn n_vehicle_types(&self) -> usize { self.data.n_vehicle_types }
    pub fn n_port_types(&self) -> usize   { self.data.n_port_types }
    pub fn n_depots(&self) -> usize       { self.data.n_depots }
    pub fn n_objectives(&self) -> usize   { self.data.objectives.len() }

    pub fn status(&self) -> String {
        format!(
            "fleet_opt_core.RustNetwork: {} trips | {} locations | {} depots | \
             {} vehicle types | {} port types | {} objectives",
            self.data.n_trips,
            self.data.n_locations,
            self.data.n_depots,
            self.data.n_vehicle_types,
            self.data.n_port_types,
            self.data.objectives.len(),
        )
    }

    /// Lookup using the successor encoding (trip node or depot sentinel).
    /// Values < n_trips are trip indices; values >= n_trips are depot sentinels.
    pub fn duration_between(&self, from_node: usize, to_node: usize) -> f64 {
        self.data.duration_between(from_node, to_node)
    }

    pub fn distance_between(&self, from_node: usize, to_node: usize) -> f64 {
        self.data.distance_between(from_node, to_node)
    }

    /// Lookup using raw location indices (0..n_locations). Used for
    /// cross-validation against Python's locations._adj matrix.
    pub fn location_duration(&self, from_loc: usize, to_loc: usize) -> f64 {
        self.data.duration_matrix[from_loc * self.data.n_locations + to_loc]
    }

    pub fn location_distance(&self, from_loc: usize, to_loc: usize) -> f64 {
        self.data.distance_matrix[from_loc * self.data.n_locations + to_loc]
    }

    /// Return the valid successor trip indices for a given trip index.
    pub fn trip_successors(&self, trip_idx: usize) -> Vec<usize> {
        self.data.trip_adj[trip_idx].clone()
    }

    // -----------------------------------------------------------------
    // solve_solution() — Step 4 cross-validation helper.
    //
    // Accepts three flat gene arrays (one entry per trip, integer indices),
    // runs the full six-stage solve pipeline, and returns a dict with the
    // solved state for comparison against Python's Solution.solve().
    //
    // Parameters
    // ----------
    // successors  : list[int]  — successor node per trip (trip idx or depot sentinel)
    // equipment   : list[int]  — vehicle type index per trip
    // supply_type : list[int]  — port type index per trip
    //
    // Returns
    // -------
    // dict with keys:
    //   feasible       : bool
    //   compliant      : bool
    //   fitness        : list[float]
    //   n_tours        : int
    //   n_supply_events: int
    //   n_vehicles     : list[int]  — vehicle count per depot
    //   n_ports        : list[int]  — port count per depot
    //   tour_starts    : list[float]
    //   tour_finishes  : list[float]
    //   tour_distances : list[float]
    //   tour_energies  : list[float]
    // -----------------------------------------------------------------
    pub fn solve_solution(
        &self,
        successors:  Vec<usize>,
        equipment:   Vec<usize>,
        supply_type: Vec<usize>,
    ) -> PyResult<HashMap<String, PyObject>> {
        let data = &self.data;

        let mut sol = Solution::new(
            data.n_trips,
            data.n_depots,
            data.objectives.len(),
        );
        sol.successors  = successors;
        sol.equipment   = equipment;
        sol.supply_type = supply_type;

        // same_depot=true mirrors Python's single-depot behaviour for cross-validation.
        solve::solve(data, &mut sol, true);
        evaluate::evaluate(data, &mut sol);

        Python::with_gil(|py| {
            let mut out: HashMap<String, PyObject> = HashMap::new();

            out.insert("feasible".into(),        sol.feasible.to_object(py));
            out.insert("compliant".into(),       sol.compliant.to_object(py));
            out.insert("fitness".into(),         sol.fitness.clone().to_object(py));
            out.insert("n_tours".into(),         sol.tours.len().to_object(py));
            out.insert("n_supply_events".into(), sol.supply_events.len().to_object(py));

            let n_vehicles: Vec<usize> = sol.vehicles.iter().map(|v| v.len()).collect();
            let n_ports:    Vec<usize> = sol.ports.iter().map(|p| p.len()).collect();
            out.insert("n_vehicles".into(), n_vehicles.to_object(py));
            out.insert("n_ports".into(),    n_ports.to_object(py));

            let tour_starts:    Vec<f64> = sol.tours.iter().map(|t| t.start).collect();
            let tour_finishes:  Vec<f64> = sol.tours.iter().map(|t| t.finish).collect();
            let tour_distances: Vec<f64> = sol.tours.iter().map(|t| t.distance).collect();
            let tour_energies:  Vec<f64> = sol.tours.iter().map(|t| t.energy).collect();

            out.insert("tour_starts".into(),    tour_starts.to_object(py));
            out.insert("tour_finishes".into(),  tour_finishes.to_object(py));
            out.insert("tour_distances".into(), tour_distances.to_object(py));
            out.insert("tour_energies".into(),  tour_energies.to_object(py));

            Ok(out)
        })
    }

    // -----------------------------------------------------------------
    // optimize() — full NSGA-II loop.
    //
    // Parameters
    // ----------
    // max_iter              : int   — number of generations
    // population_size       : int   — individuals per generation
    // mutation_probability  : float — per-gene mutation rate
    // crossover_probability : float — per-gene crossover rate
    // seed                  : int   — RNG seed (u64)
    //
    // Returns
    // -------
    // tuple of:
    //   population : list[dict] — one dict per individual in the final population
    //   generations: dict[int, dict[int, dict]] — per-generation statistics
    //     keyed by generation number (0 = initial), each value is a dict
    //     keyed by individual index, each value has:
    //       rank      : int
    //       age       : int
    //       fitness   : dict[str, float]  — objective name → value
    //       compliant : bool
    // -----------------------------------------------------------------
    #[pyo3(signature = (max_iter, population_size, mutation_probability, crossover_probability, seed, survival_threshold=0, retries=0, initial_population_size=0, corner_depth=0, min_iter=0, same_depot=true, store_interval=1))]
    pub fn optimize(
        &self,
        py:                    Python<'_>,
        max_iter:              usize,
        population_size:       usize,
        mutation_probability:  f64,
        crossover_probability: f64,
        seed:                  u64,
        survival_threshold:    usize,
        retries:               usize,
        initial_population_size: usize,
        corner_depth:          usize,
        min_iter:              usize,
        same_depot:            bool,
        store_interval:        usize,
    ) -> PyResult<(Vec<HashMap<String, PyObject>>, HashMap<usize, HashMap<usize, HashMap<String, PyObject>>>)> {
        let data = &self.data;

        let init_size = if initial_population_size == 0 {
            population_size
        } else {
            initial_population_size
        };

        // Release the GIL for the duration of the optimization so Jupyter's
        // internal Python threads (heartbeat, IOPub) can run while Rust works.
        let result = py.allow_threads(|| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let pool = rayon::ThreadPoolBuilder::new()
                .build()
                .expect("failed to build Rayon thread pool");
            pool.install(|| {
                nsga2::optimize(
                    data,
                    max_iter,
                    min_iter,
                    init_size,
                    population_size,
                    mutation_probability,
                    crossover_probability,
                    survival_threshold,
                    retries,
                    corner_depth,
                    same_depot,
                    store_interval,
                    &mut rng,
                )
            })
        });

        Python::with_gil(|py| {
            // --- Serialize final population ---
            let py_population: Vec<HashMap<String, PyObject>> = result.population.into_iter().map(|sol| {
                let mut d: HashMap<String, PyObject> = HashMap::new();
                d.insert("successors".into(),   sol.successors.to_object(py));
                d.insert("equipment".into(),    sol.equipment.to_object(py));
                d.insert("supply_type".into(),  sol.supply_type.to_object(py));
                d.insert("start_depot".into(),  sol.start_depot.to_object(py));
                d.insert("end_depot".into(),    sol.end_depot.to_object(py));
                d.insert("fitness".into(),      sol.fitness.to_object(py));
                d.insert("rank".into(),         sol.rank.to_object(py));
                d.insert("age".into(),          sol.age.to_object(py));
                d.insert("feasible".into(),     sol.feasible.to_object(py));
                d.insert("compliant".into(),    sol.compliant.to_object(py));
                d
            }).collect();

            // --- Serialize generation history ---
            // Mirrors Python: generations = {gen_idx: {sol_idx: {rank, age, fitness, compliant}}}
            // fitness is keyed by objective name (e.g. "Capital Cost") matching Python's Solution.statistics()
            let obj_names = &data.objective_names;

            let mut py_generations: HashMap<usize, HashMap<usize, HashMap<String, PyObject>>> = HashMap::new();

            for (gen_idx, gen_stats) in result.generation_stats.iter().enumerate() {
                let mut gen_dict: HashMap<usize, HashMap<String, PyObject>> = HashMap::new();

                for (sol_idx, sol_stats) in gen_stats.stats.iter().enumerate() {
                    let mut sol_dict: HashMap<String, PyObject> = HashMap::new();
                    sol_dict.insert("rank".into(),      sol_stats.rank.to_object(py));
                    sol_dict.insert("age".into(),       sol_stats.age.to_object(py));
                    sol_dict.insert("compliant".into(), sol_stats.compliant.to_object(py));

                    // Build named fitness dict: {"Capital Cost": 98.5e6, ...}
                    let mut fitness_dict: HashMap<String, f64> = HashMap::new();
                    for (i, val) in sol_stats.fitness.iter().enumerate() {
                        let name = if i < obj_names.len() {
                            obj_names[i].clone()
                        } else {
                            format!("objective_{}", i)
                        };
                        fitness_dict.insert(name, *val);
                    }
                    sol_dict.insert("fitness".into(), fitness_dict.to_object(py));

                    gen_dict.insert(sol_idx, sol_dict);
                }

                py_generations.insert(gen_idx, gen_dict);
            }

            Ok((py_population, py_generations))
        })
    }
}

// -----------------------------------------------------------------------------
// Objective kind parser
// -----------------------------------------------------------------------------

fn parse_objective(
    kind: &str,
    params: &HashMap<String, f64>,
) -> PyResult<ObjectiveKind> {
    match kind {
        "InitialCost" => Ok(ObjectiveKind::InitialCost),
        "DailyCost"   => Ok(ObjectiveKind::DailyCost {
            emissions_cost: params.get("emissions_cost").copied().unwrap_or(0.0),
        }),
        "Capital"     => Ok(ObjectiveKind::Capital {
            discount_rate: params.get("discount_rate").copied().unwrap_or(0.0),
        }),
        "Operational" => Ok(ObjectiveKind::Operational {
            schedules_per_period: params.get("schedules_per_period").copied().unwrap_or(1.0),
            discount_rate:        params.get("discount_rate").copied().unwrap_or(0.0),
        }),
        "Emissions" => Ok(ObjectiveKind::Emissions),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown objective kind '{}'. Valid kinds: \
             InitialCost, DailyCost, Capital, Operational, Emissions",
            other
        ))),
    }
}
