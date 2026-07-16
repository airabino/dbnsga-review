// =============================================================================
// solve.rs — Six-stage solve pipeline.
//
// Converts the five gene arrays (successors, equipment, supply_type,
// start_depot, end_depot) stored in a Solution into fully-scheduled tours,
// supply events, and port assignments.
//
// Stage order:
//   1. build_tours       — reconstruct trip chains from successor encoding
//   2. voting            — resolve depot/vehicle/port by plurality vote per tour
//   3. compute_tour_info — compute timing and geometry using voted depots
//   4. assign_vehicles   — assign physical vehicles; split energy-infeasible tours
//   5. build_supply_events — derive inter-tour charging/fueling events
//   6. assign_ports      — assign charging ports
//
// Mirrors Python's Solution.solve() → _tours / _voting / _tour_information /
// _vehicle_assignment / _supply_events / _port_assignment pipeline.
// =============================================================================

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use indexmap::IndexMap;

use crate::types::*;

// ---------------------------------------------------------------------------
// OrdF64 — total-order wrapper for f64 so it can live in BinaryHeap.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct OrdF64(pub f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the full six-stage solve pipeline on `solution`.
///
/// same_depot: if true, voting forces start_depot_idx == end_depot_idx per tour
///   (identical to Python single-depot behaviour). If false, start and end depots
///   are voted independently.
pub fn solve(data: &NetworkData, solution: &mut Solution, same_depot: bool) {
    solution.clear_solved_state();

    // Stage 1: reconstruct tours from successor encoding
    let mut tours = build_tours(data, solution);

    // Stage 2: vote on depot / vehicle / port type per tour
    // Must precede compute_tour_info because deadhead requires knowing the depot.
    voting(solution, &mut tours, same_depot);

    // Stage 3: compute timing and geometry using voted depot assignments
    for tour in &mut tours {
        compute_tour_info(data, tour);
    }

    // Stage 4: assign physical vehicles; split energy-infeasible tours
    assign_vehicles(data, solution, &mut tours);

    if !solution.feasible {
        solution.tours = tours;
        return;
    }

    // Stage 5: derive inter-tour supply events
    let supply_events = build_supply_events(&tours);

    // Stage 6: assign charging ports
    let supply_events = assign_ports(data, solution, supply_events);

    solution.tours = tours;
    solution.supply_events = supply_events;
}

// ---------------------------------------------------------------------------
// Stage 1 — build_tours
//
// Builds a predecessor map from solution.successors (normalising any value
// >= n_trips to the single sentinel slot n_trips), then walks backward from
// the sentinel to reconstruct each tour as a plain trip list:
//   [trip_A, ..., trip_Z]
// No depot sentinels appear in the output trip lists.
// ---------------------------------------------------------------------------

fn build_tours(data: &NetworkData, solution: &Solution) -> Vec<Tour> {
    let n_trips  = data.n_trips;
    let sentinel = n_trips; // single sentinel slot index

    // predecessors[node] = trip indices whose (normalised) successor is `node`
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n_trips + 1];
    for trip_idx in 0..n_trips {
        let s = solution.successors[trip_idx];
        // Normalise: any value >= n_trips (including old multi-depot sentinels)
        // maps to the canonical single sentinel slot.
        let mapped = if s >= n_trips { sentinel } else { s };
        predecessors[mapped].push(trip_idx);
    }

    let mut tours: Vec<Tour> = Vec::new();

    // Stack entries: (current_node, partial_in_reverse_order_with_trailing_sentinel)
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(sentinel, vec![sentinel])];

    while let Some((node, partial)) = stack.pop() {
        let preds = &predecessors[node];

        if preds.is_empty() {
            // `node` is a trip with no predecessor — it is the tour head.
            // partial = [node, trip_B, ..., trip_Z, sentinel]
            // Trim the trailing sentinel to get the plain trip list.
            let trips = partial[..partial.len() - 1].to_vec();
            tours.push(Tour { trips, ..Default::default() });
        } else {
            for &pred in preds {
                let mut new_partial = vec![pred];
                new_partial.extend_from_slice(&partial);
                stack.push((pred, new_partial));
            }
        }
    }

    tours
}

// ---------------------------------------------------------------------------
// Stage 2 — voting
//
// For each tour, votes on start_depot, end_depot, vehicle_type, and supply_type
// by plurality across the tour's trips, then writes back the winning values to
// both the solution gene arrays and the tour fields.
//
// Depot voting:
//   same_depot=true  → single vote using start_depot genes; end_depot_idx = start_depot_idx
//   same_depot=false → independent votes on start_depot and end_depot genes
//
// Supply type vote is filtered to only trips whose equipment gene equals the
// winning vehicle type (mirrors Python's _voting() behaviour).
//
// IndexMap is used for vote tallies to make tie-breaking deterministic
// (insertion order = trip sequence order).
// ---------------------------------------------------------------------------

fn voting(solution: &mut Solution, tours: &mut Vec<Tour>, same_depot: bool) {
    for tour in tours.iter_mut() {
        if tour.trips.is_empty() {
            continue;
        }

        // --- Vote on start depot ---
        let mut sd_count: IndexMap<usize, usize> = IndexMap::new();
        for &trip in &tour.trips {
            *sd_count.entry(solution.start_depot[trip]).or_insert(0) += 1;
        }
        let selected_sd = *sd_count.iter().max_by_key(|(_, &c)| c).unwrap().0;

        // --- Vote on end depot ---
        let selected_ed = if same_depot {
            selected_sd
        } else {
            let mut ed_count: IndexMap<usize, usize> = IndexMap::new();
            for &trip in &tour.trips {
                *ed_count.entry(solution.end_depot[trip]).or_insert(0) += 1;
            }
            *ed_count.iter().max_by_key(|(_, &c)| c).unwrap().0
        };

        // --- Vote on vehicle type ---
        let mut vt_count: IndexMap<usize, usize> = IndexMap::new();
        for &trip in &tour.trips {
            *vt_count.entry(solution.equipment[trip]).or_insert(0) += 1;
        }
        let selected_vt = *vt_count.iter().max_by_key(|(_, &c)| c).unwrap().0;

        // --- Vote on supply type (filtered to winning vehicle type) ---
        let mut pt_count: IndexMap<usize, usize> = IndexMap::new();
        for &trip in &tour.trips {
            if solution.equipment[trip] == selected_vt {
                *pt_count.entry(solution.supply_type[trip]).or_insert(0) += 1;
            }
        }
        let selected_pt = *pt_count.iter().max_by_key(|(_, &c)| c).unwrap().0;

        // --- Write back to gene arrays and tour fields ---
        for &trip in &tour.trips {
            solution.start_depot[trip] = selected_sd;
            solution.end_depot[trip]   = selected_ed;
            solution.equipment[trip]   = selected_vt;
            solution.supply_type[trip] = selected_pt;
        }

        tour.start_depot_idx  = selected_sd;
        tour.end_depot_idx    = selected_ed;
        tour.vehicle_type_idx = selected_vt;
        tour.port_type_idx    = selected_pt;
    }
}

// ---------------------------------------------------------------------------
// Stage 3 — compute_tour_info (single tour)
//
// Fills tour.start, .finish, .distance, .duration using the voted depot
// assignments already stored in tour.start_depot_idx / tour.end_depot_idx.
//
// tour.trips must be [trip_A, ..., trip_Z] (no sentinel brackets).
// voting() must have run before this function.
// ---------------------------------------------------------------------------

pub(crate) fn compute_tour_info(data: &NetworkData, tour: &mut Tour) {
    let n = tour.trips.len();
    debug_assert!(n >= 1, "tour must have at least 1 trip");

    let n_loc            = data.n_locations;
    let start_depot_loc  = data.depot_location_indices[tour.start_depot_idx];
    let end_depot_loc    = data.depot_location_indices[tour.end_depot_idx];
    let first_trip_idx   = tour.trips[0];
    let last_trip_idx    = tour.trips[n - 1];

    // ---- Outbound deadhead: start depot → first trip's departure location ----
    let first_start_loc = data.trips[first_trip_idx].start_location_idx;
    let dh_out_dur  = data.duration_matrix[start_depot_loc * n_loc + first_start_loc];
    let dh_out_dist = data.distance_matrix[start_depot_loc * n_loc + first_start_loc];

    let tour_start = data.trips[first_trip_idx].start - dh_out_dur;

    let mut duration = dh_out_dur + data.trips[first_trip_idx].duration;
    let mut distance = dh_out_dist + data.trips[first_trip_idx].distance;

    // ---- Inter-trip deadheads + subsequent trips ----
    // windows(2) yields (trips[0], trips[1]), (trips[1], trips[2]), …
    // Each window contributes the deadhead from s's arrival to t's departure,
    // plus trip t's own duration/distance. Trip [0] is already counted above.
    for window in tour.trips.windows(2) {
        let (s, t) = (window[0], window[1]);
        let s_end = data.trips[s].end_location_idx;
        let t_start = data.trips[t].start_location_idx;
        duration += data.duration_matrix[s_end * n_loc + t_start];
        distance += data.distance_matrix[s_end * n_loc + t_start];
        duration += data.trips[t].duration;
        distance += data.trips[t].distance;
    }

    // ---- Inbound deadhead: last trip's arrival location → end depot ----
    let last_end_loc = data.trips[last_trip_idx].end_location_idx;
    duration += data.duration_matrix[last_end_loc * n_loc + end_depot_loc];
    distance += data.distance_matrix[last_end_loc * n_loc + end_depot_loc];

    tour.start    = tour_start;
    tour.finish   = tour_start + duration;
    tour.duration = duration;
    tour.distance = distance;
    // start_depot_idx and end_depot_idx already set by voting()
}

// ---------------------------------------------------------------------------
// Stage 4 — assign_vehicles
//
// Processes tours in departure-time order. For each tour:
//   1. Draws a vehicle from vehicle_heaps[start_depot_idx][vt_idx].
//   2. Checks energy feasibility (energy ≤ capacity).
//   3. If infeasible: splits the tour at the midpoint and re-queues both
//      halves. Split halves inherit start_depot_idx, end_depot_idx, vt/pt.
//      Single-trip infeasible tours set solution.feasible = false.
//   4. If feasible: returns the vehicle to vehicle_heaps[end_depot_idx][vt_idx]
//      at tour.finish + min_resupply.
//
// Modifies solution.vehicles[start_depot_idx] and solution.successors (for splits).
// Replaces `tours` contents with only the completed (feasible) tours.
// ---------------------------------------------------------------------------

fn assign_vehicles(data: &NetworkData, solution: &mut Solution, tours: &mut Vec<Tour>) {
    if tours.is_empty() {
        return;
    }

    // vehicle_heaps[depot_idx][vt_idx] = min-heap of (return_time, vehicle_instance_idx)
    let mut vehicle_heaps: Vec<Vec<BinaryHeap<Reverse<(OrdF64, usize)>>>> = (0..data.n_depots)
        .map(|_| (0..data.n_vehicle_types).map(|_| BinaryHeap::new()).collect())
        .collect();

    let mut vehicle_counter = 0usize;

    let mut working: Vec<Tour> = std::mem::take(tours);
    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();
    for (i, tour) in working.iter().enumerate() {
        heap.push(Reverse((OrdF64(tour.start), i)));
    }

    let mut completed: Vec<Tour> = Vec::new();

    while let Some(Reverse((_, idx))) = heap.pop() {
        let vt_idx          = working[idx].vehicle_type_idx;
        let pt_idx          = working[idx].port_type_idx;
        let start_depot_idx = working[idx].start_depot_idx;
        let end_depot_idx   = working[idx].end_depot_idx;
        let depart          = working[idx].start;

        let vh = &mut vehicle_heaps[start_depot_idx][vt_idx];

        // Claim an available vehicle or create a new one
        let vid = if let Some(Reverse((OrdF64(avail), vid))) = vh.peek().copied() {
            if avail <= depart {
                vh.pop();
                vid
            } else {
                let vid = vehicle_counter;
                vehicle_counter += 1;
                solution.vehicles[start_depot_idx].push(vt_idx);
                vid
            }
        } else {
            let vid = vehicle_counter;
            vehicle_counter += 1;
            solution.vehicles[start_depot_idx].push(vt_idx);
            vid
        };

        let energy = data.vehicle_types[vt_idx].energy(working[idx].distance);

        if energy > data.vehicle_types[vt_idx].capacity {
            // Tour is energy-infeasible — split at midpoint.
            let n_trips_in_tour = working[idx].trips.len();

            if n_trips_in_tour <= 1 {
                // Cannot split a single-trip tour.
                solution.feasible = false;
                *tours = completed;
                return;
            }

            let split_idx = n_trips_in_tour / 2;

            // Mark the split in the successor gene so the gene stays consistent.
            let prev_trip = working[idx].trips[split_idx - 1];
            solution.successors[prev_trip] = data.n_trips; // single depot sentinel

            let first_half  = working[idx].trips[..split_idx].to_vec();
            let second_half = working[idx].trips[split_idx..].to_vec();

            // Both halves inherit the parent's voted depot and vehicle/port type.
            let mut half_a = Tour {
                trips:            first_half,
                vehicle_type_idx: vt_idx,
                port_type_idx:    pt_idx,
                start_depot_idx,
                end_depot_idx,
                ..Default::default()
            };
            let mut half_b = Tour {
                trips:            second_half,
                vehicle_type_idx: vt_idx,
                port_type_idx:    pt_idx,
                start_depot_idx,
                end_depot_idx,
                ..Default::default()
            };

            compute_tour_info(data, &mut half_a);
            compute_tour_info(data, &mut half_b);

            let ia = working.len();
            working.push(half_a);
            let ib = working.len();
            working.push(half_b);

            heap.push(Reverse((OrdF64(working[ia].start), ia)));
            heap.push(Reverse((OrdF64(working[ib].start), ib)));

            // Return vehicle to heap — available again at original departure time.
            vehicle_heaps[start_depot_idx][vt_idx].push(Reverse((OrdF64(depart), vid)));

            continue;
        }

        // Feasible: record energy and vehicle assignment.
        working[idx].energy               = energy;
        working[idx].vehicle_instance_idx = vid;

        // Vehicle returns to end_depot at tour.finish + min resupply time.
        let resupply    = data.port_types[pt_idx]
            .resupply_time(data.vehicle_types[vt_idx].resupply_rate, energy);
        let return_time = working[idx].finish + resupply;

        vehicle_heaps[end_depot_idx][vt_idx].push(Reverse((OrdF64(return_time), vid)));

        completed.push(working[idx].clone());
    }

    *tours = completed;
}

// ---------------------------------------------------------------------------
// Stage 5 — build_supply_events
//
// Groups completed tours by vehicle_instance_idx. For each vehicle, generates
// one supply event per inter-tour gap and one final event after the last tour
// (with latest_finish = 24 h + schedule_start).
//
// The event depot is tour.end_depot_idx — the vehicle charges at the depot it
// returned to at the end of the tour.
// ---------------------------------------------------------------------------

fn build_supply_events(tours: &[Tour]) -> Vec<SupplyEvent> {
    if tours.is_empty() {
        return Vec::new();
    }

    let schedule_start = tours.iter().map(|t| t.start).fold(f64::INFINITY, f64::min);
    let day_end        = 24.0 * 3600.0 + schedule_start;

    let mut by_vehicle: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, tour) in tours.iter().enumerate() {
        by_vehicle.entry(tour.vehicle_instance_idx).or_default().push(i);
    }
    for indices in by_vehicle.values_mut() {
        indices.sort_by(|&a, &b| OrdF64(tours[a].start).cmp(&OrdF64(tours[b].start)));
    }

    let mut events: Vec<SupplyEvent> = Vec::new();

    let mut by_vehicle_sorted: Vec<(usize, Vec<usize>)> = by_vehicle.into_iter().collect();
    by_vehicle_sorted.sort_by_key(|(vid, _)| *vid);

    for (vehicle_idx, indices) in &by_vehicle_sorted {
        let n = indices.len();
        for i in 0..n {
            let tour          = &tours[indices[i]];
            let latest_finish = if i + 1 < n {
                tours[indices[i + 1]].start
            } else {
                day_end
            };

            events.push(SupplyEvent {
                vehicle_instance_idx: *vehicle_idx,
                vehicle_type_idx:     tour.vehicle_type_idx,
                port_type_idx:        tour.port_type_idx,
                depot_idx:            tour.end_depot_idx, // charge at end depot
                earliest_start:       tour.finish,
                latest_finish,
                energy:               tour.energy,
                ..Default::default()
            });
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Stage 6 — assign_ports
//
// Processes supply events in latest_finish order (most urgent first).
// Reuses an idle port of the correct type if it can finish in time; otherwise
// creates a new port instance.
// ---------------------------------------------------------------------------

fn assign_ports(
    data:     &NetworkData,
    solution: &mut Solution,
    mut events: Vec<SupplyEvent>,
) -> Vec<SupplyEvent> {
    if events.is_empty() {
        return events;
    }

    // port_heaps[depot_idx][pt_idx] = min-heap of (finish_time, port_instance_idx)
    let mut port_heaps: Vec<Vec<BinaryHeap<Reverse<(OrdF64, usize)>>>> = (0..data.n_depots)
        .map(|_| (0..data.n_port_types).map(|_| BinaryHeap::new()).collect())
        .collect();

    let mut port_counter = 0usize;

    let mut heap: BinaryHeap<Reverse<(OrdF64, usize)>> = BinaryHeap::new();
    for (i, event) in events.iter().enumerate() {
        heap.push(Reverse((OrdF64(event.latest_finish), i)));
    }

    while let Some(Reverse((_, ei))) = heap.pop() {
        let pt_idx         = events[ei].port_type_idx;
        let vt_idx         = events[ei].vehicle_type_idx;
        let depot_idx      = events[ei].depot_idx;
        let latest_finish  = events[ei].latest_finish;
        let earliest_start = events[ei].earliest_start;
        let energy         = events[ei].energy;
        let vt_rate        = data.vehicle_types[vt_idx].resupply_rate;
        let pt             = &data.port_types[pt_idx];

        let ph = &mut port_heaps[depot_idx][pt_idx];

        let (start_time, finish_time, pid) = if let Some(Reverse((OrdF64(avail), pid))) =
            ph.peek().copied()
        {
            let resupply    = pt.resupply_time(vt_rate, energy);
            let cand_finish = avail + resupply;

            if cand_finish <= latest_finish {
                ph.pop();
                let event_start = avail.max(earliest_start);
                (event_start, event_start + resupply, pid)
            } else {
                let pid = port_counter;
                port_counter += 1;
                solution.ports[depot_idx].push(pt_idx);
                let resupply = pt.resupply_time(vt_rate, energy);
                (earliest_start, earliest_start + resupply, pid)
            }
        } else {
            let pid = port_counter;
            port_counter += 1;
            solution.ports[depot_idx].push(pt_idx);
            let resupply = pt.resupply_time(vt_rate, energy);
            (earliest_start, earliest_start + resupply, pid)
        };

        port_heaps[depot_idx][pt_idx].push(Reverse((OrdF64(finish_time), pid)));

        events[ei].start             = start_time;
        events[ei].finish            = finish_time;
        events[ei].port_instance_idx = pid;
    }

    events
}
