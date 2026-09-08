// =============================================================================
// evaluate.rs — Objective evaluation and constraint checking.
//
// Each function takes an immutable &NetworkData and &Solution and returns
// a scalar f64 cost (or bool for constraints).  They mirror the Python
// objective classes in src/optimization/objective.py.
//
// NPV formula (matching Python):
//   cost += initial_cost          (period 0, no discounting)
//   cost += annual / (1+r)^p     for p in 1..service_periods-1
//   cost += disposal / (1+r)^sp  appended at period sp
// =============================================================================

use crate::types::*;

// ---------------------------------------------------------------------------
// Public dispatch — evaluate all objectives and one constraint
// ---------------------------------------------------------------------------

/// Populate solution.fitness and solution.compliant from NetworkData.objectives
/// and NetworkData.constraints.  solution.tours and supply_events must already
/// be populated by solve().
pub fn evaluate(data: &NetworkData, solution: &mut Solution) {
    solution.fitness.clear();

    for obj in &data.objectives {
        let value = match obj {
            ObjectiveKind::InitialCost => initial_cost(data, solution),
            ObjectiveKind::DailyCost { emissions_cost: ec } => daily_cost(data, solution, *ec),
            ObjectiveKind::Capital { discount_rate } => {
                capital_cost(data, solution, *discount_rate)
            }
            ObjectiveKind::Operational { schedules_per_period, discount_rate } => {
                operational_cost(data, solution, *schedules_per_period, *discount_rate)
            }
            ObjectiveKind::Emissions => {
                emissions_cost(data, solution)
            }
            ObjectiveKind::FleetPortion { included } => {
                fleet_portion_cost(solution, included)
            }
        };
        solution.fitness.push(value);
    }

    solution.compliant = fleet_portion_compliant(data, solution)
        && port_limit_compliant(data, solution)
        && lot_size_limit_compliant(data, solution);
}

// ---------------------------------------------------------------------------
// InitialCost — unit_cost per vehicle + unit_cost per port (no amortisation)
// Mirrors Python's Initial_Cost.evaluate_solution()
// ---------------------------------------------------------------------------

fn initial_cost(data: &NetworkData, solution: &Solution) -> f64 {
    let mut cost = 0.0;

    // Vehicles — fixed_cost applied once per (depot, vehicle_type) pair
    for vt_indices in &solution.vehicles {
        let mut seen_types: Vec<usize> = Vec::new();
        for &vt_idx in vt_indices {
            let vt = &data.vehicle_types[vt_idx];
            let mut initial = vt.unit_cost;
            if !seen_types.contains(&vt_idx) {
                initial += vt.fixed_cost;
                seen_types.push(vt_idx);
            }
            cost += initial;
        }
    }

    // Ports — fixed_cost applied once per (depot, port_type) pair
    for pt_indices in &solution.ports {
        let mut seen_types: Vec<usize> = Vec::new();
        for &pt_idx in pt_indices {
            let pt = &data.port_types[pt_idx];
            let mut initial = pt.unit_cost;
            if !seen_types.contains(&pt_idx) {
                initial += pt.fixed_cost;
                seen_types.push(pt_idx);
            }
            cost += initial;
        }
    }

    cost
}

// ---------------------------------------------------------------------------
// DailyCost — tour_duration × vehicle.operational_cost
//           + energy × port.operational_cost
//           + energy × port.emissions × emissions_cost
// Mirrors Python's Daily_Cost.evaluate_solution()
// ---------------------------------------------------------------------------

fn daily_cost(data: &NetworkData, solution: &Solution, ec: f64) -> f64 {
    let v: f64 = solution.tours.iter()
        .map(|t| t.duration * data.vehicle_types[t.vehicle_type_idx].operational_cost)
        .sum();

    let p: f64 = solution.supply_events.iter()
        .map(|e| {
            let pt = &data.port_types[e.port_type_idx];
            e.energy * pt.operational_cost + e.energy * pt.emissions * ec
        })
        .sum();

    v + p
}

// ---------------------------------------------------------------------------
// Capital_Cost — NPV of (unit_cost [+ fixed_cost if first of type at depot])
//               + annual_cost discounted over service life
//               + disposal_cost at end of service life
// Mirrors Python's Capital_Cost.evaluate_solution()
// ---------------------------------------------------------------------------

fn capital_cost(data: &NetworkData, solution: &Solution, discount_rate: f64) -> f64 {
    let mut cost = 0.0;
    let d = 1.0 + discount_rate;

    // Vehicles — fixed_cost applied once per (depot, vehicle_type) pair
    for (depot_idx, vt_indices) in solution.vehicles.iter().enumerate() {
        let mut seen_types: Vec<usize> = Vec::new();

        for &vt_idx in vt_indices {
            let vt = &data.vehicle_types[vt_idx];

            let mut initial = vt.unit_cost;
            if !seen_types.contains(&vt_idx) {
                initial += vt.fixed_cost;
                seen_types.push(vt_idx);
            }

            cost += initial;
            let sp = vt.service_periods;
            for period in 1..sp {
                cost += vt.annual_cost / d.powi(period as i32);
            }
            cost += vt.disposal_cost / d.powi(sp as i32);
        }

        let _ = depot_idx; // suppress unused warning
    }

    // Ports — fixed_cost applied once per (depot, port_type) pair
    for (depot_idx, pt_indices) in solution.ports.iter().enumerate() {
        let mut seen_types: Vec<usize> = Vec::new();

        for &pt_idx in pt_indices {
            let pt = &data.port_types[pt_idx];

            let mut initial = pt.unit_cost;
            if !seen_types.contains(&pt_idx) {
                initial += pt.fixed_cost;
                seen_types.push(pt_idx);
            }

            cost += initial;
            let sp = pt.service_periods;
            for period in 1..sp {
                cost += pt.annual_cost / d.powi(period as i32);
            }
            cost += pt.disposal_cost / d.powi(sp as i32);
        }

        let _ = depot_idx;
    }

    cost
}

// ---------------------------------------------------------------------------
// Operational_Cost — NPV of time-based vehicle costs + energy-based port costs,
//                    scaled by schedules_per_period.
// Mirrors Python's Operational_Cost.evaluate_solution()
// ---------------------------------------------------------------------------

fn operational_cost(
    data:                 &NetworkData,
    solution:             &Solution,
    schedules_per_period: f64,
    discount_rate:        f64,
) -> f64 {
    let mut cost = 0.0;
    let d = 1.0 + discount_rate;

    // Vehicle time costs
    for tour in &solution.tours {
        let vt = &data.vehicle_types[tour.vehicle_type_idx];
        let period_cost = schedules_per_period * tour.duration * vt.operational_cost;
        let sp = vt.service_periods;
        for p in 0..sp {
            cost += period_cost / d.powi(p as i32);
        }
    }

    // Port energy costs
    for event in &solution.supply_events {
        let pt = &data.port_types[event.port_type_idx];
        let period_cost = schedules_per_period * event.energy * pt.operational_cost;
        let sp = pt.service_periods;
        for p in 0..sp {
            cost += period_cost / d.powi(p as i32);
        }
    }

    cost
}

// ---------------------------------------------------------------------------
// Emissions_Cost — sum of port.emissions × event.energy over supply events.
// Mirrors Python's Emissions_Cost.evaluate_solution()
// ---------------------------------------------------------------------------

fn emissions_cost(
    data:     &NetworkData,
    solution: &Solution,
) -> f64 {
    solution.supply_events.iter()
        .map(|e| e.energy * data.port_types[e.port_type_idx].emissions)
        .sum()
}

// ---------------------------------------------------------------------------
// FleetPortion objective — fraction of the fleet (across all depots) whose
// vehicle type is in `included`. Mirrors Python's Minimize_Fleet_Portion.
// ---------------------------------------------------------------------------

fn fleet_portion_cost(solution: &Solution, included: &[usize]) -> f64 {
    let total: usize = solution.vehicles.iter().map(|v| v.len()).sum();
    if total == 0 {
        return 0.0;
    }

    let count: usize = solution.vehicles.iter().flatten()
        .filter(|&&vt_idx| included.contains(&vt_idx))
        .count();

    count as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Fleet_Portion constraint — fraction of fleet that is an "included" type
// must be ≥ fleet_portion_min.  Empty included list means no constraint.
// ---------------------------------------------------------------------------

fn fleet_portion_compliant(data: &NetworkData, solution: &Solution) -> bool {
    let included = &data.constraints.fleet_portion_included;
    if included.is_empty() {
        return true;
    }

    let total: usize = solution.vehicles.iter().map(|v| v.len()).sum();
    if total == 0 {
        return true;
    }

    let count: usize = solution.vehicles.iter().flatten()
        .filter(|&&vt_idx| included.contains(&vt_idx))
        .count();

    (count as f64 / total as f64) >= data.constraints.fleet_portion_min
}

// ---------------------------------------------------------------------------
// Port_Limit constraint — number of ports at each depot must not exceed the
// per-depot limit.  Empty limits vec means no constraint.
// ---------------------------------------------------------------------------

fn port_limit_compliant(data: &NetworkData, solution: &Solution) -> bool {
    let limits = &data.constraints.port_limits;
    if limits.is_empty() {
        return true;
    }
    limits.iter().enumerate().all(|(depot_idx, &limit)| {
        solution.ports[depot_idx].len() <= limit
    })
}

// ---------------------------------------------------------------------------
// Lot_Size_Limit constraint — number of vehicles at each depot must not
// exceed the per-depot limit.  Empty limits vec means no constraint.
// ---------------------------------------------------------------------------

fn lot_size_limit_compliant(data: &NetworkData, solution: &Solution) -> bool {
    let limits = &data.constraints.lot_size_limits;
    if limits.is_empty() {
        return true;
    }
    limits.iter().enumerate().all(|(depot_idx, &limit)| {
        solution.vehicles[depot_idx].len() <= limit
    })
}
