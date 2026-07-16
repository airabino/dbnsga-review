// =============================================================================
// ga.rs — Genetic algorithm operators: generate, mate, mutate.
//
// All operators work on integer-indexed gene arrays (Vec<usize>) rather than
// the Python string-keyed dicts, eliminating hash-map overhead per operator.
//
// Successor encoding:
//   successors[trip_idx] < n_trips  → another trip
//   successors[trip_idx] >= n_trips → depot sentinel (idx - n_trips)
// =============================================================================

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{WeightedIndex, Distribution};
use std::collections::{HashMap, HashSet};
use indexmap::IndexMap;

use crate::types::*;

// ---------------------------------------------------------------------------
// generate_solution
//
// Builds random gene arrays for one Solution.
//
// Successor strategy (mirrors Python's generate_successors):
//   Shuffle trip indices. For each trip in order, pick a random valid
//   successor from trip_adj that hasn't already been claimed as a target,
//   falling back to a random depot sentinel when none are free.
//   This prevents "forking" (two trips leading to the same trip).
//
// Equipment: uniform random vehicle type.
// Supply type: uniform random port type compatible with the chosen vehicle.
// ---------------------------------------------------------------------------

pub fn generate_solution<R: Rng>(data: &NetworkData, rng: &mut R) -> Solution {
    let n = data.n_trips;

    // ---- successors --------------------------------------------------------
    // Any value >= n is the single depot sentinel. Depot probability scales with
    // 1 / (n_free_adj_trips + 1), mirroring Python's generate_successors().
    let mut successors = vec![0usize; n];
    let mut claimed_targets: HashSet<usize> = HashSet::new();

    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);

    for &src in &order {
        let mut candidates: Vec<usize> = data.trip_adj[src]
            .iter()
            .copied()
            .filter(|t| !claimed_targets.contains(t))
            .collect();
        candidates.push(n); // single depot sentinel

        let target = candidates[rng.gen_range(0..candidates.len())];

        successors[src] = target;
        if target < n {
            claimed_targets.insert(target);
        }
    }

    // ---- equipment & supply_type -------------------------------------------
    let mut equipment   = vec![0usize; n];
    let mut supply_type = vec![0usize; n];

    // const n_vt: usize = &data.vt_probabilities.len();
    // let vt_probabilities_gen: [f64; 5] = [(); 5].map(|_| rng.gen_range(0.0..100.0));
    // rng.fill(&mut vt_probabilities_gen[..]);
    // let vt_probabilities_gen = [1, 0, 0, 0, 0];

    let vt_probabilities_gen: Vec<f64> = (0..data.vt_probabilities.len())
        .map(|_| rng.gen::<f64>())
        .collect();


    let vt_dist = WeightedIndex::new(&vt_probabilities_gen)
        .expect("vt_probabilities must be non-empty and non-negative");

    // let vt_dist = WeightedIndex::new(&data.vt_probabilities)
    //     .expect("vt_probabilities must be non-empty and non-negative");
    for i in 0..n {
        let vt_idx = vt_dist.sample(rng);
        equipment[i] = vt_idx;

        let sti = &data.vehicle_types[vt_idx].supply_type_indices;
        supply_type[i] = if sti.is_empty() {
            0
        } else {
            sti[rng.gen_range(0..sti.len())]
        };
    }

    // ---- start_depot & end_depot -------------------------------------------
    let mut start_depot = vec![0usize; n];
    let mut end_depot   = vec![0usize; n];

    let depot_probabilities_gen: Vec<f64> = (0..data.depot_probabilities.len())
        .map(|_| rng.gen::<f64>())
        .collect();

    let depot_dist = WeightedIndex::new(&depot_probabilities_gen)
        .expect("depot_probabilities must be non-empty and non-negative");

    for i in 0..n {
        start_depot[i] = depot_dist.sample(rng);
        end_depot[i]   = depot_dist.sample(rng);
    }

    let mut sol = Solution::new(n, data.n_depots, data.objectives.len());
    sol.successors   = successors;
    sol.equipment    = equipment;
    sol.supply_type  = supply_type;
    sol.start_depot  = start_depot;
    sol.end_depot    = end_depot;
    sol
}

// ---------------------------------------------------------------------------
// mate_mutate
//
// Combined crossover + mutation operator. Mirrors Python's Solution.mate_mutate():
//
// 1. Copy parent_a's trip→trip successor links (depot targets dropped).
// 2. Select crossover genes from partner (non-depot successors) into
//    a contribution dict.
// 3. Mutate the contribution genes — pick new targets from trip_adj
//    (trip-only, no depots), resolving conflicts within contribution.
// 4. Integrate contribution into child's successors with adjacency-validated
//    conflict resolution.
// 5. Fill gaps with random depot sentinels.
// 6. Equipment: crossover from partner, then weighted mutation.
// ---------------------------------------------------------------------------

pub fn mate_mutate<R: Rng>(
    data:                  &NetworkData,
    parent_a:              &Solution,
    parent_b:              &Solution,
    crossover_probability: f64,
    mutation_probability:  f64,
    rng:                   &mut R,
) -> Solution {
    let n = data.n_trips;

    // ---- Step 1: Copy parent_a's trip→trip links (drop depot targets) ----
    let mut successors:   IndexMap<usize, usize> = IndexMap::new();
    let mut predecessors: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        let s = parent_a.successors[i];
        if s < n {
            successors.insert(i, s);
            predecessors.insert(s, i);
        }
    }

    // ---- Step 2: Select crossover genes from partner --------------------
    let rn_crossover: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

    let mut contribution: IndexMap<usize, usize> = IndexMap::new();

    for i in 0..n {
        if rn_crossover[i] <= crossover_probability {
            let target = parent_b.successors[i];
            if target < n {
                contribution.insert(i, target);
            }
        }
    }

    // ---- Step 3: Integrate contribution into successors -----------------
    for (&source, &target) in &contribution {
        if predecessors.contains_key(&target) {
            if successors.contains_key(&source) {
                let conflict_key = predecessors[&target];
                let conflict_value = successors.remove(&source).unwrap();

                successors.insert(source, target);
                predecessors.insert(target, source);

                if data.trip_adj[conflict_key].contains(&conflict_value) {
                    successors.insert(conflict_key, conflict_value);
                    predecessors.insert(conflict_value, conflict_key);
                } else {
                    successors.remove(&conflict_key);
                    predecessors.remove(&conflict_value);
                }
            } else {
                let conflict_key = predecessors[&target];
                successors.remove(&conflict_key);

                successors.insert(source, target);
                predecessors.insert(target, source);
            }
        } else {
            successors.insert(source, target);
            predecessors.insert(target, source);
        }
    }

    // ---- Step 4: Mutation -----------------------------------------------
    let mut inverse_contribution: HashMap<usize, usize> = HashMap::new();
    for (&k, &v) in &successors {
        inverse_contribution.insert(v, k);
    }

    let rn_mutation: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

    for i in 0..n {
        if rn_mutation[i] <= mutation_probability {
            let candidates = &data.trip_adj[i];
            if candidates.is_empty() {
                continue;
            }
            let target = candidates[rng.gen_range(0..candidates.len())];

            if let Some(&old_target) = successors.get(&i) {
                inverse_contribution.remove(&old_target);
            }

            if let Some(&conflict_key) = inverse_contribution.get(&target) {
                successors.shift_remove(&conflict_key);
            }

            successors.insert(i, target);
            inverse_contribution.insert(target, i);
        }
    }

    // ---- Step 5: Fill gaps with the single depot sentinel ---------------
    let mut child_successors = vec![0usize; n];
    for i in 0..n {
        child_successors[i] = successors.get(&i).copied().unwrap_or(n);
    }

    // ---- Step 6: Equipment + depot crossover then mutation --------------
    let mut equipment    = parent_a.equipment.clone();
    let mut supply_type  = parent_a.supply_type.clone();
    let mut start_depot  = parent_a.start_depot.clone();
    let mut end_depot    = parent_a.end_depot.clone();

    let rn_eq_cross: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    for i in 0..n {
        if rn_eq_cross[i] <= crossover_probability {
            equipment[i]   = parent_b.equipment[i];
            supply_type[i] = parent_b.supply_type[i];
            start_depot[i] = parent_b.start_depot[i];
            end_depot[i]   = parent_b.end_depot[i];
        }
    }

    // Mutation: resample equipment, supply_type, start_depot, end_depot
    let rn_eq_mut: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

    let vt_dist = WeightedIndex::new(&data.vt_probabilities)
        .expect("vt_probabilities must be non-empty and non-negative");
    let depot_dist = WeightedIndex::new(&data.depot_probabilities)
        .expect("depot_probabilities must be non-empty and non-negative");

    let pre_sampled_vt:    Vec<usize> = (0..n).map(|_| vt_dist.sample(rng)).collect();
    let pre_sampled_sd:    Vec<usize> = (0..n).map(|_| depot_dist.sample(rng)).collect();
    let pre_sampled_ed:    Vec<usize> = (0..n).map(|_| depot_dist.sample(rng)).collect();

    for i in 0..n {
        if rn_eq_mut[i] <= mutation_probability {
            equipment[i]   = pre_sampled_vt[i];
            start_depot[i] = pre_sampled_sd[i];
            end_depot[i]   = pre_sampled_ed[i];

            let sti = &data.vehicle_types[equipment[i]].supply_type_indices;
            supply_type[i] = if sti.is_empty() {
                0
            } else {
                sti[rng.gen_range(0..sti.len())]
            };
        }
    }

    let mut child = Solution::new(n, data.n_depots, data.objectives.len());
    child.successors  = child_successors;
    child.equipment   = equipment;
    child.supply_type = supply_type;
    child.start_depot = start_depot;
    child.end_depot   = end_depot;
    child
}
