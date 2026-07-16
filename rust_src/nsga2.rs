// =============================================================================
// nsga2.rs — NSGA-II loop: non-dominated sort, crowding distance, main loop.
//
// Implements the fast non-dominated sort (Deb 2002) and crowding-distance
// selection exactly as the Python sorting.py + network.py optimize() do,
// then wraps the full generation loop for exposure via RustNetwork.optimize().
// =============================================================================

use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use indicatif::ProgressBar;
use std::convert::TryFrom;

use crate::types::*;
use crate::ga;
use crate::solve;
use crate::evaluate;

// ---------------------------------------------------------------------------
// fast_non_dominated_sort
//
// Returns rank[i] for each solution index, where rank 0 = Pareto front.
// Minimisation assumed for all objectives.
// ---------------------------------------------------------------------------

pub fn fast_non_dominated_sort(fitness: &[Vec<f64>]) -> Vec<usize> {
    let n = fitness.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dominated:         Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_count:  Vec<usize>      = vec![0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            // Does i dominate j?  all i[k] ≤ j[k]  AND  some i[k] < j[k]
            let all_le = fitness[i].iter().zip(&fitness[j]).all(|(a, b)| a <= b);
            let any_lt = fitness[i].iter().zip(&fitness[j]).any(|(a, b)| a < b);
            if all_le && any_lt {
                dominated[i].push(j);
                domination_count[j] += 1;
            }
        }
    }

    let mut ranks = vec![0usize; n];
    let mut front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();
    let mut rank_idx = 0usize;

    while !front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &front {
            ranks[i] = rank_idx;
            for &j in &dominated[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        rank_idx += 1;
        front = next_front;
    }

    ranks
}

// ---------------------------------------------------------------------------
// crowding_distance_order
//
// Returns indices sorted by crowding distance descending (most diverse first).
// Mirrors Python's crowding_distance_assignment() from sorting.py.
// ---------------------------------------------------------------------------

pub fn crowding_distance_order(fitness: &[Vec<f64>]) -> Vec<usize> {
    let n = fitness.len();
    if n == 0 {
        return Vec::new();
    }
    let m = fitness[0].len();

    let mut distances = vec![0.0f64; n];

    for obj in 0..m {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| fitness[a][obj].total_cmp(&fitness[b][obj]));

        let min_val = fitness[order[0]][obj];
        let max_val = fitness[order[n - 1]][obj];
        let range   = max_val - min_val;

        if range == 0.0 { continue; }

        distances[order[0]]     = f64::INFINITY;
        distances[order[n - 1]] = f64::INFINITY;

        for j in 1..n - 1 {
            let prev = fitness[order[j - 1]][obj];
            let next = fitness[order[j + 1]][obj];
            distances[order[j]] += ((next - prev) / range).powi(2);
        }
    }

    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]));
    sorted
}

// ---------------------------------------------------------------------------
// rank_population
//
// Assigns solution.rank to every feasible solution using NSGA-II dominance.
// Mirrors Python's Network.rank_population():
//   - Viable (feasible + compliant) solutions are ranked first.
//   - Non-viable (feasible but non-compliant) solutions are ranked above all
//     viable solutions, with their own internal NSGA-II ordering.
//   - Infeasible solutions are excluded entirely (kept but unranked).
// Returns the subset that received a rank (viable + non-viable).
// ---------------------------------------------------------------------------

pub fn rank_population(population: &mut Vec<Solution>) {
    let viable:     Vec<usize> = (0..population.len())
        .filter(|&i| population[i].feasible && population[i].compliant)
        .collect();
    let non_viable: Vec<usize> = (0..population.len())
        .filter(|&i| population[i].feasible && !population[i].compliant)
        .collect();

    // Rank viable solutions
    if !viable.is_empty() {
        let fit: Vec<Vec<f64>> = viable.iter().map(|&i| population[i].fitness.clone()).collect();
        let ranks = fast_non_dominated_sort(&fit);
        for (pos, &idx) in viable.iter().enumerate() {
            population[idx].rank = ranks[pos];
        }
    }

    // Rank non-viable solutions above all viable ones
    let viable_max_rank = viable.iter().map(|&i| population[i].rank).max().unwrap_or(0);

    if !non_viable.is_empty() {
        let fit: Vec<Vec<f64>> = non_viable.iter().map(|&i| population[i].fitness.clone()).collect();
        let ranks = fast_non_dominated_sort(&fit);
        for (pos, &idx) in non_viable.iter().enumerate() {
            population[idx].rank = ranks[pos] + viable_max_rank + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// cull_population
//
// Selects `size` solutions from `pool` using front rank then crowding
// distance, exactly as Python's Network.cull_population().
// ---------------------------------------------------------------------------

pub fn cull_population<R: Rng>(
    pool:         Vec<Solution>,
    size:         usize,
    corner_depth: usize,
    rng:          &mut R,
) -> Vec<Solution> {
    let n = pool.len();
    if n <= size {
        return pool;
    }

    // Collect ranks; exclude infeasible (rank = usize::MAX)
    let ranked: Vec<usize> = (0..n)
        .filter(|&i| pool[i].rank != usize::MAX && pool[i].feasible)
        .collect();

    // Shadow vec of effective ranks — mirrors Python's local `rank` dict.
    // Corner overrides write here only, leaving pool[i].rank untouched.
    let mut effective_rank: Vec<usize> = pool.iter().map(|s| s.rank).collect();

    // ---- Spread-Elitism: override corner extrema to rank 0 -----------------
    // Mirrors Python:
    //   candidates_array = np.array(candidates)
    //   indices = np.argsort(candidates_array, axis=0)
    //   min_indices = indices[:corner_depth].flatten()
    //   max_indices = indices[-corner_depth:].flatten()
    //   corner_points = set([*min_indices, *max_indices])
    //   for cp in corner_points: rank[cp] = 0
    if corner_depth > 0 {
        let n_obj = if n > 0 { pool[0].fitness.len() } else { 0 };
        let depth = corner_depth.min(n);

        for obj in 0..n_obj {
            // Sort all population indices by this objective (ascending).
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| pool[a].fitness[obj].total_cmp(&pool[b].fitness[obj]));

            // First `depth` entries = minima; last `depth` entries = maxima.
            // Writing 0 is idempotent so duplicates across objectives are fine.
            for &idx in order.iter().take(depth) {
                effective_rank[idx] = 0;
            }
            for &idx in order.iter().rev().take(depth) {
                effective_rank[idx] = 0;
            }
        }
    }
    // ------------------------------------------------------------------------

    let max_rank = ranked.iter().map(|&i| effective_rank[i]).max().unwrap_or(0);

    // Crowding-distance order over all candidates (global, like Python)
    let all_fitness: Vec<Vec<f64>> = (0..n).map(|i| pool[i].fitness.clone()).collect();
    let cd_order = crowding_distance_order(&all_fitness);

    let mut next_gen: Vec<usize> = Vec::with_capacity(size);

    'outer: for front_rank in 0..=max_rank {
        let in_front: Vec<usize> = ranked.iter().copied()
            .filter(|&i| effective_rank[i] == front_rank)
            .collect();

        let remaining = size.saturating_sub(next_gen.len());
        if remaining == 0 { break; }

        if in_front.len() <= remaining {
            next_gen.extend_from_slice(&in_front);
        } else {
            // Take the `remaining` most diverse from this front
            let mut added = 0;
            for &global_idx in &cd_order {
                if in_front.contains(&global_idx) {
                    next_gen.push(global_idx);
                    added += 1;
                    if added >= remaining {
                        break 'outer;
                    }
                }
            }
        }
    }

    // If still short (e.g. all infeasible), fill by random duplication
    let current = next_gen.len();
    if current < size && !next_gen.is_empty() {
        let extras: Vec<usize> = (0..size - current)
            .map(|_| next_gen[rng.gen_range(0..current)])
            .collect();
        next_gen.extend(extras);
    }

    next_gen.into_iter().map(|i| pool[i].clone()).collect()
}

// ---------------------------------------------------------------------------
// optimize — main NSGA-II loop
//
// Exposed via RustNetwork.optimize(). Sequential evaluation here; Rayon
// parallelisation is added in Step 6.
// ---------------------------------------------------------------------------

/// Per-solution statistics snapshot (mirrors Python's Solution.statistics()).
#[derive(Clone, Debug)]
pub struct SolutionStats {
    pub rank: usize,
    pub age: usize,
    pub fitness: Vec<f64>,
    pub compliant: bool,
}

/// Per-generation snapshot: one SolutionStats per individual in the population.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub stats: Vec<SolutionStats>,
}

impl GenerationStats {
    /// Capture a snapshot of the current population.
    fn capture(population: &[Solution]) -> Self {
        GenerationStats {
            stats: population.iter().map(|sol| SolutionStats {
                rank: sol.rank,
                age: sol.age,
                fitness: sol.fitness.clone(),
                compliant: sol.compliant,
            }).collect(),
        }
    }
}

pub struct OptimizeResult {
    pub population: Vec<Solution>,
    pub generations: usize,
    /// Per-generation statistics: index 0 = initial population, 1..N = after each generation.
    pub generation_stats: Vec<GenerationStats>,
}

pub fn optimize<R: Rng>(
    data:                    &NetworkData,
    max_iter:                usize,
    min_iter:                usize,
    initial_population_size: usize,
    population_size:         usize,
    mutation_probability:    f64,
    crossover_probability:   f64,
    survival_threshold:      usize,
    retries:                 usize,
    corner_depth:            usize,
    same_depot:              bool,
    store_interval:          usize,
    rng:                     &mut R,
) -> OptimizeResult {

    // ---- initial population ------------------------------------------------
    let mut population: Vec<Solution> = (0..initial_population_size)
        .map(|_| ga::generate_solution(data, rng))
        .collect();

    population.par_iter_mut().for_each(|sol| {
        solve::solve(data, sol, same_depot);
        if sol.feasible {
            evaluate::evaluate(data, sol);
        }
    });

    // Cull initial population by crowding distance (mirrors Python:
    //   candidates = [list(s.fitness.values()) for s in population]
    //   order = crowding_distance_assignment(candidates)
    //   population = [population[o] for o in order][:population_size]
    // if population.len() > population_size {
    //     let all_fitness: Vec<Vec<f64>> = population.iter()
    //         .map(|s| s.fitness.clone())
    //         .collect();
    //     let cd_order = crowding_distance_order(&all_fitness);
    //     let selected: Vec<Solution> = cd_order.into_iter()
    //         .take(population_size)
    //         .map(|i| population[i].clone())
    //         .collect();
    //     population = selected;
    // }

    rank_population(&mut population);

    // ---- generation history ------------------------------------------------
    let mut generation_stats: Vec<GenerationStats> = Vec::with_capacity(max_iter + 1);
    generation_stats.push(GenerationStats::capture(&population));

    // ---- generation loop ---------------------------------------------------
    let mut generations_run = 0usize;
    let mut retries_used    = 0usize;

    let max_iter_u64 = u64::try_from(max_iter).unwrap();

    let bar = ProgressBar::new(max_iter_u64);
    for _gen in 0..max_iter {

        bar.inc(1);

        for sol in &mut population {
            sol.age += 1;
        }

        // Crossover + Mutation: shuffle pool, pair consecutive parents
        population.shuffle(rng);

        let mut offspring: Vec<Solution> = Vec::with_capacity(population_size);
        for chunk in population.chunks(2) {
            match chunk {
                [pa, pb] => {
                    offspring.push(ga::mate_mutate(data, pa, pb, crossover_probability, mutation_probability, rng));
                    offspring.push(ga::mate_mutate(data, pb, pa, crossover_probability, mutation_probability, rng));
                }
                [pa] => {
                    // Odd population — duplicate
                    offspring.push(pa.clone());
                }
                _ => {}
            }
        }

        // Solve + evaluate offspring (parallel)
        offspring.par_iter_mut().for_each(|sol| {
            solve::solve(data, sol, same_depot);
            if sol.feasible {
                evaluate::evaluate(data, sol);
            }
        });

        // Combine, rank, cull
        let mut pool = population;
        pool.extend(offspring);

        rank_population(&mut pool);
        population = cull_population(pool, population_size, corner_depth, rng);

        generations_run += 1;

        // Store statistics only on multiples of store_interval.
        // Mirrors Python: if (idx + 1) % store_interval == 0.
        // store_interval=1 (default) stores every generation.
        if store_interval == 0 || generations_run % store_interval == 0 {
            generation_stats.push(GenerationStats::capture(&population));
        }

        // Mirror Python's survival_threshold / retries early-termination logic:
        //   count age-0 solutions (offspring that survived into next generation);
        //   if that count ≤ survival_threshold, increment retries_used;
        //   if retries_used > retries, stop early.
        if generations_run >= min_iter {
            let remaining_offspring = population.iter().filter(|s| s.age == 0).count();
            if remaining_offspring <= survival_threshold {
                retries_used += 1;
            }
            if retries_used > retries {
                break;
            }
        }
    }

    bar.finish();

    OptimizeResult {
        population,
        generations: generations_run,
        generation_stats,
    }
}
