# Running the Rust Implementation End-to-End

## 1. Build the Extension

Every time you change Rust code, rebuild with:

```bash
source rust_env/bin/activate
maturin develop   # dev build (fast, no optimizations)
```

Or for a production-speed wheel targeting your main Python:

```bash
maturin build --release --interpreter /opt/anaconda3/bin/python
pip install target/wheels/fleet_opt_core-*.whl --force-reinstall
```

## 2. Construct the Network in Python (unchanged)

You still build the `Network` object exactly as before — loading trips, locations, vehicle types, port types, objectives, and constraints in Python. Nothing about that changes.

## 3. Cross the PyO3 Boundary Once

```python
from src.rust_bridge import build_rust_network
rust_net, index_maps = build_rust_network(network)
```

`build_rust_network` converts all NetworkX graphs and Python objects into flat numpy arrays and passes them to the Rust constructor. After this call, Rust owns the entire problem definition. `index_maps` holds bidirectional string↔integer mappings you will need to interpret results.

## 4. Run the Optimizer

```python
population = rust_net.optimize(
    max_iter=300,
    population_size=150,
    mutation_probability=0.05,
    crossover_probability=0.5,
    seed=42,
)
```

This runs entirely in Rust — all 300 generations of generate/mate/mutate/solve/evaluate/rank/cull — using Rayon to evaluate individuals in parallel. Python's GIL is released for the duration.

## 5. Interpret the Results

`population` is a `list[dict]` where every dict has integer-indexed arrays:

| Key | Type | Description |
|---|---|---|
| `successors` | `list[int]` | Gene array: trip index → successor node index |
| `equipment` | `list[int]` | Gene array: trip index → vehicle type index |
| `supply_type` | `list[int]` | Gene array: trip index → port type index |
| `fitness` | `list[float]` | Objective values in objective definition order |
| `rank` | `int` | Non-dominated front rank (0 = Pareto-optimal) |
| `age` | `int` | Generations this individual has survived |
| `feasible` | `bool` | Energy balance was solvable |
| `compliant` | `bool` | All constraints satisfied |

To use these with existing Python plotting and analysis code, translate back through `index_maps` — e.g. `index_maps['idx_to_vt_name']` to recover vehicle type strings, `index_maps['idx_to_trip']` to recover trip IDs.

## What Does Not Change

The Network construction, the data files, the objectives and constraints configuration, and all downstream analysis and plotting code are unchanged. The Rust layer is a drop-in replacement only for the `network.optimize()` call itself.
