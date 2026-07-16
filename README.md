# Reviewer Preview of Code Repository for "Efficient Multi-Objective Transit Fleet Size and Composition Optimization via Modified NSGA Methodology"

This repository contains Python and Rust implementations of the methodology described in the paper as well as an example notebook. There are several steps required to set up and run the code and reviewers may run into dependency issues as this repository has not been extensively tested for such issues. The authors have tested implementation in Ubunut/Debian and Mac but not on Windows.

## 1. Create a new Python env

The following instructions assume an Anaconda installation. It is reccommended that users make a new Python env for this codebase. The rust package fleet_opt_core will be installed as a site-package in the active env at time of compilation. After this, fleet_opt_core will be importable from that env.

```bash
conda create -n my_env python=3.12
```

## 2. Make an ipykernel for the env

This is required to run the example notebook. When using the fleet_opt_core package from Jupyter select this kernel.

```bash
pip install ipykernel
python -m ipykernel install --user --name=my_env_name
```

## 3. Set up the Rust environment

```bash
python setup_rust_env.py
```

## 4. Build the Rust package

This will create a package fleet_opt_core in the site-packages directory for the active python env.

```bash
python build_rust_package.py
```

## 5. Build the graphs

The code relies on two graphs to run. These are the Locations and Trips graph and they are built based on a GTFS feed unzipped in a folder. The user may also specify depot locations by adding a depots.txt file to the folder. The following creates these graphs. Note that a roadmap graph is created using OSMNX so an internet connection is required.

GTFS feeds contain many schedules. One may specify a single or several schedules in one of two ways. First, using the -s argument and a schedule

```bash
python gtfs_to_graph.py Data/Unitrans -s 31
```

or by specifying a date using the -d argument 

```bash
python gtfs_to_graph.py Data/Unitrans -d 2026-03-16
```

## 6. Open Example.ipynb