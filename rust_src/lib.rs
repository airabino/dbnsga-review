use pyo3::prelude::*;

mod types;
mod network;
pub(crate) mod solve;
pub(crate) mod evaluate;
pub(crate) mod ga;
pub(crate) mod nsga2;

use network::RustNetwork;

/// fleet_opt_core — Rust core for the vehicle fleet optimizer.
///
/// Exposed to Python via PyO3. The single public class is RustNetwork,
/// which owns the immutable problem definition and runs the NSGA-II loop
/// entirely in Rust using Rayon for parallel population evaluation.
#[pymodule]
fn fleet_opt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustNetwork>()?;
    Ok(())
}
