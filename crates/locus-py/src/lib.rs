use pyo3::prelude::*;
use locus_core;

/// A dummy detection function for Phase 0 verification.
#[pyfunction]
fn dummy_detect() -> PyResult<String> {
    Ok(format!("{} - Python Bindings Active", locus_core::core_info()))
}

/// The locus Python module.
#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dummy_detect, m)?)?;
    Ok(())
}
