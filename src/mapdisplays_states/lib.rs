use numpy::PyArray3;
use pyo3::prelude::*;
use std::sync::Arc;
use parking_lot::Mutex;

#[pymodule]
fn mapdisplays_states(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_class::<PyDisplayState>()?;
    m.add_class::<idle_state::PyIdleState>()?;
    Ok(())
}

pub mod idle_state;

pub trait DisplayState: Send {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)>;
    fn stop(&self) -> PyResult<()>;
}

#[pyclass(subclass)]
pub struct PyDisplayState {
    pub inner: Arc<Mutex<Box<dyn DisplayState>>>,
}

#[pymethods]
impl PyDisplayState {
    fn get_full_frame<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray3<u8>>, u16)> {
        self.inner.lock().get_full_frame(py)
    }

    fn stop(&self) -> PyResult<()> {
        self.inner.lock().stop()
    }
}

#[pyfunction]
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}