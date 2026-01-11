use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use locus_core::image::ImageView;

/// Re-export Detection for Python
#[pyclass]
pub struct Detection {
    #[pyo3(get)]
    pub id: u32,
    #[pyo3(get)]
    pub center: [f64; 2],
    #[pyo3(get)]
    pub corners: [[f64; 2]; 4],
    #[pyo3(get)]
    pub hamming: u32,
    #[pyo3(get)]
    pub decision_margin: f64,
}

impl From<locus_core::Detection> for Detection {
    fn from(d: locus_core::Detection) -> Self {
        Self {
            id: d.id,
            center: d.center,
            corners: d.corners,
            hamming: d.hamming,
            decision_margin: d.decision_margin,
        }
    }
}

/// A dummy detection function for Phase 0 verification.
#[pyfunction]
fn dummy_detect() -> PyResult<String> {
    Ok(format!("{} - Python Bindings Active", locus_core::core_info()))
}

/// Detect tags in an image. Zero-copy ingestion of NumPy arrays.
#[pyfunction]
fn detect_tags(img: PyReadonlyArray2<u8>) -> PyResult<Vec<Detection>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    
    // Extract strides from NumPy array
    let strides = img.strides();
    let stride = strides[0] as usize; // Stride of the first dimension (rows)
    
    // Safety: we ensure the second dimension has a stride of 1 (C-contiguous rows).
    // This is a common constraint for many CV algorithms.
    if strides[1] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Image must have C-contiguous rows (inner stride must be 1)"
        ));
    }

    // Calculate the total extent of the memory we might access.
    // The last byte accessed is at index: (height - 1) * stride + (width - 1).
    // So the required size is that index + 1.
    let required_size = if height > 0 && width > 0 {
        (height - 1) * stride + width
    } else {
        0
    };

    // Access raw data via buffer protocol.
    // We create a slice that covers the entire extent of the strided image.
    let data = unsafe {
        std::slice::from_raw_parts(img.data(), required_size)
    };


    let view = ImageView::new(data, width, height, stride)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    // Dummy detection logic for Phase 1
    let mut detections = Vec::new();
    if view.height > 10 && view.width > 10 {
        // Return a dummy detection if image is large enough
        detections.push(Detection {
            id: 42,
            center: [view.width as f64 / 2.0, view.height as f64 / 2.0],
            corners: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            hamming: 0,
            decision_margin: 1.0,
        });
    }

    Ok(detections)
}

/// The locus Python module.
#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detection>()?;
    m.add_function(wrap_pyfunction!(dummy_detect, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags, m)?)?;
    Ok(())
}

