use locus_core::image::ImageView;
use numpy::{PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

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
    Ok(format!(
        "{} - Python Bindings Active",
        locus_core::core_info()
    ))
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
            "Image must have C-contiguous rows (inner stride must be 1)",
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
    let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };

    let view = ImageView::new(data, width, height, stride)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    // Use the optimized Detector pipeline
    let mut detector = locus_core::Detector::new();
    let detections = detector.detect(&view);

    Ok(detections.into_iter().map(Detection::from).collect())
}

/// Detect tags using the gradient-based pipeline (faster).
#[pyfunction]
fn detect_tags_gradient(img: PyReadonlyArray2<u8>) -> PyResult<Vec<Detection>> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride = strides[0] as usize;

    if strides[1] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Image must have C-contiguous rows (inner stride must be 1)",
        ));
    }

    let required_size = if height > 0 && width > 0 {
        (height - 1) * stride + width
    } else {
        0
    };

    let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
    let view = ImageView::new(data, width, height, stride)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let mut detector = locus_core::Detector::new();
    let detections = detector.detect_gradient(&view);

    Ok(detections.into_iter().map(Detection::from).collect())
}

/// For debugging: Apply thresholding and return the binarized image.
#[pyfunction]
fn debug_threshold(img: PyReadonlyArray2<u8>) -> PyResult<PyObject> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride = strides[0] as usize;

    if strides[1] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Inner stride must be 1",
        ));
    }

    let required_size = if height > 0 && width > 0 {
        (height - 1) * stride + width
    } else {
        0
    };

    let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
    let view = ImageView::new(data, width, height, stride)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let mut detector = locus_core::Detector::new();
    // We need to reach into the internal threshold engine for this specific debug helper
    // or we can just expose a helper that returns the binarized image from the arena.
    // For now, let's keep it simple as we are refactoring.

    let mut output = vec![0u8; width * height];
    // This is slightly inefficient (copies) but it's a debug helper.
    let _detections = detector.detect(&view); // This runs thresholding too but ignores output

    // Let's re-run thresholding into our output vec for this debug function
    let stats = locus_core::threshold::ThresholdEngine::new().compute_tile_stats(&view);
    locus_core::threshold::ThresholdEngine::new().apply_threshold(&view, &stats, &mut output);

    Python::with_gil(|py| {
        use numpy::PyArrayMethods;
        let array = numpy::PyArray1::from_vec(py, output);
        let array2d = array.reshape([height, width]).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to reshape NumPy array")
        })?;
        Ok(array2d.into_any().unbind())
    })
}

/// For debugging: Return the labeled connected components.
#[pyfunction]
fn debug_segmentation(img: PyReadonlyArray2<u8>) -> PyResult<PyObject> {
    let shape = img.shape();
    let height = shape[0];
    let width = shape[1];
    let strides = img.strides();
    let stride = strides[0] as usize;

    if strides[1] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Inner stride must be 1",
        ));
    }

    let required_size = if height > 0 && width > 0 {
        (height - 1) * stride + width
    } else {
        0
    };

    let data = unsafe { std::slice::from_raw_parts(img.data(), required_size) };
    let view = ImageView::new(data, width, height, stride)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let _detector = locus_core::Detector::new();

    // Run the pipeline to get labels
    // We need a helper in locus-core to access intermediate labels or return them
    // For now, let's just re-implement the labeling here manually or add a pub helper.
    let engine = locus_core::threshold::ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&view);
    let mut binarized = vec![0u8; width * height];
    engine.apply_threshold(&view, &stats, &mut binarized);

    let arena = bumpalo::Bump::new();
    let labels = locus_core::segmentation::label_components(&arena, &binarized, width, height);

    // Copy labels to a Vec for Python
    let labels_vec = labels.to_vec();

    Python::with_gil(|py| {
        use numpy::PyArrayMethods;
        let array = numpy::PyArray1::from_vec(py, labels_vec);
        let array2d = array.reshape([height, width]).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Failed to reshape NumPy array")
        })?;
        Ok(array2d.into_any().unbind())
    })
}

/// The locus Python module.
#[pymodule]
fn locus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detection>()?;
    m.add_function(wrap_pyfunction!(dummy_detect, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags, m)?)?;
    m.add_function(wrap_pyfunction!(detect_tags_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(debug_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(debug_segmentation, m)?)?;
    Ok(())
}
