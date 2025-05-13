//! Primary library module for sketches crate; provides Python bindings.

#[cfg(feature = "extension-module")]
use pyo3::PyObject;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyBytes;

pub mod cpc;
pub mod hll;
pub mod pp;
pub mod theta;

/// Python binding for CPC sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "CpcSketch")]
pub struct CpcSketch {
    inner: cpc::CpcSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CpcSketch {
    /// Create a new CPC sketch with log2(k) specified by `lg_k`. Defaults to 11.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        CpcSketch {
            inner: cpc::CpcSketch::new(lg_k.unwrap_or(11)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another CPC sketch into this one (in-place union).
    pub fn merge(&mut self, other: &CpcSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).to_object(py)
    }
}

/// Python binding for HyperLogLog sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllSketch")]
pub struct HllSketch {
    inner: hll::HllSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllSketch {
    /// Create a new HLL sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllSketch {
            inner: hll::HllSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another HLL sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).to_object(py)
    }
}

/// Python binding for HyperLogLog++ sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllPlusPlusSketch")]
pub struct HllPlusPlusSketch {
    inner: hll::HllPlusPlusSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllPlusPlusSketch {
    /// Create a new HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSketch {
            inner: hll::HllPlusPlusSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another HLL++ sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllPlusPlusSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).to_object(py)
    }
}

/// Python binding for sparse HyperLogLog++ sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllPlusPlusSparseSketch")]
pub struct HllPlusPlusSparseSketch {
    inner: hll::HllPlusPlusSparseSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllPlusPlusSparseSketch {
    /// Create a new sparse HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSparseSketch {
            inner: hll::HllPlusPlusSparseSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another sparse sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllPlusPlusSparseSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).to_object(py)
    }
}

/// Python binding for Theta sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "ThetaSketch")]
pub struct ThetaSketch {
    inner: theta::ThetaSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl ThetaSketch {
    /// Create a new Theta sketch with sample size `k`. Defaults to 4096.
    #[new]
    fn new(k: Option<usize>) -> Self {
        ThetaSketch {
            inner: theta::ThetaSketch::new(k.unwrap_or(4096)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Union of two sketches, returning a new sketch.
    pub fn union(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::union(&self.inner, &other.inner),
        }
    }

    /// Intersection of two sketches, returning a new sketch.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::intersect(&self.inner, &other.inner),
        }
    }

    /// Difference of two sketches (self \\ other), returning a new sketch.
    pub fn difference(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::difference(&self.inner, &other.inner, self.inner.k),
        }
    }

    /// Return the sample capacity (heap size).
    pub fn sample_capacity(&self) -> usize {
        self.inner.sample_capacity()
    }
}

/// Python binding for PP sketch (placeholder).
#[cfg(feature = "extension-module")]
#[pyclass(name = "PpSketch")]
pub struct PpSketch {
    inner: pp::PpSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PpSketch {
    /// Create a new PP sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        PpSketch {
            inner: pp::PpSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality (not yet implemented).
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }
}

/// Python module definition
#[cfg(feature = "extension-module")]
#[pymodule]
fn sketches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes to module
    m.add_class::<CpcSketch>()?;
    m.add_class::<HllSketch>()?;
    m.add_class::<HllPlusPlusSketch>()?;
    m.add_class::<HllPlusPlusSparseSketch>()?;
    m.add_class::<ThetaSketch>()?;
    m.add_class::<PpSketch>()?;

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

