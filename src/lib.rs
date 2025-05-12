//! Primary library module for sketches crate; provides Python bindings.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod cpc;
mod hll;
mod theta;
mod pp;

/// Python binding for CPC sketch.
#[pyclass(name = "CpcSketch")]
pub struct CpcSketch {
    inner: cpc::CpcSketch,
}

#[pymethods]
impl CpcSketch {
    /// Create a new CPC sketch with log2(k) specified by `lg_k`. Defaults to 11.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        CpcSketch { inner: cpc::CpcSketch::new(lg_k.unwrap_or(11)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
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
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.inner.to_bytes())
    }
}

/// Python binding for HyperLogLog sketch.
#[pyclass(name = "HllSketch")]
pub struct HllSketch {
    inner: hll::HllSketch,
}

#[pymethods]
impl HllSketch {
    /// Create a new HLL sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllSketch { inner: hll::HllSketch::new(lg_k.unwrap_or(12)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
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
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.inner.to_bytes())
    }
}

/// Python binding for HyperLogLog++ sketch.
#[pyclass(name = "HllPlusPlusSketch")]
pub struct HllPlusPlusSketch {
    inner: hll::HllPlusPlusSketch,
}

#[pymethods]
impl HllPlusPlusSketch {
    /// Create a new HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSketch { inner: hll::HllPlusPlusSketch::new(lg_k.unwrap_or(12)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
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
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.inner.to_bytes())
    }
}

/// Python binding for sparse HyperLogLog++ sketch.
#[pyclass(name = "HllPlusPlusSparseSketch")]
pub struct HllPlusPlusSparseSketch {
    inner: hll::HllPlusPlusSparseSketch,
}

#[pymethods]
impl HllPlusPlusSparseSketch {
    /// Create a new sparse HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSparseSketch { inner: hll::HllPlusPlusSparseSketch::new(lg_k.unwrap_or(12)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
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
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &self.inner.to_bytes())
    }
}

/// Python binding for Theta sketch.
#[pyclass(name = "ThetaSketch")]
pub struct ThetaSketch {
    inner: theta::ThetaSketch,
}

#[pymethods]
impl ThetaSketch {
    /// Create a new Theta sketch with sample size `k`. Defaults to 4096.
    #[new]
    fn new(k: Option<usize>) -> Self {
        ThetaSketch { inner: theta::ThetaSketch::new(k.unwrap_or(4096)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Union of two sketches, returning a new sketch.
    pub fn union(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch { inner: theta::ThetaSketch::union(&self.inner, &other.inner) }
    }

    /// Intersection of two sketches, returning a new sketch.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch { inner: theta::ThetaSketch::intersect(&self.inner, &other.inner) }
    }

    /// Difference of two sketches (self \\ other), returning a new sketch.
    pub fn difference(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch { inner: theta::ThetaSketch::difference(&self.inner, &other.inner, self.inner.k) }
    }

    /// Return the sample capacity (heap size).
    pub fn sample_capacity(&self) -> usize {
        self.inner.sample_capacity()
    }
}

/// Python binding for PP sketch (placeholder).
#[pyclass(name = "PpSketch")]
pub struct PpSketch {
    inner: pp::PpSketch,
}

#[pymethods]
impl PpSketch {
    /// Create a new PP sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        PpSketch { inner: pp::PpSketch::new(lg_k.unwrap_or(12)) }
    }

    /// Update the sketch with an item. Uses the string representation of the Python object.
    pub fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let s = item.str()?.to_str()?;
        self.inner.update(&s);
        Ok(())
    }

    /// Estimate the cardinality (not yet implemented).
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }
}

/// Python module definition
#[pymodule]
fn sketches(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CpcSketch>()?;
    m.add_class::<HllSketch>()?;
    m.add_class::<HllPlusPlusSketch>()?;
    m.add_class::<HllPlusPlusSparseSketch>()?;
    m.add_class::<ThetaSketch>()?;
    m.add_class::<PpSketch>()?;
    Ok(())
}