use pyo3::prelude::*;
use pyo3::types::PyBytes;

// import core sketches
use cpc::CpcSketch as RustCpc;
use hll::HllSketch as RustHll;
use theta::ThetaSketch as RustTheta;
// … and so on for AOD, KLL, Quantiles, FrequentStrings

#[pymodule]
fn datasketches(py: Python, m: &PyModule) -> PyResult<()> {
    /// CPC Sketch
    #[pyclass]
    struct CpcSketch {
        sketch: RustCpc,
    }
    #[pymethods]
    impl CpcSketch {
        #[new]
        fn new(lg_k: Option<u8>) -> Self {
            CpcSketch {
                sketch: RustCpc::new(lg_k.unwrap_or(11)),
            }
        }
        fn update(&mut self, item: &str) {
            self.sketch.update(&item);
        }
        fn estimate(&self) -> f64 {
            self.sketch.estimate()
        }
        fn merge(&mut self, other: &CpcSketch) {
            self.sketch.merge(&other.sketch);
        }
        fn to_bytes<'p>(&self, py: Python<'p>) -> &'p PyBytes {
            PyBytes::new(py, &self.sketch.to_bytes())
        }
    }
    m.add_class::<CpcSketch>()?;

    /// HLL Sketch
    #[pyclass]
    struct HllSketch {
        sketch: RustHll,
    }
    #[pymethods]
    impl HllSketch {
        #[new]
        fn new(lg_k: Option<u8>) -> Self {
            HllSketch {
                sketch: RustHll::new(lg_k.unwrap_or(12)),
            }
        }
        fn update(&mut self, item: &str) {
            self.sketch.update(&item);
        }
        fn estimate(&self) -> f64 {
            self.sketch.estimate()
        }
        fn merge(&mut self, other: &HllSketch) {
            self.sketch.merge(&other.sketch);
        }
    }
    m.add_class::<HllSketch>()?;

    /// Theta Sketch
    #[pyclass]
    struct ThetaSketch {
        sketch: RustTheta,
    }
    #[pymethods]
    impl ThetaSketch {
        #[new]
        fn new(k: Option<usize>) -> Self {
            ThetaSketch {
                sketch: RustTheta::new(k.unwrap_or(1024)),
            }
        }
        fn update(&mut self, item: &str) {
            self.sketch.update(&item);
        }
        fn estimate(&self) -> f64 {
            self.sketch.estimate()
        }
        fn union(&self, other: &ThetaSketch) -> ThetaSketch {
            ThetaSketch {
                sketch: RustTheta::union(&[&self.sketch, &other.sketch], self.sketch.k),
            }
        }
        fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
            ThetaSketch {
                sketch: RustTheta::intersect(&self.sketch, &other.sketch, self.sketch.k),
            }
        }
        fn difference(&self, other: &ThetaSketch) -> ThetaSketch {
            ThetaSketch {
                sketch: RustTheta::difference(&self.sketch, &other.sketch, self.sketch.k),
            }
        }
    }
    m.add_class::<ThetaSketch>()?;
    // m.add_class::<KllFloatSketch>()?;
    // m.add_class::<KllDoubleSketch>()?;
    // m.add_class::<AodSketch>()?;
    // m.add_class::<QuantilesSketch>()?;
    // m.add_class::<FrequentStringsSketch>()?;

    Ok(())
}
