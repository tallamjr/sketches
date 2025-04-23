// crates/theta/src/lib.rs
// bindings/theta.rs
#[pyclass(name = "ThetaSketch")]
pub struct ThetaSketch {
    sketch: crate::theta::ThetaSketch,
}

#[pymethods]
impl ThetaSketch {
    #[new]
    fn new(obj: &PyRawObject, k: Option<usize>) -> PyResult<()> {
        obj.init(ThetaSketch {
            sketch: crate::theta::ThetaSketch::new(k.unwrap_or(4096)),
        });
        Ok(())
    }
    fn update(&mut self, item: &PyAny) -> PyResult<()> {
        let data = if let Ok(s) = item.extract::<String>() {
            s
        } else if let Ok(i) = item.extract::<i64>() {
            i.to_string()
        } else {
            format!("{:?}", item)
        };
        self.sketch.update(&data);
        Ok(())
    }
    #[pyo3(name = "estimate")]
    fn estimate_py(&self) -> f64 {
        self.sketch.estimate()
    }
    /// Compute the union of this sketch with another, returning a new sketch.
    fn union(&self, other: &ThetaSketch) -> PyResult<ThetaSketch> {
        let result_sk =
            crate::theta::ThetaSketch::union(&[&self.sketch, &other.sketch], self.sketch.k);
        Ok(ThetaSketch { sketch: result_sk })
    }
    /// Compute the intersection of this sketch with another.
    fn intersect(&self, other: &ThetaSketch) -> PyResult<ThetaSketch> {
        let result_sk =
            crate::theta::ThetaSketch::intersect(&self.sketch, &other.sketch, self.sketch.k);
        Ok(ThetaSketch { sketch: result_sk })
    }
    /// Compute A-not-B (this sketch minus elements of other).
    fn difference(&self, other: &ThetaSketch) -> PyResult<ThetaSketch> {
        let result_sk =
            crate::theta::ThetaSketch::difference(&self.sketch, &other.sketch, self.sketch.k);
        Ok(ThetaSketch { sketch: result_sk })
    }
}
