pub trait LossFunction {
	fn apply(&self,r:f64,t:f64) -> f64;
	fn derive(&self,r:f64,t:f64) -> f64;
}
pub struct Mse {

}
impl Mse {
	pub fn new() -> Mse {
		Mse {}
	}
}
impl LossFunction for Mse {
	fn apply(&self,r:f64,t:f64) -> f64 {
		(r - t) * (r - t) / 2f64
	}

	fn derive(&self,r:f64,t:f64) -> f64 {
		r- t
	}
}
pub struct CrossEntropy {

}
impl CrossEntropy {
	pub fn new() -> CrossEntropy {
		CrossEntropy {}
	}
}
impl LossFunction for CrossEntropy {
	fn apply(&self,r:f64,t:f64) -> f64 {
		-t * r.ln() - (1.0f64 - t) * (1.0f64 - r).ln()
	}

	fn derive(&self,r:f64,t:f64) -> f64 {
		(r - t) / (r * (1.0f64 - r))
	}
}
pub struct CrossEntropyMulticlass {

}
impl CrossEntropyMulticlass {
	pub fn new() -> CrossEntropyMulticlass {
		CrossEntropyMulticlass {}

	}
}
impl LossFunction for CrossEntropyMulticlass {
	fn apply(&self,r:f64,t:f64) -> f64 {
		-t * r.ln()
	}

	fn derive(&self,r:f64,t:f64) -> f64 {
		-t / r
	}
}