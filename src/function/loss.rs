use function::activation::ActivateF;

pub trait LossFunction: Send + Sync + 'static {
	fn apply(&self,r:f64,t:f64) -> f64;
	fn derive(&self,r:f64,t:f64) -> f64;
	fn is_canonical_link(&self,_:&Box<dyn ActivateF>) -> bool {
		false
	}
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

	fn is_canonical_link(&self,f:&Box<dyn ActivateF>) -> bool {
		match f.kind() {
			"identity" => true,
			_ => false,
		}
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

	fn is_canonical_link(&self,f:&Box<dyn ActivateF>) -> bool {
		match f.kind() {
			"sigmoid" => true,
			_ => false,
		}
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

	fn is_canonical_link(&self,f:&Box<dyn ActivateF>) -> bool {
		match f.kind() {
			"softmax" => true,
			_ => false,
		}
	}
}