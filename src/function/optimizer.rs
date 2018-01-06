use std::clone::Clone;

pub trait Optimizer: Clone {
	fn update(&mut self,e:&Vec<f64>, w:&mut Vec<f64>);
}
pub struct SGD {
	a:f64,
	lambda:f64
}
impl SGD {
	pub fn new(a:f64) -> SGD {
		SGD {
			a:a,
			lambda:0.0f64
		}
	}
	pub fn with_lambda(a:f64,lambda:f64) -> SGD {
		SGD {
			a:a,
			lambda:lambda,
		}
	}
}
impl Optimizer for SGD {
	fn update(&mut self,e:&Vec<f64>, w:&mut Vec<f64>) {
		let a = self.a;
		let lambda = self.lambda;
		for (wi,&ei) in w.iter_mut().zip(e) {
			*wi = *wi - a * ei + lambda * *wi;
		}
	}
}
impl Clone for SGD {
	fn clone(&self) -> SGD {
		SGD {
			a:self.a,
			lambda:self.lambda,
		}
	}
}
pub struct Adagrad {
	a:f64,
	gt:Vec<f64>,
}
impl Adagrad {
	pub fn new() -> Adagrad {
		Adagrad {
			a:0.01f64,
			gt:Vec::new(),
		}
	}
}
impl Optimizer for Adagrad {
	fn update(&mut self,e:&Vec<f64>, w:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		match self.gt.len() {
			0 => self.gt.resize(w.len(), 0f64),
			_ => (),
		};

		let a = self.a;

		for (wi,(gi,&ei)) in w.iter_mut().zip(self.gt.iter_mut().zip(e)) {
			*gi = ei * ei;
			*wi = *wi - a * ei / (gi.sqrt() + EPS);
		}
	}
}
impl Clone for Adagrad {
	fn clone(&self) ->  Adagrad {
		Adagrad {
			a:self.a,
			gt:Vec::new(),
		}
	}
}
pub struct RMSprop {
	a:f64,
	mu:f64,
	gt:Vec<f64>,
}
impl RMSprop {
	pub fn new()-> RMSprop {
		RMSprop {
			a:0.0001f64,
			mu:0.9f64,
			gt:Vec::new(),
		}
	}
}
impl Optimizer for RMSprop {
	fn update(&mut self,e:&Vec<f64>, w:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		match self.gt.len() {
			0 => self.gt.resize(w.len(), 0f64),
			_ => (),
		};

		let a = self.a;
		let mu = self.mu;

		for (wi,(gi,&ei)) in w.iter_mut().zip(self.gt.iter_mut().zip(e)) {
			*gi = mu * *gi + (1f64 - mu) * ei * ei;
			*wi = *wi - a * ei / (gi.sqrt() + EPS);
		}
	}
}
impl Clone for RMSprop {
	fn clone(&self) -> RMSprop {
		RMSprop {
			a:0.0001f64,
			mu:0.9f64,
			gt:Vec::new(),
		}
	}
}
pub struct Adam {
	a:f64,
	mt:Vec<f64>,
	vt:Vec<f64>,
	b1:f64,
	b2:f64,
	b1t:f64,
	b2t:f64,
}
impl Adam {
	pub fn new() -> Adam {
		Adam {
			a:0.001f64,
			mt:Vec::new(),
			vt:Vec::new(),
			b1:0.9f64,
			b2:0.999f64,
			b1t:0.9f64,
			b2t:0.999f64,
		}
	}
}
impl Optimizer for Adam {
	fn update(&mut self,e:&Vec<f64>, w:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		match self.mt.len() {
			0 => self.mt.resize(w.len(), 0f64),
			_ => (),
		};

		match self.vt.len() {
			0 => self.vt.resize(w.len(), 0f64),
			_ => (),
		};

		let a = self.a;
		let b1 = self.b1;
		let b2 = self.b2;
		let b1t = self.b1t;
		let b2t = self.b2t;

		for (wi,(&ei, (mi,vi))) in w.iter_mut()
									.zip(e.iter().zip(
											self.mt.iter_mut().zip(self.vt.iter_mut()))) {
			*mi = b1 * *mi + (1f64 - self.b1) * ei;
			*vi = b2 * *vi + (1f64 - self.b2) * ei * ei;
			*wi = *wi - a * ei / (vi.sqrt() + EPS);

			*wi = *wi - a * (*mi / (1f64 - b1t)) / ((*vi / (1f64 - b2t)) + EPS).sqrt();
		}
	}
}
impl Clone for Adam {
	fn clone(&self) -> Adam {
		Adam {
			a:0.001f64,
			mt:Vec::new(),
			vt:Vec::new(),
			b1:0.9f64,
			b2:0.999f64,
			b1t:0.9f64,
			b2t:0.999f64,
		}
	}
}
