use NNModel;

pub trait Optimizer {
	fn update(&mut self,l:usize,u:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>);
}
pub struct Gradient {
	g:Vec<Vec<Vec<f64>>>
}
impl Gradient {
	pub fn new(model:&NNModel) -> Gradient {
		let mut g = Vec::with_capacity(model.layers.len());

		for layer in &model.layers {
			let mut l = Vec::with_capacity(layer.len());

			for unit in layer {
				let u = vec![0f64; unit.len()];
				l.push(u);
			}

			g.push(l);
		}

		Gradient {
			g:g
		}
	}

	pub fn get(&mut self,l:usize,u:usize) -> &mut Vec<f64> {
		&mut self.g[l][u]
	}
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
	fn update(&mut self,_:usize,_:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>) {
		let a = self.a;
		let lambda = self.lambda;
		for (w,(wi,ei)) in wout.iter_mut().zip(win.iter().zip(e)) {
			*w = wi - a * (ei + lambda * wi);
		}
	}
}
pub struct MomentumSGD {
	a:f64,
	mu:f64,
	lambda:f64,
	vt:Gradient
}
impl MomentumSGD {
	pub fn new(model:&NNModel,a:f64) -> MomentumSGD {
		MomentumSGD {
			a:a,
			mu:0.9,
			lambda:0.0f64,
			vt:Gradient::new(model)
		}
	}
	pub fn with_mu(model:&NNModel,a:f64,mu:f64) -> MomentumSGD {
		MomentumSGD {
			a:a,
			mu:mu,
			lambda:0.0f64,
			vt:Gradient::new(model)
		}
	}
	pub fn with_params(model:&NNModel,a:f64,mu:f64,lambda:f64) -> MomentumSGD {
		MomentumSGD {
			a:a,
			mu:mu,
			lambda:lambda,
			vt:Gradient::new(model)
		}
	}
}
impl Optimizer for MomentumSGD {
	fn update(&mut self,l:usize,u:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>) {
		let vt = self.vt.get(l,u);

		let a = self.a;
		let mu = self.mu;

		let lambda = self.lambda;
		for ((w,wi),(vi,ei)) in wout.iter_mut().zip(win).zip(vt.iter_mut().zip(e)) {
			*vi =  mu * *vi - (1f64 - mu) * a * (ei + lambda * *wi);
			*w = *wi + *vi;
		}
	}
}
pub struct Adagrad {
	a:f64,
	gt:Gradient,
}
impl Adagrad {
	pub fn new(model:&NNModel) -> Adagrad {
		Adagrad {
			a:0.01f64,
			gt:Gradient::new(model)
		}
	}
}
impl Optimizer for Adagrad {
	fn update(&mut self,l:usize,u:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		let gt = self.gt.get(l,u);

		let a = self.a;

		for ((w,wi),(gi,&ei)) in (wout.iter_mut().zip(win)).zip(gt.iter_mut().zip(e)) {
			*gi += ei * ei;
			*w = wi - a * ei / (gi.sqrt() + EPS);
		}
	}
}
pub struct RMSprop {
	a:f64,
	mu:f64,
	gt:Gradient,
}
impl RMSprop {
	pub fn new(model:&NNModel)-> RMSprop {
		RMSprop {
			a:0.0001f64,
			mu:0.9f64,
			gt:Gradient::new(model),
		}
	}
}
impl Optimizer for RMSprop {
	fn update(&mut self,l:usize,u:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		let gt = self.gt.get(l,u);

		let a = self.a;
		let mu = self.mu;

		for ((w,wi),(gi,&ei)) in (wout.iter_mut().zip(win)).zip(gt.iter_mut().zip(e)) {
			*gi = mu * *gi + (1f64 - mu) * ei * ei;
			*w = wi - a * ei / (gi.sqrt() + EPS);
		}
	}
}
pub struct Adam {
	a:f64,
	mt:Gradient,
	vt:Gradient,
	b1:f64,
	b2:f64,
	b1t:f64,
	b2t:f64,
}
impl Adam {
	pub fn new(model:&NNModel) -> Adam {
		Adam {
			a:0.001f64,
			mt:Gradient::new(model),
			vt:Gradient::new(model),
			b1:0.9f64,
			b2:0.999f64,
			b1t:0.9f64,
			b2t:0.999f64,
		}
	}
}
impl Optimizer for Adam {
	fn update(&mut self,l:usize,u:usize,e:&[f64], win:&Vec<f64>,wout:&mut Vec<f64>) {
		const EPS:f64 = 1e-8f64;

		let mt = self.mt.get(l,u);
		let vt = self.vt.get(l,u);

		let a = self.a;
		let b1 = self.b1;
		let b2 = self.b2;
		let b1t = self.b1t;
		let b2t = self.b2t;

		for ((w,wi),(&ei, (mi,vi))) in (wout.iter_mut().zip(win))
									.zip(e.iter().zip(
											mt.iter_mut().zip(vt.iter_mut()))) {
			*mi = b1 * *mi + (1f64 - self.b1) * ei;
			*vi = b2 * *vi + (1f64 - self.b2) * ei * ei;

			*w = wi - a * (*mi / (1f64 - b1t)) / ((*vi / (1f64 - b2t)) + EPS).sqrt();
		}

		self.b1t = b1t * b1;
		self.b2t = b2t * b2;
	}
}
