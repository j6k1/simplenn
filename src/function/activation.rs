pub trait ActivateF: Send + Sync + 'static {
	fn apply(&self,u:f64,v:&[f64]) -> f64;
	fn derive(&self,e:f64) -> f64;
	fn kind(&self) -> &str;
}
pub struct FIdentity {
}
impl FIdentity {
	pub fn new() -> FIdentity {
		FIdentity {}
	}
}
impl ActivateF for FIdentity {
	fn apply(&self,u:f64,_:&[f64]) -> f64 {
		u
	}

	fn derive(&self,_:f64) -> f64 {
		1f64
	}

	fn kind(&self) -> &str {
		"identity"
	}
}
pub struct FSigmoid {

}
impl FSigmoid {
	pub fn new() -> FSigmoid {
		FSigmoid {}
	}
}
impl ActivateF for FSigmoid {
	fn apply(&self,u:f64,_:&[f64]) -> f64 {
		1.0 / (1.0 + (-u).exp())
	}

	fn derive(&self,e:f64) -> f64 {
		let e = 1.0 / (1.0 + (-e).exp());
		e * (1f64 - e)
	}

	fn kind(&self) -> &str {
		"sigmoid"
	}
}
pub struct FReLU {

}
impl FReLU {
	pub fn new() -> FReLU {
		FReLU {}
	}
}
impl ActivateF for FReLU {
	fn apply(&self,u:f64,_:&[f64]) -> f64 {
		match u {
			u if u > 0f64 => {
				u
			},
			_ => 0f64,
		}
	}

	fn derive(&self,e:f64) -> f64 {
		match e {
			e if e > 0f64 => {
				1f64
			},
			_ => 0f64,
		}
	}

	fn kind(&self) -> &str {
		"relu"
	}
}
pub struct FTanh {

}
impl FTanh {
	pub fn new() -> FTanh {
		FTanh {}
	}
}
impl ActivateF for FTanh {
	fn apply(&self,u:f64,_:&[f64]) -> f64 {
		u.tanh()
	}

	fn derive(&self,e:f64) -> f64 {
		let e = e.tanh();
		1.0f64 - e * e
	}

	fn kind(&self) -> &str {
		"tanh"
	}
}
pub struct FSoftMax {
}
impl FSoftMax {
	pub fn new() -> FSoftMax {
		FSoftMax {}
	}
}
impl ActivateF for FSoftMax {
	fn apply(&self,u:f64,v:&[f64]) -> f64 {
		let alpha = v.iter().fold(0.0/0.0, |m,v| v.max(m));
		let numer = (u - alpha).exp();
		numer / v.iter().fold(0.0f64,|acc, &x| acc + (x - alpha).exp())
	}

	fn derive(&self,e:f64) -> f64 {
		e * (1.0f64 - e)
	}

	fn kind(&self) -> &str {
		"softmax"
	}
}