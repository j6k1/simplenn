use types::*;
use std::marker::PhantomData;

pub trait ActivateF<T>: AsActivateF<FxS8> + AsActivateF<FxS16> +
	Send + Sync + 'static
	where T: UnitValue<T> {

	fn apply(&self,u:T,v:&[T]) -> T;
	fn derive(&self,e:T) -> T;
	fn kind(&self) -> &str;
}
pub trait AsActivateF<T>: Send + Sync + 'static where T: Send + Sync + 'static {
	fn as_activate_function(&self) -> Box<dyn ActivateF<T>>;
}
#[derive(Clone)]
pub struct FIdentity<T> {
	t:PhantomData<T>
}
impl<T> FIdentity<T> {
	pub fn new() -> FIdentity<T> {
		FIdentity {
			t:PhantomData::<T>
		}
	}
}
impl<T> ActivateF<T> for FIdentity<T> where T: UnitValue<T> {
	#[inline]
	fn apply(&self,u:T,_:&[T]) -> T {
		u
	}

	#[inline]
	fn derive(&self,_:T) -> T {
		T::one()
	}

	#[inline]
	fn kind(&self) -> &str {
		"identity"
	}
}
impl<T> AsActivateF<FxS8> for FIdentity<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS8>> {
		Box::new(FIdentity::new())
	}
}
impl<T> AsActivateF<FxS16> for FIdentity<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS16>> {
		Box::new(FIdentity::new())
	}
}
#[derive(Clone)]
pub struct FSigmoid<T> {
	t:PhantomData::<T>
}
impl<T> FSigmoid<T> {
	pub fn new() -> FSigmoid<T> {
		FSigmoid {
			t:PhantomData::<T>
		}
	}
}
impl<T> ActivateF<T> for FSigmoid<T> where T: UnitValue<T> {
	#[inline]
	fn apply(&self,u:T,_:&[T]) -> T {
		T::one() / (T::one() + (-u).exp())
	}

	#[inline]
	fn derive(&self,e:T) -> T {
		let e = T::one() / (T::one() + (-e).exp());
		e * (T::one() - e)
	}

	#[inline]
	fn kind(&self) -> &str {
		"sigmoid"
	}
}
impl<T> AsActivateF<FxS8> for FSigmoid<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS8>> {
		Box::new(FSigmoid::new())
	}
}
impl<T> AsActivateF<FxS16> for FSigmoid<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS16>> {
		Box::new(FSigmoid::new())
	}
}
#[derive(Clone)]
pub struct FReLU<T> {
	t:PhantomData::<T>
}
impl<T> FReLU<T> {
	pub fn new() -> FReLU<T> {
		FReLU {
			t:PhantomData::<T>
		}
	}
}
impl<T> ActivateF<T> for FReLU<T> where T: UnitValue<T> {
	#[inline]
	fn apply(&self,u:T,_:&[T]) -> T {
		match u {
			u if u > T::default() => {
				u
			},
			_ => T::default(),
		}
	}

	#[inline]
	fn derive(&self,e:T) -> T {
		match e {
			e if e > T::default() => {
				T::one()
			},
			_ => T::default(),
		}
	}

	#[inline]
	fn kind(&self) -> &str {
		"relu"
	}
}
impl<T> AsActivateF<FxS8> for FReLU<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS8>> {
		Box::new(FReLU::new())
	}
}
impl<T> AsActivateF<FxS16> for FReLU<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS16>> {
		Box::new(FReLU::new())
	}
}
#[derive(Clone)]
pub struct FTanh<T> {
	t:PhantomData::<T>
}
impl<T> FTanh<T> {
	pub fn new() -> FTanh<T> {
		FTanh {
			t:PhantomData::<T>
		}
	}
}
impl<T> ActivateF<T> for FTanh<T> where T: UnitValue<T> {
	#[inline]
	fn apply(&self,u:T,_:&[T]) -> T {
		u.tanh()
	}

	#[inline]
	fn derive(&self,e:T) -> T {
		let e = e.tanh();
		T::one() - e * e
	}

	#[inline]
	fn kind(&self) -> &str {
		"tanh"
	}
}
impl<T> AsActivateF<FxS8> for FTanh<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS8>> {
		Box::new(FTanh::new())
	}
}
impl<T> AsActivateF<FxS16> for FTanh<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS16>> {
		Box::new(FTanh::new())
	}
}
#[derive(Clone)]
pub struct FSoftMax<T> {
	t:PhantomData::<T>
}
impl<T> FSoftMax<T> {
	pub fn new() -> FSoftMax<T> {
		FSoftMax {
			t:PhantomData::<T>
		}
	}
}
impl<T> ActivateF<T> for FSoftMax<T> where T: UnitValue<T> {
	#[inline]
	fn apply(&self,u:T,v:&[T]) -> T {
		let alpha = v.iter().fold(T::initial_max_value(), |m, &v| v.max(&m));
		let numer = (u - alpha).exp();
		numer / v.iter().fold(T::default(),|acc, &x| acc + (x - alpha).exp())
	}

	#[inline]
	fn derive(&self,e:T) -> T {
		e * (T::one() - e)
	}

	#[inline]
	fn kind(&self) -> &str {
		"softmax"
	}
}
impl<T> AsActivateF<FxS8> for FSoftMax<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS8>> {
		Box::new(FSoftMax::new())
	}
}
impl<T> AsActivateF<FxS16> for FSoftMax<T> where T: Send + Sync + 'static {
	#[inline]
	fn as_activate_function(&self) -> Box<dyn ActivateF<FxS16>> {
		Box::new(FSoftMax::new())
	}
}
