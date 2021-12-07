use std::ops::{Add, Sub, Mul, Div, Neg};
use function::{Exp, IntegerPartOne, Tanh, InitialMax, Max};
use std::marker::PhantomData;

pub trait ActivateF<T>: Send + Sync + 'static
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {

	fn apply(&self,u:T,v:&[T]) -> T;
	fn derive(&self,e:T) -> T;
	fn kind(&self) -> &str;
}
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
impl<T> ActivateF<T> for FIdentity<T>
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {
	fn apply(&self,u:T,_:&[T]) -> T {
		u
	}

	fn derive(&self,_:T) -> T {
		T::integer_part_one()
	}

	fn kind(&self) -> &str {
		"identity"
	}
}
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
impl<T> ActivateF<T> for FSigmoid<T>
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {
	fn apply(&self,u:T,_:&[T]) -> T {
		T::integer_part_one() / (T::integer_part_one() + (-u).exp())
	}

	fn derive(&self,e:T) -> T {
		let e = T::integer_part_one() / (T::integer_part_one() + (-e).exp());
		e * (T::integer_part_one() - e)
	}

	fn kind(&self) -> &str {
		"sigmoid"
	}
}
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
impl<T> ActivateF<T> for FReLU<T>
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {
	fn apply(&self,u:T,_:&[T]) -> T {
		match u {
			u if u > T::default() => {
				u
			},
			_ => T::default(),
		}
	}

	fn derive(&self,e:T) -> T {
		match e {
			e if e > T::default() => {
				T::integer_part_one()
			},
			_ => T::default(),
		}
	}

	fn kind(&self) -> &str {
		"relu"
	}
}
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
impl<T> ActivateF<T> for FTanh<T>
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {
	fn apply(&self,u:T,_:&[T]) -> T {
		u.tanh()
	}

	fn derive(&self,e:T) -> T {
		let e = e.tanh();
		T::integer_part_one() - e * e
	}

	fn kind(&self) -> &str {
		"tanh"
	}
}
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
impl<T> ActivateF<T> for FSoftMax<T>
	where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
			 PartialOrd + Exp + Tanh + InitialMax + IntegerPartOne + Max +
			 Default + Clone + Copy + Send + Sync + 'static {

	fn apply(&self,u:T,v:&[T]) -> T {
		let alpha = v.iter().fold(T::initial_max(), |m,&v| v.max(&m));
		let numer = (u - alpha).exp();
		numer / v.iter().fold(T::default(),|acc, &x| acc + (x - alpha).exp())
	}

	fn derive(&self,e:T) -> T {
		e * (T::integer_part_one() - e)
	}

	fn kind(&self) -> &str {
		"softmax"
	}
}