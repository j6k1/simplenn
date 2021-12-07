pub mod activation;
pub mod loss;
pub mod optimizer;

const FIXED_I8_E:i8 = (2.71828182845904 * 8.) as i8;

pub trait Max {
    fn max(&self,other:&Self) -> Self;
}
impl Max for f64 {
    fn max(&self,other:&f64) -> f64 {
        (*self).max(*other)
    }
}
impl Max for i8 {
    fn max(&self,other:&i8) -> i8 {
        (*self).max(*other)
    }
}
pub trait IntegerPartOne {
    fn integer_part_one() -> Self;
}
pub trait Exp {
    fn exp(&self) -> Self;
}
pub trait Tanh: Exp {
    fn tanh(&self) -> Self;
}
impl IntegerPartOne for f64 {
    #[inline]
    fn integer_part_one() -> f64 {
        1f64
    }
}
impl IntegerPartOne for i8 {
    #[inline]
    fn integer_part_one() -> i8 {
        8
    }
}
impl Exp for f64 {
    #[inline]
    fn exp(&self) -> f64 {
        (*self).exp()
    }
}
impl Exp for i8 {
    #[inline]
    fn exp(&self) -> i8 {
        FIXED_I8_E.pow(*self as u32)
    }
}
pub trait InitialMax {
    fn initial_max() -> Self;
}
impl InitialMax for f64 {
    #[inline]
    fn initial_max() -> f64 {
        0.0/0.0
    }
}
impl InitialMax for i8 {
    #[inline]
    fn initial_max() -> i8 {
        -128i8
    }
}
impl Tanh for f64 {
    #[inline]
    fn tanh(&self) -> f64 {
        (*self).tanh()
    }
}
impl Tanh for i8 {
    #[inline]
    fn tanh(&self) -> i8 {
        (1 - self.exp()) / (1 + self.exp())
    }
}