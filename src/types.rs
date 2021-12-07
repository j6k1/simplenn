use std::convert::From;
use std::ops::{Add, Mul, Sub, Div};

const FIXED_I8_E:i8 = (2.71828182845904 * 8.) as i8;

#[derive(Clone,Copy,PartialOrd, PartialEq,Ord,Eq)]
pub struct FxS8 {
    raw:i8
}
impl From<i8> for FxS8 {
    fn from(raw:i8) -> FxS8 {
        FxS8 {
            raw:raw
        }
    }
}
impl Add for FxS8 {
    type Output = Self;

    fn add(self,other:FxS8) -> FxS8 {
        (self.raw + other.raw).into()
    }
}
impl Sub for FxS8 {
    type Output = Self;

    fn sub(self,other:FxS8) -> FxS8 {
        (self.raw - other.raw).into()
    }
}
impl Mul for FxS8 {
    type Output = Self;

    fn mul(self,other:FxS8) -> FxS8 {
        (self.raw * other.raw >> 3).into()
    }
}
impl Div for FxS8 {
    type Output = Self;

    fn div(self,other:FxS8) -> FxS8 {
        (self.raw << 3 / other.raw).into()
    }
}
pub trait Max {
    fn max(&self,other:&Self) -> Self;
}
impl Max for f64 {
    fn max(&self,other:&f64) -> f64 {
        (*self).max(*other)
    }
}
impl Max for FxS8 {
    fn max(&self,other:&FxS8) -> FxS8 {
        self.raw.max(other.raw).into()
    }
}
pub trait IntegerPartOne {
    fn integer_part_one() -> Self;
}
pub trait Pow {
    fn pow(&self,u32) -> Self;
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
impl IntegerPartOne for FxS8 {
    #[inline]
    fn integer_part_one() -> FxS8 {
        8i8.into()
    }
}
impl Pow for FxS8 {
    #[inline]
    fn pow(&self,e:u32) -> FxS8 {
        if e == 1 {
            *self
        } else {
            let mut p = self.pow(e/2);
            p = p * p;

            if e % 2 == 0 {
                p
            } else {
                p * *self
            }
        }
    }
}
impl Exp for f64 {
    #[inline]
    fn exp(&self) -> f64 {
        (*self).exp()
    }
}
impl Exp for FxS8 where FxS8: From<i8> {
    #[inline]
    fn exp(&self) -> FxS8 {
        let e = FxS8::from(FIXED_I8_E);
        e.pow(self.raw as u32)
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
impl InitialMax for FxS8 {
    #[inline]
    fn initial_max() -> FxS8 {
        FxS8::from(-128i8)
    }
}
impl Tanh for f64 {
    #[inline]
    fn tanh(&self) -> f64 {
        (*self).tanh()
    }
}
impl Tanh for FxS8 where FxS8: Exp + IntegerPartOne {
    #[inline]
    fn tanh(&self) -> FxS8 {
        (FxS8::integer_part_one() - self.exp()) / (FxS8::integer_part_one() + self.exp())
    }
}