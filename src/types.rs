use std::convert::From;
use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
use std::fmt;
use Bias;
use std::fmt::Debug;

pub trait UnitValue<T>: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
                        AddAssign + PartialOrd +
                        Clone + Copy + Default + Debug + From<i8> + Send + Sync + 'static +
                        Exp + Tanh + One + Max + Min + MaxValue + InitialMaxValue + Abs + Bias {

}
#[derive(Debug,Clone,Copy,PartialOrd, PartialEq,Ord,Eq)]
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
impl AddAssign for FxS8 {
    fn add_assign(&mut self,other:FxS8) {
        *self = *self + other;
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
        ((self.raw * other.raw) >> 3).into()
    }
}
impl Div for FxS8 {
    type Output = Self;

    fn div(self,other:FxS8) -> FxS8 {
        (((((self.raw as u32) << 3) &
            0b01111111 |
            ((self.raw as u32) & 0b10000000)) as i32 /
            (other.raw as i32)) as i8).into()
    }
}
impl Neg for FxS8 {
    type Output = Self;

    fn neg(self) -> FxS8 {
        FxS8 {
            raw:-self.raw
        }
    }
}
impl Default for FxS8 {
    fn default() -> FxS8 {
        0.into()
    }
}
impl fmt::Display for FxS8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}",self.raw)
    }
}
impl From<f64> for FxS8 {
    fn from(source:f64) -> FxS8 {
        FxS8 {
            raw: (source * 8.) as i8
        }
    }
}
impl From<FxS8> for f64 {
    fn from(source:FxS8) -> f64 {
        source.raw as f64 / 8.
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
pub trait Min {
    fn min(&self,other:&Self) -> Self;
}
impl Min for f64 {
    fn min(&self,other:&f64) -> f64 {
        (*self).min(*other)
    }
}
impl Min for FxS8 {
    fn min(&self,other:&FxS8) -> FxS8 {
        self.raw.min(other.raw).into()
    }
}
pub trait MaxValue {
    fn max_value() -> Self;
}
impl MaxValue for f64 {
    fn max_value() -> f64 {
        f64::MAX
    }
}
impl MaxValue for FxS8 {
    fn max_value() -> FxS8 {
        FxS8 {
            raw:i8::MAX
        }
    }
}
pub trait InitialMaxValue {
    fn initial_max_value() -> Self;
}
impl InitialMaxValue for f64 {
    fn initial_max_value() -> f64 {
        0.0/0.0
    }
}
impl InitialMaxValue for FxS8 {
    fn initial_max_value() -> FxS8 {
        (-128i8).into()
    }
}
pub trait One {
    fn one() -> Self;
}
pub trait Pow {
    fn pow(&self,u32) -> Self;
}
pub trait Exp {
    fn exp(&self) -> Self;
}
pub trait Tanh {
    fn tanh(&self) -> Self;
}
impl One for f64 {
    #[inline]
    fn one() -> f64 {
        1f64
    }
}
impl One for FxS8 {
    #[inline]
    fn one() -> FxS8 {
        8i8.into()
    }
}
impl Pow for FxS8 {
    #[inline]
    fn pow(&self,e:u32) -> FxS8 {
        if e == 0 {
            1.into()
        } else if e == 1 {
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
impl Exp for FxS8 where FxS8: From<i8>, f64: From<FxS8> {
    #[inline]
    fn exp(&self) -> FxS8 {
        f64::from(*self).exp().into()
    }
}
impl Tanh for f64 {
    #[inline]
    fn tanh(&self) -> f64 {
        (*self).tanh()
    }
}
impl Tanh for FxS8 where FxS8: Exp + One {
    #[inline]
    fn tanh(&self) -> FxS8 {
        (FxS8::one() - self.exp()) / (FxS8::one() + self.exp())
    }
}
pub trait Abs {
    fn abs(&self) -> Self;
}
impl Abs for f64 {
    fn abs(&self) -> f64 {
        (*self).abs()
    }
}
impl Abs for FxS8 {
    fn abs(&self) -> FxS8 {
        if self.raw < 0 {
            -(*self)
        } else {
            *self
        }
    }
}
impl UnitValue<f64> for f64 {}
impl UnitValue<FxS8> for FxS8 {}