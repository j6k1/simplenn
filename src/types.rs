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
    #[inline]
    fn from(raw:i8) -> FxS8 {
        FxS8 {
            raw:raw
        }
    }
}
impl Add for FxS8 {
    type Output = Self;

    #[inline]
    fn add(self,other:FxS8) -> FxS8 {
        (self.raw + other.raw).into()
    }
}
impl AddAssign for FxS8 {
    #[inline]
    fn add_assign(&mut self,other:FxS8) {
        *self = *self + other;
    }
}
impl Sub for FxS8 {
    type Output = Self;

    #[inline]
    fn sub(self,other:FxS8) -> FxS8 {
        (self.raw - other.raw).into()
    }
}
impl Mul for FxS8 {
    type Output = Self;

    #[inline]
    fn mul(self,other:FxS8) -> FxS8 {
        ((self.raw * other.raw) >> 3).into()
    }
}
impl Div for FxS8 {
    type Output = Self;

    #[inline]
    fn div(self,other:FxS8) -> FxS8 {
        ((((self.raw as i16) << 3) / (other.raw as i16)) as i8).into()
    }
}
impl Neg for FxS8 {
    type Output = Self;

    #[inline]
    fn neg(self) -> FxS8 {
        FxS8 {
            raw:-self.raw
        }
    }
}
impl Default for FxS8 {
    #[inline]
    fn default() -> FxS8 {
        0.into()
    }
}
impl fmt::Display for FxS8 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}",self.raw)
    }
}
impl From<f64> for FxS8 {
    #[inline]
    fn from(source:f64) -> FxS8 {
        FxS8 {
            raw: (source * 8.) as i8
        }
    }
}
impl From<FxS8> for f64 {
    #[inline]
    fn from(source:FxS8) -> f64 {
        source.raw as f64 / 8.
    }
}
#[derive(Debug,Clone,Copy,PartialOrd, PartialEq,Ord,Eq)]
pub struct FxS16 {
    raw:i16
}
impl From<i16> for FxS16 {
    #[inline]
    fn from(raw:i16) -> FxS16 {
        FxS16 {
            raw:raw
        }
    }
}
impl Add for FxS16 {
    type Output = Self;

    #[inline]
    fn add(self,other:FxS16) -> FxS16 {
        (self.raw + other.raw).into()
    }
}
impl AddAssign for FxS16 {
    #[inline]
    fn add_assign(&mut self,other:FxS16) {
        *self = *self + other;
    }
}
impl Sub for FxS16 {
    type Output = Self;

    #[inline]
    fn sub(self,other:FxS16) -> FxS16 {
        (self.raw - other.raw).into()
    }
}
impl Mul for FxS16 {
    type Output = Self;

    #[inline]
    fn mul(self,other:FxS16) -> FxS16 {
        ((self.raw * other.raw) >> 7).into()
    }
}
impl Div for FxS16 {
    type Output = Self;

    #[inline]
    fn div(self,other:FxS16) -> FxS16 {
        ((((self.raw as i32) << 7) / (other.raw as i32)) as i16).into()
    }
}
impl Neg for FxS16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> FxS16 {
        FxS16 {
            raw:-self.raw
        }
    }
}
impl Default for FxS16 {
    #[inline]
    fn default() -> FxS16 {
        0i16.into()
    }
}
impl fmt::Display for FxS16 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}",self.raw)
    }
}
impl From<f64> for FxS16 {
    #[inline]
    fn from(source:f64) -> FxS16 {
        FxS16 {
            raw: (source * 128.) as i16
        }
    }
}
impl From<FxS16> for f64 {
    #[inline]
    fn from(source:FxS16) -> f64 {
        source.raw as f64 / 128.
    }
}
impl From<i8> for FxS16 {
    #[inline]
    fn from(source:i8) -> FxS16 {
        FxS16 {
            raw: source as i16
        }
    }
}
pub trait Max {
    fn max(&self,other:&Self) -> Self;
}
impl Max for f64 {
    #[inline]
    fn max(&self,other:&f64) -> f64 {
        (*self).max(*other)
    }
}
impl Max for FxS8 {
    #[inline]
    fn max(&self,other:&FxS8) -> FxS8 {
        self.raw.max(other.raw).into()
    }
}
impl Max for FxS16 {
    #[inline]
    fn max(&self,other:&FxS16) -> FxS16 {
        self.raw.max(other.raw).into()
    }
}
pub trait Min {
    fn min(&self,other:&Self) -> Self;
}
impl Min for f64 {
    #[inline]
    fn min(&self,other:&f64) -> f64 {
        (*self).min(*other)
    }
}
impl Min for FxS8 {
    #[inline]
    fn min(&self,other:&FxS8) -> FxS8 {
        self.raw.min(other.raw).into()
    }
}
impl Min for FxS16 {
    #[inline]
    fn min(&self,other:&FxS16) -> FxS16 {
        self.raw.min(other.raw).into()
    }
}
pub trait MaxValue {
    fn max_value() -> Self;
}
impl MaxValue for f64 {
    #[inline]
    fn max_value() -> f64 {
        f64::MAX
    }
}
impl MaxValue for FxS8 {
    #[inline]
    fn max_value() -> FxS8 {
        FxS8 {
            raw:i8::MAX
        }
    }
}
impl MaxValue for FxS16 {
    #[inline]
    fn max_value() -> FxS16 {
        FxS16 {
            raw:i16::MAX
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
    #[inline]
    fn initial_max_value() -> FxS8 {
        (-128i8).into()
    }
}
impl InitialMaxValue for FxS16 {
    #[inline]
    fn initial_max_value() -> FxS16 {
        (i16::MIN).into()
    }
}
pub trait MaxRaw<T> {
    fn max_raw() -> T;
}
impl MaxRaw<f64> for FxS8 {
    #[inline]
    fn max_raw() -> f64 {
        127i8.into()
    }
}
impl MaxRaw<f64> for FxS16 {
    #[inline]
    fn max_raw() -> f64 {
        i16::MAX.into()
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
impl One for FxS16 {
    #[inline]
    fn one() -> FxS16 {
        128i16.into()
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
impl Exp for FxS16 where FxS16: From<i16>, f64: From<FxS16> {
    #[inline]
    fn exp(&self) -> FxS16 {
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
impl Tanh for FxS16 where FxS16: Exp + One {
    #[inline]
    fn tanh(&self) -> FxS16 {
        (FxS16::one() - self.exp()) / (FxS16::one() + self.exp())
    }
}
pub trait Abs {
    fn abs(&self) -> Self;
}
impl Abs for f64 {
    #[inline]
    fn abs(&self) -> f64 {
        (*self).abs()
    }
}
impl Abs for FxS8 {
    #[inline]
    fn abs(&self) -> FxS8 {
        if self.raw < 0 {
            -(*self)
        } else {
            *self
        }
    }
}
impl Abs for FxS16 {
    #[inline]
    fn abs(&self) -> FxS16 {
        if self.raw < 0 {
            -(*self)
        } else {
            *self
        }
    }
}
impl UnitValue<f64> for f64 {}
impl UnitValue<FxS8> for FxS8 {}
impl UnitValue<FxS16> for FxS16 {}