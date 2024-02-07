use std::fmt;

pub use angle::{Angle, Degree, Radian};
pub use force::{Force, Newton, Pound};
pub use length::{Kilometer, Length, Meter};

pub type Position<U = Meter> = Length<U>;

pub trait Unit: From<f32> + fmt::Debug + Clone + Copy {}

pub mod angle {
    use super::Unit;
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Degree(pub f32);
    impl Unit for Degree {}
    impl From<f32> for Degree {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Degree {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Degree {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Degree {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Degree {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }
    impl Degree {
        pub fn radians(&self) -> Radian {
            Radian(self.0.to_radians())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Radian(pub f32);
    impl Unit for Radian {}
    impl From<f32> for Radian {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Radian {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Radian {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Radian {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Radian {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }
    impl Radian {
        pub fn degrees(&self) -> Degree {
            Degree(self.0.to_degrees())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Angle<U: Unit = Degree> {
        value: U,
    }

    impl<U: Unit> From<U> for Angle<U> {
        fn from(value: U) -> Self {
            Self { value }
        }
    }

    impl<U: Unit> From<&U> for Angle<U> {
        fn from(value: &U) -> Self {
            Self { value: *value }
        }
    }

    impl<U: Unit> From<f32> for Angle<U> {
        fn from(value: f32) -> Self {
            Self {
                value: value.into(),
            }
        }
    }

    impl<U: Unit> Angle<U> {
        pub fn get(&self) -> U {
            self.value
        }
    }
}

pub mod length {
    use super::Unit;
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Meter(pub f32);
    impl Unit for Meter {}
    impl From<f32> for Meter {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Meter {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Meter {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Meter {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Meter {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Kilometer(pub f32);
    impl Unit for Kilometer {}
    impl From<f32> for Kilometer {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Kilometer {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Kilometer {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Kilometer {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Kilometer {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Length<U: Unit = Meter> {
        value: U,
    }

    impl<U: Unit> From<U> for Length<U> {
        fn from(value: U) -> Self {
            Self { value }
        }
    }

    impl<U: Unit> From<&U> for Length<U> {
        fn from(value: &U) -> Self {
            Self { value: *value }
        }
    }

    impl<U: Unit> From<f32> for Length<U> {
        fn from(value: f32) -> Self {
            Self {
                value: value.into(),
            }
        }
    }

    impl<U: Unit> Length<U> {
        pub fn get(&self) -> U {
            self.value
        }
    }
}

pub mod force {
    use super::Unit;
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy)]
    pub struct Pound(pub f32);
    impl Unit for Pound {}
    impl From<f32> for Pound {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Pound {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Pound {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Pound {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Pound {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Newton(pub f32);
    impl Unit for Newton {}
    impl From<f32> for Newton {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for Newton {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for Newton {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for Newton {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for Newton {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Force<U: Unit = Newton> {
        value: U,
    }

    impl<U: Unit> From<U> for Force<U> {
        fn from(value: U) -> Self {
            Self { value }
        }
    }

    impl<U: Unit> From<&U> for Force<U> {
        fn from(value: &U) -> Self {
            Self { value: *value }
        }
    }

    impl<U: Unit> From<f32> for Force<U> {
        fn from(value: f32) -> Self {
            Self {
                value: value.into(),
            }
        }
    }

    impl<U: Unit> Force<U> {
        pub fn get(&self) -> U {
            self.value
        }
    }
}

pub mod moment {
    use super::Unit;
    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NewtonMeter(pub f32);
    impl Unit for NewtonMeter {}
    impl From<f32> for NewtonMeter {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for NewtonMeter {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for NewtonMeter {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for NewtonMeter {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for NewtonMeter {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PoundFoot(pub f32);
    impl Unit for PoundFoot {}
    impl From<f32> for PoundFoot {
        fn from(value: f32) -> Self {
            return Self(value);
        }
    }
    impl Div<f32> for PoundFoot {
        type Output = Self;

        fn div(self, rhs: f32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }
    impl Mul<f32> for PoundFoot {
        type Output = Self;

        fn mul(self, rhs: f32) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
    impl Add<f32> for PoundFoot {
        type Output = Self;

        fn add(self, rhs: f32) -> Self::Output {
            Self(self.0 + rhs)
        }
    }
    impl Sub<f32> for PoundFoot {
        type Output = Self;

        fn sub(self, rhs: f32) -> Self::Output {
            Self(self.0 - rhs)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Moment<U: Unit = NewtonMeter> {
        value: U,
    }

    impl<U: Unit> From<U> for Moment<U> {
        fn from(value: U) -> Self {
            Self { value }
        }
    }

    impl<U: Unit> From<&U> for Moment<U> {
        fn from(value: &U) -> Self {
            Self { value: *value }
        }
    }

    impl<U: Unit> From<f32> for Moment<U> {
        fn from(value: f32) -> Self {
            Self {
                value: value.into(),
            }
        }
    }

    impl<U: Unit> Moment<U> {
        pub fn get(&self) -> U {
            self.value
        }
    }
}
