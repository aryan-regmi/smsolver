// FIXME: Remove
#![allow(dead_code)]

pub mod supports;

pub mod units;
pub use units::angle::{Degree, Radian};
pub use units::force::{Newton, Pound};
pub use units::length::{Kilometer, Meter};

pub mod system_solver;

// TODO: Make all structs w/ units generic over the units
