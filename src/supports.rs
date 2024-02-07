use crate::units;

/// A 2D position and direction.
#[derive(Debug, Clone, Copy)]
pub struct Position2D {
    /// The position along the x-axis.
    pub x: units::Position,

    /// The position along the y-axis.
    pub y: units::Position,

    /// The angle from the x-axis.
    pub angle: units::Angle,
}

/// Various support types for structures.
#[derive(Debug, Clone, Copy)]
pub enum Support {
    /// A smooth pin/hinge support.
    ///
    /// This has x and y reaction forces.
    Pin(Position2D),

    /// A roller support.
    ///
    /// This has only y-reaction forces.
    Roller(Position2D),
}

impl Support {
    pub(crate) fn reaction_forces(&self) -> [f32; 2] {
        match self {
            Support::Pin(_) => [1.0, 1.0],
            Support::Roller(_) => [0.0, 1.0],
        }
    }

    pub(crate) fn num_reactions(&self) -> usize {
        match self {
            Support::Pin(_) => 2,
            Support::Roller(_) => 1,
        }
    }

    pub(crate) fn reaction_moments(&self) -> f32 {
        match self {
            Support::Pin(_) => 0.0,
            Support::Roller(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::units::*;
    use crate::Meter;

    use super::*;

    #[test]
    fn can_setup_system() {
        let l = Meter(4.0);
        let a = Support::Pin(Position2D {
            x: Meter(0.0).into(),
            y: Meter(0.0).into(),
            angle: Degree(0.0).into(),
        });
        let b = Support::Roller(Position2D {
            x: l.into(),
            y: Meter(0.0).into(),
            angle: Degree(0.0).into(),
        });

        dbg!(l);
        dbg!(a);
        dbg!(b);
    }
}
