use crate::{supports::Support, units};

/// A force acting on the system.
#[derive(Debug, Clone, Copy)]
pub struct Force {
    /// The x-position where the force acts.
    pub x_pos: units::Position,

    /// The y-position where the force acts.
    pub y_pos: units::Position,

    /// The magnitude of the force.
    pub magnitude: units::Force,

    /// The direction of the force.
    ///
    /// This is the angle from the x-axis.
    pub direction: units::Angle,
}

pub enum MomentDirection {
    Clockwise,
    CounterClockwise,
}

pub struct Moment {
    /// The x-position where the moment acts.
    pub x_pos: units::Position,

    /// The y-position where the moment acts.
    pub y_pos: units::Position,

    /// The magnitude of the force.
    pub magnitude: units::Force,

    /// The direction of the moment (CW or CCW).
    pub direction: MomentDirection,
}

// TODO: Replace `Vec`s with `ndarray::Array`s
//
/// The system of equations to solve for.
#[derive(Debug, Clone)]
pub struct System {
    /// The external forces acting on a system.
    external_forces: Vec<Force>,

    /// The supports in the system.
    supports: Vec<Support>,
    // /// The forces to solve for.
    // unknowns: Vec<Force>,
}

impl System {
    pub fn new(external_forces: Vec<Force>, supports: Vec<Support>) -> Self {
        Self {
            external_forces,
            supports,
        }
    }

    fn a_matrix(&self) -> Vec<Vec<f32>> {
        let mut size = 0;
        let mut x_forces = vec![0.0, 0.0, 0.0];
        let mut y_forces = vec![0.0, 0.0, 0.0];
        let mut moments = vec![0.0, 0.0, 0.0];
        for (i, support) in self.supports.iter().enumerate() {
            size += support.num_reactions();
            let reaction_forces = support.reaction_forces();
            let reaction_moment = support.reaction_moments();
            x_forces[i] = reaction_forces[0];
            y_forces[i + 1] = reaction_forces[1];
            moments[i] = reaction_moment;
        }

        dbg!(&x_forces);
        dbg!(&y_forces);
        dbg!(&moments);

        // TODO: Factor in external forces for force calcs

        let mut a_matrix: Vec<Vec<f32>> = Vec::with_capacity(size);
        a_matrix.push(x_forces);
        a_matrix.push(y_forces);
        a_matrix.push(moments);

        // TODO: Finish impl!

        a_matrix
    }

    fn b_matrix(&self) -> Vec<f32> {
        todo!()
    }

    /// Solves `Ax = b` for `x`.
    pub fn solve(&self) -> Vec<f32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::supports::Position2D;

    use super::*;
    use units::{Degree, Meter, Newton};

    #[test]
    fn can_solve_systems() {
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
        let p = Force {
            x_pos: (l / 2.0).into(),
            y_pos: 0.0.into(),
            magnitude: Newton(2.0).into(),
            direction: Degree(270.0).into(),
        };

        let system = System::new(vec![p], vec![a, b]);

        let a_matrix = system.a_matrix();
        dbg!(a_matrix);
    }
}
