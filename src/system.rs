use crate::units::{self, Angle};

/// A position in 2D.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Position2d {
    x: units::Position,
    y: units::Position,
}

/// Types of supports.
enum Support {
    Pin,
    Roller,
}

/// A node in the structure/truss.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Node {
    /// The position of the node.
    position: Position2d,

    /// Direction of forces at the node.
    ///
    /// This is used to apply boundary constraints.
    force_unit_vectors: [f32; 2],
}

impl Node {
    /// Builds a node constrained by the specified type of support.
    fn support(kind: Support, xpos: f32, ypos: f32) -> Self {
        let position = Position2d {
            x: xpos.into(),
            y: ypos.into(),
        };

        match kind {
            Support::Pin => Self {
                position,
                force_unit_vectors: [1.0, 1.0],
            },

            Support::Roller => Self {
                position,
                force_unit_vectors: [0.0, 1.0],
            },
        }
    }
}

/// An element in the structure/truss.
///
/// This consists of two nodes (that it exists between) and an angle from the X-axis.
#[derive(Debug, Clone, Copy)]
struct Element(Node, Node);

/// A force at `position` that acts in the `unit_vector` direction.
#[derive(Debug, Clone, Copy)]
struct Force {
    position: Position2d,
    unit_vector: [f32; 2],
}

/// The system to solve.
#[derive(Debug, Clone)]
struct System {
    nodes: Vec<Node>,
    elements: Vec<Element>,
    forces: Vec<Force>,
}

impl System {
    /// Sums the forces in the direction of the X-axis.
    fn sum_forces_x(&self) -> Vec<f32> {
        let num_unknowns = self.nodes.len() + self.forces.len();
        let mut fx_coeffs = Vec::with_capacity(num_unknowns);

        for node in &self.nodes {
            fx_coeffs.push(node.force_unit_vectors[0]);
        }

        for force in &self.forces {
            fx_coeffs.push(force.unit_vector[0]);
        }

        fx_coeffs
    }

    /// Sums the forces in the direction of the Y-axis.
    fn sum_forces_y(&self) -> Vec<f32> {
        let num_unknowns = self.nodes.len() + self.forces.len();
        let mut fy_coeffs = Vec::with_capacity(num_unknowns);

        for node in &self.nodes {
            fy_coeffs.push(node.force_unit_vectors[1]);
        }

        for force in &self.forces {
            fy_coeffs.push(force.unit_vector[1]);
        }

        fy_coeffs
    }

    /// Sums the moments about the origin (0,0).
    fn sum_moments(&self) -> Result<Vec<f32>, String> {
        let num_unknowns = self.nodes.len() + self.forces.len();
        let mut moment_coeffs: Vec<f32> = Vec::with_capacity(2 * num_unknowns);

        let r = |x: f32, y: f32| (x.powi(2) + y.powi(2)).sqrt();

        let theta = |n1: Node, n2: Node| {
            let x1 = n1.position.x.get().0;
            let x2 = n2.position.x.get().0;
            let y1 = n1.position.y.get().0;
            let y2 = n2.position.y.get().0;

            (y2 - y1).atan2(x2 - x1)
        };

        for element in &self.elements {
            let n1 = element.0;
            let n2 = element.1;

            let x1 = n1.position.x.get().0;
            let x2 = n2.position.x.get().0;
            let y1 = n1.position.y.get().0;
            let y2 = n2.position.y.get().0;

            let r1 = r(x1, y1);
            moment_coeffs.push(r1 * theta(n1, n2).sin());
            moment_coeffs.push(r1 * theta(n1, n2).cos());

            let r2 = r(x2, y2);
            moment_coeffs.push(r2 * theta(n1, n2).sin());
            moment_coeffs.push(r2 * theta(n1, n2).cos());
        }

        for force in &self.forces {
            let idx = self.get_force_elem(force)?;
            let element = self.elements[idx];
            let n1 = element.0;
            let n2 = element.1;

            let x = force.position.x.get().0;
            let y = force.position.y.get().0;
            let rf = r(x, y);

            moment_coeffs.push(rf * theta(n1, n2).sin());
            moment_coeffs.push(rf * theta(n1, n2).cos());
        }

        Ok(moment_coeffs)
    }

    /// Gets the element that a force is action on.
    fn get_force_elem(&self, f: &Force) -> Result<usize, String> {
        let mut expected_y_pos = 0.0;
        let mut expected_elem_idx = 0;
        for (i, element) in self.elements.iter().enumerate() {
            let n1 = element.0;
            let n2 = element.1;

            let x1 = n1.position.x.get().0;
            let x2 = n2.position.x.get().0;
            let y1 = n1.position.y.get().0;
            let y2 = n2.position.y.get().0;

            let f_xpos = f.position.x.get().0;
            let f_ypos = f.position.y.get().0;

            let slope = (y2 - y1) / (x2 - x1);
            let intercept = y2 - (slope * x2);

            let y_expected = (slope * f_xpos) + intercept;
            expected_y_pos = y_expected;
            expected_elem_idx = i;

            if y_expected == f_ypos {
                return Ok(i);
            }
        }

        Err(format!(
            "Unable to determine the element that the force acts on: Update the y-position of the force to `{}` so that it acts on the element at index {:?}",
            expected_y_pos, expected_elem_idx
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::Meter;

    use super::*;

    #[test]
    fn can_create_system() -> Result<(), String> {
        let l = Meter(4.0);
        let n1 = Node::support(Support::Pin, 0.0, 0.0);
        let n2 = Node::support(Support::Roller, l.0, 0.0);
        let e1 = Element(n1, n2);
        let p = Force {
            position: Position2d {
                x: (l / 2.0).into(),
                y: 0.0.into(),
            },
            unit_vector: [0.0, -1.0],
        };

        let system = System {
            nodes: vec![n1, n2],
            elements: vec![e1],
            forces: vec![p],
        };

        dbg!(system.sum_forces_x());
        dbg!(system.sum_forces_y());
        dbg!(system.sum_moments()?);

        Ok(())
    }
}
