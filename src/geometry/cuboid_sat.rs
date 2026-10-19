//! Batched cuboid edge tests with the scalar support-point arithmetic and tie order.
use crate::math::{Pose, Real, Vector};
use parry::query::sat::cuboid_cuboid_find_local_separating_edge_twoway as reference;
use parry::shape::Cuboid;

/// Computes the best cuboid edge separating axis in the first box's local space.
///
/// On x86-64/f32, independent axes share SIMD instructions; they retain Parry's
/// normalization, support points, arithmetic order and first-axis tie break. Other
/// configurations retain the original routine, including enhanced determinism.
pub fn cuboid_edge_separation(first: &Cuboid, second: &Cuboid, pose: &Pose) -> (Real, Vector) {
    #[cfg(all(
        feature = "f32",
        target_arch = "x86_64",
        not(feature = "enhanced-determinism")
    ))]
    if pose.translation.is_finite()
        && pose.rotation.is_finite()
        && first.half_extents.is_finite()
        && second.half_extents.is_finite()
    {
        return batched::edge_lanes(first, second, pose);
    }
    reference(first, second, pose)
}

#[cfg(all(
    feature = "f32",
    target_arch = "x86_64",
    not(feature = "enhanced-determinism")
))]
mod batched {
    use super::*;
    use crate::math::{Rotation, SIMD_WIDTH, SimdReal};
    use simba::simd::{SimdComplexField, SimdRealField, SimdValue};
    #[inline(always)]
    fn rotate(q: Rotation, v: [SimdReal; 3]) -> [SimdReal; 3] {
        // Match glam's quaternion-vector expression and each addition's ordering.
        let b = Vector::new(q.x, q.y, q.z);
        let factor = SimdReal::splat(q.w * q.w - b.dot(b));
        let dot = ((v[0] * SimdReal::splat(q.x) + v[1] * SimdReal::splat(q.y))
            + v[2] * SimdReal::splat(q.z))
            * SimdReal::splat(2.0);
        let cross = [
            SimdReal::splat(q.y) * v[2] - SimdReal::splat(q.z) * v[1],
            SimdReal::splat(q.z) * v[0] - SimdReal::splat(q.x) * v[2],
            SimdReal::splat(q.x) * v[1] - SimdReal::splat(q.y) * v[0],
        ];
        core::array::from_fn(|i| {
            (v[i] * factor + SimdReal::splat(b[i]) * dot) + cross[i] * SimdReal::splat(q.w * 2.0)
        })
    }

    pub(super) fn edge_lanes(first: &Cuboid, second: &Cuboid, pose: &Pose) -> (Real, Vector) {
        let columns = [
            pose.rotation * Vector::X,
            pose.rotation * Vector::Y,
            pose.rotation * Vector::Z,
        ];
        let mut axes = [Vector::ZERO; 9];
        for (j, c) in columns.iter().enumerate() {
            axes[j * 3] = Vector::new(0.0, -c.z, c.y);
            axes[j * 3 + 1] = Vector::new(c.z, 0.0, -c.x);
            axes[j * 3 + 2] = Vector::new(-c.y, c.x, 0.0);
        }
        let mut best = (-Real::MAX, Vector::ZERO);
        for start in (0..8).step_by(SIMD_WIDTH) {
            let raw: [SimdReal; 3] = core::array::from_fn(|i| {
                SimdReal::from(core::array::from_fn::<_, SIMD_WIDTH, _>(|lane| {
                    axes[start + lane][i]
                }))
            });
            let lengths = ((raw[0] * raw[0] + raw[1] * raw[1]) + raw[2] * raw[2]).simd_sqrt();
            let axis: [SimdReal; 3] = raw.map(|v| v / lengths);
            let dot = (axis[0] * SimdReal::splat(pose.translation.x)
                + axis[1] * SimdReal::splat(pose.translation.y))
                + axis[2] * SimdReal::splat(pose.translation.z);
            let sign = SimdReal::splat(1.0).simd_copysign(dot);
            let axis = axis.map(|v| v * sign);
            let inverse = rotate(pose.rotation.inverse(), axis.map(|v| -v));
            let p1: [SimdReal; 3] = core::array::from_fn(|i| {
                SimdReal::splat(first.half_extents[i]).simd_copysign(axis[i])
            });
            let p2 = core::array::from_fn(|i| {
                SimdReal::splat(second.half_extents[i]).simd_copysign(inverse[i])
            });
            let p2 = rotate(pose.rotation, p2);
            let gap: [SimdReal; 3] =
                core::array::from_fn(|i| (p2[i] + SimdReal::splat(pose.translation[i])) - p1[i]);
            let separation = (gap[0] * axis[0] + gap[1] * axis[1]) + gap[2] * axis[2];
            for lane in 0..SIMD_WIDTH {
                if lengths.extract(lane) > Real::EPSILON && separation.extract(lane) > best.0 {
                    best = (
                        separation.extract(lane),
                        Vector::new(
                            axis[0].extract(lane),
                            axis[1].extract(lane),
                            axis[2].extract(lane),
                        ),
                    );
                }
            }
        }
        // Nine edge axes leave one scalar tail at either supported lane width.
        let norm = axes[8].length();
        if norm > Real::EPSILON {
            let tail = parry::query::sat::cuboid_cuboid_compute_separation_wrt_local_line(
                first,
                second,
                pose,
                axes[8] / norm,
            );
            if tail.0 > best.0 {
                best = tail;
            }
        }
        best
    }
}

#[cfg(all(test, feature = "f32"))]
mod tests {
    use super::*;
    use crate::alloc_prelude::*;
    use crate::math::Rotation;
    use std::println;
    fn cases(count: usize) -> Vec<(Cuboid, Cuboid, Pose)> {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as Real) / ((1u32 << 24) as Real)
        };
        (0..count)
            .map(|i| {
                let scale = match i % 7 {
                    0 => 1.0e-6,
                    1 => 1.0e6,
                    _ => 1.0,
                };
                let first = Cuboid::new(
                    Vector::new(next(), next(), next()) * scale + Vector::splat(scale * 0.01),
                );
                let second = Cuboid::new(
                    Vector::new(next(), next(), next()) * scale + Vector::splat(scale * 0.01),
                );
                let mut rotation =
                    Rotation::from_scaled_axis(Vector::new(next(), next(), next()) * 6.0);
                if i % 5 == 0 {
                    rotation = Rotation::IDENTITY;
                }
                if i % 5 == 1 {
                    rotation = Rotation::from_rotation_x(10.0_f32.powi(-((i % 8) as i32)));
                }
                let pose = Pose::from_parts(
                    (Vector::new(next(), next(), next()) - Vector::splat(0.5)) * scale * 5.0,
                    rotation,
                );
                (first, second, pose)
            })
            .collect()
    }

    #[test]
    fn edge_lanes_preserve_reference_results() {
        for (index, (first, second, pose)) in cases(200_000).iter().enumerate() {
            let expected = reference(first, second, pose);
            let actual = cuboid_edge_separation(first, second, pose);
            assert_eq!(
                actual.0.to_bits(),
                expected.0.to_bits(),
                "distance case {index}: {pose:?}"
            );
            assert_eq!(
                actual.1.to_array().map(Real::to_bits),
                expected.1.to_array().map(Real::to_bits),
                "normal case {index}: {pose:?}"
            );
        }
    }

    #[test]
    #[ignore = "bounded microbenchmark; no hardware-specific timing gate"]
    fn edge_lanes_cost_diagnostic() {
        use parry::query::sat::cuboid_cuboid_find_local_separating_normal_oneway as face;
        // Match the edge routine's actual caller: unit boxes passing both face tests.
        // Broad scale/degenerate coverage belongs to the separate correctness fixture.
        let cases: Vec<_> = cases(40_000)
            .into_iter()
            .filter_map(|(_, _, pose)| {
                let first = Cuboid::new(Vector::splat(0.5));
                let second = Cuboid::new(Vector::splat(0.5));
                (pose.translation.length() > 0.1
                    && pose.translation.length() < 3.0
                    && face(&first, &second, &pose).0 <= 0.002
                    && face(&second, &first, &pose.inverse()).0 <= 0.002)
                    .then_some((first, second, pose))
            })
            .collect();
        println!("edge-query cases={}", cases.len());
        for repeat in 0..3 {
            for (name, function) in [
                (
                    "reference",
                    reference as fn(&Cuboid, &Cuboid, &Pose) -> (Real, Vector),
                ),
                ("candidate", cuboid_edge_separation),
            ] {
                let start = std::time::Instant::now();
                for _ in 0..2000 {
                    for (first, second, pose) in &cases {
                        std::hint::black_box(function(
                            std::hint::black_box(first),
                            std::hint::black_box(second),
                            std::hint::black_box(pose),
                        ));
                    }
                }
                println!(
                    "{repeat} {name} elapsed_ms={}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }

    #[test]
    fn edge_lanes_preserve_degenerate_axes_and_exact_ties() {
        for half in [0.0, 1.0e-20, 0.5, 1.0, 1.0e10] {
            let a = Cuboid::new(Vector::splat(half));
            let b = Cuboid::new(Vector::new(half, half * 0.5, half * 2.0));
            for rotation in [
                Rotation::IDENTITY,
                Rotation::from_rotation_x(core::f32::consts::PI),
                Rotation::from_rotation_y(core::f32::consts::FRAC_PI_2),
                Rotation::from_rotation_z(Real::EPSILON * 0.5),
            ] {
                for x in [-0.0, 0.0, -half, half, half * 2.0] {
                    for y in [-half, half] {
                        let pose = Pose::from_parts(Vector::new(x, y, x), rotation);
                        let expected = reference(&a, &b, &pose);
                        let actual = cuboid_edge_separation(&a, &b, &pose);
                        assert_eq!(actual.0.to_bits(), expected.0.to_bits(), "{pose:?}");
                        assert_eq!(
                            actual.1.to_array().map(Real::to_bits),
                            expected.1.to_array().map(Real::to_bits),
                            "{pose:?}"
                        );
                    }
                }
            }
        }
    }
}
