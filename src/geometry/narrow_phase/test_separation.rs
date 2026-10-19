use super::cuboid_face_separates;
use crate::alloc_prelude::*;
use crate::prelude::*;
use parry::query::{ContactManifold, DefaultQueryDispatcher, PersistentQueryDispatcher};
use parry::shape::{Cuboid, Shape};

#[test]
fn separating_face_certificate_matches_fresh_stock_contact_queries() {
    let mut random = oorandom::Rand32::new(0x5e9a_12ab);
    let mut verified = 0;
    for case in 0..4096 {
        let mut dimensions = || {
            Vector::new(
                0.1 + random.rand_float() * 4.0,
                0.1 + random.rand_float() * 4.0,
                0.1 + random.rand_float() * 4.0,
            )
        };
        let first = Cuboid::new(dimensions() * if case % 17 == 0 { 2000.0 } else { 1.0 });
        let second = Cuboid::new(dimensions());
        let rotation = Rotation::from_scaled_axis(Vector::new(
            random.rand_float() * 6.0,
            random.rand_float() * 6.0,
            random.rand_float() * 6.0,
        ));
        let origin = Pose::from_parts(Vector::ZERO, rotation);
        let extent = second.compute_aabb(&origin).half_extents();
        let gap = 0.005 + random.rand_float() * 0.2;
        let base = Pose::from_parts(
            Vector::new(first.half_extents.x + extent.x + gap, 0.0, 0.0),
            rotation,
        );
        let prediction = 0.002 + random.rand_float() * 0.05;
        for (first, second, base) in [(&first, &second, base), (&second, &first, base.inverse())] {
            // Recheck a current face, not a drift radius: the tangential moves here
            // deliberately exceed the ordinary pose cache's 5 cm allowance by orders of magnitude.
            for movement in [
                Vector::ZERO,
                Vector::new(-0.01, 0.4, -0.2),
                Vector::new(0.0, 4.0, 9.0),
            ] {
                let current = Pose::from_parts(
                    base.translation + movement,
                    Rotation::from_scaled_axis(movement * 0.03) * base.rotation,
                );
                if !(0..3)
                    .any(|axis| cuboid_face_separates(first, second, &current, axis, prediction))
                {
                    continue;
                }
                let mut manifolds: Vec<ContactManifold<(), ()>> = Vec::new();
                DefaultQueryDispatcher
                    .contact_manifolds(
                        &current,
                        first,
                        second,
                        prediction,
                        &mut manifolds,
                        &mut None,
                    )
                    .unwrap();
                assert!(
                    manifolds.iter().all(|manifold| manifold.points.is_empty()),
                    "case={case}, movement={movement:?}"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 8000, "insufficient oracle coverage: {verified}");
}

fn separated_boxes() -> (
    PhysicsWorld,
    RigidBodyHandle,
    ColliderHandle,
    ColliderHandle,
) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    // Their world AABBs overlap, while a local face separates the rotated boxes. This
    // exercises a real broad-phase false positive rather than injecting a graph edge.
    let rotation = Vector::Z * core::f32::consts::FRAC_PI_4;
    let axis = Rotation::from_scaled_axis(rotation) * Vector::X;
    let (_, first) = world.insert(
        RigidBodyBuilder::fixed().rotation(rotation),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    let (body, second) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(axis * 1.075)
            .rotation(rotation)
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    world.step();
    let pair = world.narrow_phase.contact_pair(first, second).unwrap();
    assert!(!pair.has_any_active_contact());
    assert!(pair.recycle_state.unwrap().separating_axis.is_some());
    assert_eq!(pair.recycle_state.unwrap().qualified_count, 0);
    (world, body, first, second)
}

#[test]
fn empty_pairs_reuse_only_until_the_prediction_region_is_reached() {
    let (mut world, body, first, second) = separated_boxes();
    let axis = world.colliders[first].rotation() * Vector::X;
    for step in 1..=5 {
        world.bodies[body].set_translation(axis * (1.075 - step as f32 * 0.008), true);
        world.step();
        let pair = world.narrow_phase.contact_pair(first, second).unwrap();
        assert!(!pair.has_any_active_contact());
        assert!(pair.recycle_state.unwrap().separating_axis.is_some());
    }
    world.bodies[body].set_translation(axis * 1.014, true);
    world.step();
    assert!(
        world
            .narrow_phase
            .contact_pair(first, second)
            .unwrap()
            .has_any_active_contact()
    );
    assert!(
        world
            .narrow_phase
            .contact_pair(first, second)
            .unwrap()
            .recycle_state
            .unwrap()
            .qualified_count
            > 0
    );
}

#[test]
#[cfg(feature = "serde-serialize")]
fn cached_solver_count_survives_sleep_restore_and_reactivation() {
    let mut world = PhysicsWorld::new();
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::Y * -0.5),
        ColliderBuilder::cuboid(4.0, 0.5, 4.0),
    );
    let (body, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::Y * 0.5),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    for _ in 0..600 {
        world.step();
    }
    assert!(world.bodies[body].is_sleeping());
    let pair = world.narrow_phase.contact_pairs().next().unwrap();
    assert!(pair.recycle_state.unwrap().qualified_count > 0);
    assert!(
        world
            .narrow_phase
            .pair_solver_hints
            .iter()
            .all(|hint| hint & super::PAIR_HINT_COUNT_MASK == 0)
    );

    // A sleeping snapshot retains the count even though its dispatch hints have been
    // cleared. Waking at the same pose must re-enter the solver through recycling.
    let mut restored: PhysicsWorld =
        bincode::deserialize(&bincode::serialize(&world).unwrap()).unwrap();
    for current in [&mut world, &mut restored] {
        current.bodies[body].wake_up(true);
    }
    for step in 0..30 {
        world.step();
        restored.step();
        for current in [&world, &restored] {
            let pair = current.narrow_phase.contact_pairs().next().unwrap();
            assert_eq!(
                pair.recycle_state.unwrap().qualified_count,
                super::pair_qualified_manifold_count(pair)
            );
            let selected = current
                .narrow_phase
                .pair_solver_hints
                .iter()
                .any(|hint| hint & super::PAIR_HINT_COUNT_MASK != 0);
            assert_eq!(selected, !current.bodies[body].is_sleeping(), "step={step}");
            assert!(
                step != 0 || selected,
                "the first wake must restore solver participation"
            );
        }
        assert_eq!(
            (world.bodies[body].position(), world.bodies[body].linvel()),
            (
                restored.bodies[body].position(),
                restored.bodies[body].linvel()
            )
        );
    }
}

#[test]
fn separating_face_survives_large_tangential_motion_and_relative_rotation() {
    let (mut world, body, first, second) = separated_boxes();
    let rotation = world.colliders[first].rotation();
    for step in 1..=20 {
        let local = Vector::new(1.075, (step as f32 * 0.7).sin() * 0.3, 0.1);
        world.bodies[body].set_position(
            Pose::from_parts(
                rotation * local,
                rotation * Rotation::from_scaled_axis(Vector::X * (step as f32 * 0.3)),
            ),
            true,
        );
        world.step();
        let pair = world.narrow_phase.contact_pair(first, second).unwrap();
        assert!(!pair.has_any_active_contact());
        assert!(
            (pair.recycle_state.unwrap().pos12.translation - local).length() < 1.0e-5,
            "certified empty update must advance the ordinary reference at step {step}"
        );
        #[cfg(feature = "profiler")]
        assert_eq!(world.narrow_phase.contact_update_counts()[2], 1);
    }
    // A move beyond the ordinary reuse interval must compute contacts when the face closes.
    world.bodies[body].set_position(
        Pose::from_parts(rotation * Vector::X * 1.014, rotation),
        true,
    );
    world.step();
    assert!(
        world
            .narrow_phase
            .contact_pair(first, second)
            .unwrap()
            .has_any_active_contact()
    );
    #[cfg(feature = "profiler")]
    assert_eq!(world.narrow_phase.contact_update_counts()[9], 1);
}

#[test]
fn uncertain_face_proofs_fall_back() {
    let cube = Cuboid::new(Vector::splat(0.5));
    let pose = Pose::translation(1.02, 0.0, 0.0);
    assert!(!cuboid_face_separates(&cube, &cube, &pose, 0, 0.02));
    assert!(!cuboid_face_separates(&cube, &cube, &pose, 3, 0.0));
    assert!(!cuboid_face_separates(&cube, &cube, &pose, 0, Real::NAN));
    assert!(!cuboid_face_separates(&cube, &cube, &pose, 0, -1.0));
    let invalid = Cuboid::new(Vector::new(Real::INFINITY, 0.5, 0.5));
    assert!(!cuboid_face_separates(&invalid, &cube, &pose, 0, 0.0));
}

#[test]
fn empty_pair_certificate_preserves_legacy_contact_onset_and_trajectory() {
    let (mut optimized, body, first, second) = separated_boxes();
    let (mut legacy, _, _, _) = separated_boxes();
    let rotation = optimized.colliders[first].rotation();
    for step in 0..300 {
        // The first three moves distinguish the legacy short interval from immediately
        // creating speculative contacts as soon as a previously separated face closes.
        let local = match step {
            0 => Vector::X * 1.025,
            1 => Vector::X * 1.014,
            2 => Vector::X,
            _ => Vector::new(
                1.04 + 0.09 * (step as Real * 0.3).sin(),
                0.3 * (step as Real * 0.7).sin(),
                0.1,
            ),
        };
        let rot = rotation
            * Rotation::from_scaled_axis(
                Vector::X
                    * if step < 3 {
                        0.0
                    } else {
                        (step as Real * 0.1).sin()
                    },
            );
        for world in [&mut optimized, &mut legacy] {
            world.bodies[body].set_position(Pose::from_parts(rotation * local, rot), true);
            world.bodies[body].set_linvel(Vector::ZERO, true);
            world.bodies[body].set_angvel(Vector::ZERO, true);
        }
        // Clearing only the new certificate uses the original pose-recycling/full-query
        // path as an independent oracle, including its contact and solver transitions.
        for edge in &mut legacy.narrow_phase.contact_graph.graph.edges {
            if let Some(state) = &mut edge.weight.recycle_state {
                state.separating_axis = None;
            }
        }
        optimized.step();
        legacy.step();
        let a = &optimized.bodies[body];
        let b = &legacy.bodies[body];
        assert_eq!(
            (a.position(), a.linvel(), a.angvel()),
            (b.position(), b.linvel(), b.angvel()),
            "step {step}"
        );
        let a = optimized.narrow_phase.contact_pair(first, second).unwrap();
        let b = legacy.narrow_phase.contact_pair(first, second).unwrap();
        assert_eq!(
            a.has_any_active_contact(),
            b.has_any_active_contact(),
            "step {step}"
        );
        if step == 1 {
            assert!(!a.has_any_active_contact());
        }
        if step == 2 {
            assert!(a.has_any_active_contact());
        }
    }
}

#[test]
#[cfg(feature = "profiler")]
fn update_counts_reduce_parallel_blocks_and_reset_each_pass() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::Y * -0.5),
        ColliderBuilder::cuboid(100.0, 0.5, 100.0),
    );
    // More than one parallel dispatch block, with exactly one floor pair per body.
    for i in 0..128 {
        world.insert(
            RigidBodyBuilder::dynamic()
                .can_sleep(false)
                .translation(Vector::new(
                    (i % 16) as Real * 2.0,
                    0.51,
                    (i / 16) as Real * 2.0,
                )),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
    }
    world.step();
    let counts = world.narrow_phase.contact_update_counts();
    assert_eq!(counts.iter().sum::<u64>(), 128);
    assert_eq!(counts[4], 128, "new pairs need an initial full update");
    world.step();
    let counts = world.narrow_phase.contact_update_counts();
    assert_eq!(counts.iter().sum::<u64>(), 128);
    assert_eq!(counts[1], 128, "second pass reports only its pose reuses");
}

#[test]
fn empty_pair_certificate_invalidates_on_prediction_skin_shape_and_soft_ccd_edits() {
    for edit in 0..4 {
        let (mut world, body, first, second) = separated_boxes();
        let axis = world.colliders[first].rotation() * Vector::X;
        match edit {
            0 => world.integration_parameters.normalized_prediction_distance = 0.1,
            1 => world.colliders[second].set_contact_skin(0.1),
            2 => world.colliders[second].set_shape(SharedShape::cuboid(0.6, 0.5, 0.5)),
            _ => {
                world.bodies[body].set_soft_ccd_prediction(1.0);
                world.bodies[body].set_linvel(-axis * 6.0, true);
            }
        }
        world.step();
        let pair = world.narrow_phase.contact_pair(first, second).unwrap();
        assert!(
            pair.has_any_active_contact(),
            "edit {edit} kept an obsolete empty pair"
        );
        assert!(pair.recycle_state.unwrap().separating_axis.is_none());
    }
}

#[test]
fn empty_pair_certificate_never_skips_contact_hooks() {
    use crate::pipeline::{ActiveHooks, PairFilterContext, PhysicsHooks};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Hooks(AtomicUsize);
    impl PhysicsHooks for Hooks {
        fn filter_contact_pair(&self, _: &PairFilterContext) -> Option<SolverFlags> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Some(SolverFlags::default())
        }
    }
    let (mut world, body, first, second) = separated_boxes();
    let axis = world.colliders[first].rotation() * Vector::X;
    let hooks = Hooks(AtomicUsize::new(0));
    world.colliders[second].set_active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS);
    for step in 1..=3 {
        world.bodies[body].set_translation(axis * (1.075 - step as f32 * 0.003), true);
        world.step_with_events(&hooks, &());
        assert!(
            !world
                .narrow_phase
                .contact_pair(first, second)
                .unwrap()
                .has_any_active_contact()
        );
    }
    assert_eq!(hooks.0.load(Ordering::Relaxed), 3);
}

#[test]
#[cfg(feature = "serde-serialize")]
fn empty_pair_certificate_survives_snapshot_restore() {
    let (mut original, body, first, second) = separated_boxes();
    let axis = original.colliders[first].rotation() * Vector::X;
    let bytes = bincode::serialize(&original).unwrap();
    let mut restored: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
    for step in 1..=7 {
        for world in [&mut original, &mut restored] {
            world.bodies[body].set_translation(axis * (1.075 - step as f32 * 0.008), true);
            world.step();
        }
        assert_eq!(
            original.bodies[body].position(),
            restored.bodies[body].position()
        );
        let a = original.narrow_phase.contact_pair(first, second).unwrap();
        let b = restored.narrow_phase.contact_pair(first, second).unwrap();
        assert_eq!(a.has_any_active_contact(), b.has_any_active_contact());
        assert_eq!(
            a.recycle_state.unwrap().pos12,
            b.recycle_state.unwrap().pos12
        );
        assert_eq!(
            a.recycle_state.unwrap().separating_axis,
            b.recycle_state.unwrap().separating_axis
        );
    }
}
