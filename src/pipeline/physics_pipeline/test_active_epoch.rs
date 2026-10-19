//! Cache validity under public body edits and real membership transitions.
use crate::alloc_prelude::*;
use crate::dynamics::{FixedJointBuilder, RigidBodyBuilder, RigidBodyType};
use crate::geometry::ColliderBuilder;
use crate::math::{Real, Vector};
use crate::pipeline::PhysicsWorld;

fn block() -> ColliderBuilder {
    #[cfg(feature = "dim2")]
    return ColliderBuilder::cuboid(0.5, 0.5);
    #[cfg(feature = "dim3")]
    return ColliderBuilder::cuboid(0.5, 0.5, 0.5);
}

fn check_indices(world: &PhysicsWorld) {
    let handles: Vec<_> = world.islands.active_bodies().collect();
    for (index, handle) in handles.iter().enumerate() {
        assert_eq!(world.bodies[*handle].ids.active_set_id as usize, index);
        assert!(!handles[..index].contains(handle));
    }
}

#[test]
fn active_body_value_edits_preserve_membership_while_contacts_refresh() {
    let mut world = PhysicsWorld::new();
    world.insert(RigidBodyBuilder::fixed(), block());
    let (a, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::Y)
            .can_sleep(false),
        block(),
    );
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::Y * 2.0)
            .can_sleep(false),
        block(),
    );
    for _ in 0..4 {
        world.step();
    }
    assert!(
        world
            .narrow_phase
            .contact_pairs()
            .any(|p| p.has_any_active_contact())
    );
    let epoch = world.islands.active_set_epoch;
    for step in 0..12 {
        let sign = if step % 2 == 0 { 1.0 } else { -1.0 };
        let body = world.bodies.get_mut(a).unwrap();
        let mut pose = *body.position();
        pose.translation += Vector::X * (sign * 0.0001);
        body.set_position(pose, true);
        body.set_linvel(Vector::X * (sign * 0.001), true);
        body.reset_forces(false);
        body.add_force(Vector::Y * 0.0001, false);
        body.set_additional_mass(0.1 + step as Real * 0.01, false);
        world.step();
        // The shadow validators compare the cached qualification and contact selection
        // against complete recomputation on every step, including these value-only edits.
        assert_eq!(world.islands.active_set_epoch, epoch);
        check_indices(&world);
    }
}

#[test]
fn type_enable_remove_reuse_sleep_and_wake_invalidate_active_membership() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let (a, _) = world.insert(RigidBodyBuilder::dynamic().can_sleep(false), block());
    world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::X * 4.0)
            .can_sleep(false),
        block(),
    );
    world.step();
    for kind in [
        RigidBodyType::KinematicPositionBased,
        RigidBodyType::KinematicVelocityBased,
        RigidBodyType::Dynamic,
        RigidBodyType::Fixed,
        RigidBodyType::Dynamic,
    ] {
        let before = world.islands.active_set_epoch;
        world.bodies.get_mut(a).unwrap().set_body_type(kind, true);
        world.step();
        assert_ne!(
            world.islands.active_set_epoch, before,
            "type change {kind:?}"
        );
        check_indices(&world);
    }
    for enabled in [false, true] {
        let before = world.islands.active_set_epoch;
        world.bodies.get_mut(a).unwrap().set_enabled(enabled);
        world.step();
        assert_ne!(world.islands.active_set_epoch, before);
        assert_eq!(world.islands.active_bodies().any(|h| h == a), enabled);
        check_indices(&world);
    }
    let before = world.islands.active_set_epoch;
    world.remove_body(a).unwrap();
    assert_ne!(world.islands.active_set_epoch, before);
    let (replacement, _) = world.insert(RigidBodyBuilder::dynamic().can_sleep(false), block());
    assert_eq!(replacement.into_raw_parts().0, a.into_raw_parts().0);
    assert_ne!(replacement, a);
    let before = world.islands.active_set_epoch;
    world.step();
    assert_ne!(world.islands.active_set_epoch, before);
    assert!(world.bodies.get(a).is_none());
    check_indices(&world);

    let (sleeper, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::X * 100.0),
        block(),
    );
    world.step();
    let before = world.islands.active_set_epoch;
    for _ in 0..120 {
        world.step();
        if !world.islands.active_bodies().any(|h| h == sleeper) {
            break;
        }
    }
    assert!(world.bodies[sleeper].is_sleeping());
    assert!(!world.islands.active_bodies().any(|h| h == sleeper));
    assert_ne!(world.islands.active_set_epoch, before);
    let before = world.islands.active_set_epoch;
    world
        .bodies
        .get_mut(sleeper)
        .unwrap()
        .set_linvel(Vector::X * 0.1, true);
    world.step();
    assert_ne!(world.islands.active_set_epoch, before);
    assert!(world.islands.active_bodies().any(|h| h == sleeper));
    check_indices(&world);
}

#[test]
fn jointed_body_edits_invalidate_joint_assembly_without_reindexing_bodies() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let (a, _) = world.insert(RigidBodyBuilder::dynamic().can_sleep(false), block());
    let (b, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::X)
            .can_sleep(false),
        block(),
    );
    world.impulse_joints.insert(
        a,
        b,
        FixedJointBuilder::new()
            .local_anchor1(Vector::X * 0.5)
            .local_anchor2(Vector::X * -0.5),
        true,
    );
    world.step();
    let active_epoch = world.islands.active_set_epoch;
    let joint_epoch = world.impulse_joints.assembly_epoch;
    world
        .bodies
        .get_mut(a)
        .unwrap()
        .set_additional_mass(2.0, true);
    world.step();
    assert_eq!(world.islands.active_set_epoch, active_epoch);
    assert_ne!(world.impulse_joints.assembly_epoch, joint_epoch);
    check_indices(&world);
    assert!(
        (world.bodies[b].translation() - world.bodies[a].translation() - Vector::X).length() < 0.01
    );
}
