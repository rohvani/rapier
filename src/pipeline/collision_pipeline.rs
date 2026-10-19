//! Physics pipeline structures.

use crate::alloc_prelude::*;

use crate::dynamics::{
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyJointSet, RigidBodyChanges,
};
use crate::geometry::{
    BroadPhaseBvh, BroadPhasePairEvent, ColliderChanges, ColliderHandle, ModifiedColliders,
    NarrowPhase,
};
use crate::math::Real;
use crate::pipeline::{EventHandler, PhysicsHooks};
use crate::prelude::ModifiedRigidBodies;
use crate::{dynamics::RigidBodySet, geometry::ColliderSet};

/// A collision detection pipeline that can be used without full physics simulation.
///
/// This runs only collision detection (broad-phase + narrow-phase) without dynamics/forces.
/// Use when you want to detect collisions but don't need physics simulation.
///
/// **For full physics**, use [`PhysicsPipeline`](crate::pipeline::PhysicsPipeline) instead which includes this internally.
///
/// ## Use cases
///
/// - Collision detection in a non-physics game
/// - Custom physics integration where you handle forces yourself
/// - Debugging collision detection separately from dynamics
///
/// Like PhysicsPipeline, this only holds temporary buffers. Reuse the same instance for performance.
// NOTE: this contains only workspace data, so there is no point in making this serializable.
pub struct CollisionPipeline {
    broad_phase_events: Vec<BroadPhasePairEvent>,
}

#[allow(dead_code)]
fn check_pipeline_send_sync() {
    fn do_test<T: Sync>() {}
    do_test::<CollisionPipeline>();
}

impl Default for CollisionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CollisionPipeline {
    /// Initializes a new physics pipeline.
    pub fn new() -> CollisionPipeline {
        CollisionPipeline {
            broad_phase_events: Vec::new(),
        }
    }

    fn detect_collisions(
        &mut self,
        prediction_distance: Real,
        islands: &mut IslandManager,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        modified_colliders: &[ColliderHandle],
        removed_colliders: &[ColliderHandle],
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
        handle_user_changes: bool,
    ) {
        // Update broad-phase.
        self.broad_phase_events.clear();

        let params = IntegrationParameters {
            normalized_prediction_distance: prediction_distance,
            dt: 0.0,
            ..Default::default()
        };

        broad_phase.update(
            &params,
            colliders,
            bodies,
            modified_colliders,
            removed_colliders,
            &mut self.broad_phase_events,
        );

        // Update narrow-phase.
        if handle_user_changes {
            narrow_phase.handle_user_changes(
                None,
                modified_colliders,
                removed_colliders,
                colliders,
                bodies,
                events,
            );
        }

        narrow_phase.register_pairs(None, colliders, bodies, &self.broad_phase_events, events);
        narrow_phase.compute_contacts(
            prediction_distance,
            0.0,
            false,
            0.0,
            islands,
            bodies,
            colliders,
            &ImpulseJointSet::new(),
            &MultibodyJointSet::new(),
            modified_colliders,
            hooks,
            events,
        );
        narrow_phase.compute_intersections(
            islands,
            bodies,
            colliders,
            modified_colliders,
            hooks,
            events,
        );
    }

    fn clear_modified_colliders(
        &mut self,
        colliders: &mut ColliderSet,
        modified_colliders: &mut ModifiedColliders,
    ) {
        for handle in modified_colliders.iter() {
            if let Some(co) = colliders.get_mut_internal(*handle) {
                co.changes = ColliderChanges::empty();
            }
        }

        modified_colliders.clear();
    }

    fn clear_modified_bodies(
        &mut self,
        bodies: &mut RigidBodySet,
        modified_bodies: &mut ModifiedRigidBodies,
    ) {
        // Without this, a body modified by the user keeps its MODIFIED flag forever, so
        // `RigidBodySet::get_mut` never re-inserts it into the modified set and later user
        // changes stop propagating to its colliders (same as `PhysicsPipeline`).
        for handle in modified_bodies.iter() {
            if let Some(rb) = bodies.get_mut_internal(*handle) {
                rb.changes = RigidBodyChanges::empty();
            }
        }

        modified_bodies.clear();
    }

    /// Executes one step of the collision detection.
    pub fn step(
        &mut self,
        prediction_distance: Real,
        islands: &mut IslandManager,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
    ) {
        self.step_impl(
            prediction_distance,
            islands,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            hooks,
            events,
            false,
        );
    }

    /// Refresh collision queries after accepting the pending targets of modified kinematic bodies.
    ///
    /// This matches calling `set_position(*body.next_position(), false)` on each already-modified
    /// kinematic body before [`Self::step`]. It does not integrate dynamics or advance time.
    /// Targets whose modification queue was consumed by an earlier step remain untouched; callers
    /// switching from dynamics must reconcile those targets separately on that transition.
    pub fn step_with_kinematic_targets(
        &mut self,
        prediction_distance: Real,
        islands: &mut IslandManager,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
    ) {
        self.step_impl(
            prediction_distance,
            islands,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            hooks,
            events,
            true,
        );
    }

    fn step_impl(
        &mut self,
        prediction_distance: Real,
        islands: &mut IslandManager,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
        accept_kinematic_targets: bool,
    ) {
        // Collider edits can append parents to the body frontier. They must not cause acceptance
        // of a target that was not pending at entry, so capture the original prefix first.
        let kinematic_target_prefix = if accept_kinematic_targets {
            bodies.modified_bodies.len()
        } else {
            0
        };
        let mut modified_colliders = colliders.take_modified();
        let mut removed_colliders = colliders.take_removed();

        super::user_changes::handle_user_changes_to_colliders(
            bodies,
            colliders,
            &modified_colliders[..],
        );
        // Collider edits can queue their parents for mass-property/activation updates. Drain
        // bodies afterward, as PhysicsPipeline does, so these new entries belong to this pass
        // instead of being overwritten when its original body worklist is returned below.
        let mut modified_bodies = bodies.take_modified();
        super::user_changes::handle_user_changes_to_rigid_bodies(
            None,
            bodies,
            colliders,
            &mut ImpulseJointSet::new(),
            &mut MultibodyJointSet::new(),
            &modified_bodies,
            &mut modified_colliders,
            kinematic_target_prefix,
        );

        // Disabled colliders are treated as if they were removed.
        removed_colliders.extend(
            modified_colliders
                .iter()
                .copied()
                .filter(|h| colliders.get(*h).map(|c| !c.is_enabled()).unwrap_or(false)),
        );

        self.detect_collisions(
            prediction_distance,
            islands,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            &modified_colliders[..],
            &removed_colliders,
            hooks,
            events,
            true,
        );

        self.clear_modified_colliders(colliders, &mut modified_colliders);
        self.clear_modified_bodies(bodies, &mut modified_bodies);
        removed_colliders.clear();
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serde-serialize")]
    use crate::alloc_prelude::Vec;

    #[cfg(feature = "serde-serialize")]
    fn target_query_step(world: &mut crate::prelude::PhysicsWorld, fused: bool) {
        use crate::prelude::*;
        let mut pipeline = CollisionPipeline::new();
        if !fused {
            let targets: Vec<_> = world
                .bodies
                .iter_modified()
                .filter_map(|(handle, body)| {
                    (body.is_kinematic() && body.position() != body.next_position())
                        .then_some((handle, *body.next_position()))
                })
                .collect();
            for (handle, pose) in targets {
                world.bodies[handle].set_position(pose, false);
            }
        }
        let step = if fused {
            CollisionPipeline::step_with_kinematic_targets
        } else {
            CollisionPipeline::step
        };
        step(
            &mut pipeline,
            0.01,
            &mut world.islands,
            &mut world.broad_phase,
            &mut world.narrow_phase,
            &mut world.bodies,
            &mut world.colliders,
            &(),
            &(),
        );
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn fused_targets_match_manual_commit_across_topology_and_mass_edits() {
        use crate::prelude::*;
        let mut seed = PhysicsWorld::new();
        let kinds = [
            RigidBodyType::Dynamic,
            RigidBodyType::Fixed,
            RigidBodyType::KinematicPositionBased,
            RigidBodyType::KinematicVelocityBased,
        ];
        let handles: Vec<_> = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| {
                seed.insert(
                    RigidBodyBuilder::new(kind).translation(Vector::splat(i as Real * 3.0)),
                    ColliderBuilder::ball(0.7)
                        .translation(Vector::Y * 0.3)
                        .density(2.0),
                )
            })
            .collect();
        // Deserialize identical arenas/maps, including the same pending modification order.
        let bytes = bincode::serialize(&seed).unwrap();
        let mut control: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
        let mut candidate: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
        for frame in 0..8 {
            for world in [&mut control, &mut candidate] {
                for (index, &(body, collider)) in handles.iter().enumerate() {
                    let translation =
                        Vector::X * (frame as Real * 0.1 + index as Real) + Vector::Y * 0.25;
                    #[cfg(feature = "dim2")]
                    let target = Pose::new(translation, frame as Real * 0.17);
                    #[cfg(feature = "dim3")]
                    let target =
                        Pose::new(translation, Vector::new(0.11, 0.17, -0.09) * frame as Real);
                    world.bodies[body].set_next_kinematic_position(target);
                    if frame == 2 {
                        world.colliders[collider].set_density(4.0);
                    }
                    if frame == 3 {
                        world.bodies[body].set_enabled(false);
                    }
                    if frame == 4 {
                        world.bodies[body].set_enabled(true);
                        world.bodies[body].set_locked_axes(LockedAxes::ROTATION_LOCKED_Z, false);
                        world.bodies[body].set_body_type(kinds[(index + 1) % kinds.len()], false);
                    }
                }
                if frame == 5 {
                    world
                        .colliders
                        .set_parent(handles[0].1, Some(handles[2].0), &mut world.bodies);
                }
                if frame == 6 {
                    // Removed entries may remain in the prefix; slot reuse cannot admit them.
                    let dead = world
                        .bodies
                        .insert(RigidBodyBuilder::kinematic_position_based());
                    world.bodies.remove(
                        dead,
                        &mut world.islands,
                        &mut world.colliders,
                        &mut world.impulse_joints,
                        &mut world.multibody_joints,
                        true,
                    );
                    let replacement = world
                        .bodies
                        .insert(RigidBodyBuilder::kinematic_position_based());
                    world.bodies[replacement].set_next_kinematic_position(Pose::from_translation(
                        Vector::X + Vector::Y * 2.0,
                    ));
                }
            }
            target_query_step(&mut control, false);
            target_query_step(&mut candidate, true);
            assert_eq!(
                bincode::serialize(&control).unwrap(),
                bincode::serialize(&candidate).unwrap(),
                "fused target state changed at frame {frame}"
            );
        }
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn collider_only_parent_admission_does_not_accept_an_old_target() {
        use crate::prelude::*;
        let mut seed = PhysicsWorld::new();
        let (body, collider) = seed.insert(
            RigidBodyBuilder::kinematic_position_based(),
            ColliderBuilder::ball(0.7).density(2.0),
        );
        let target = Pose::from_translation(Vector::X * 2.0 + Vector::Y * 3.0);
        seed.bodies[body].set_next_kinematic_position(target);
        // Ordinary collision stepping consumes the queue without committing a target.
        CollisionPipeline::new().step(
            0.01,
            &mut seed.islands,
            &mut seed.broad_phase,
            &mut seed.narrow_phase,
            &mut seed.bodies,
            &mut seed.colliders,
            &(),
            &(),
        );
        assert_ne!(
            seed.bodies[body].position(),
            seed.bodies[body].next_position()
        );
        assert_eq!(seed.bodies.iter_modified().count(), 0);
        seed.colliders[collider].set_density(4.0);
        let bytes = bincode::serialize(&seed).unwrap();
        let mut control: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
        let mut candidate: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
        target_query_step(&mut control, false);
        target_query_step(&mut candidate, true);
        assert_ne!(*candidate.bodies[body].position(), target);
        assert_eq!(
            bincode::serialize(&control).unwrap(),
            bincode::serialize(&candidate).unwrap()
        );
        // A subsequent explicit body edit does make that retained target eligible.
        control.bodies.get_mut(body).unwrap();
        candidate.bodies.get_mut(body).unwrap();
        target_query_step(&mut control, false);
        target_query_step(&mut candidate, true);
        assert_eq!(*candidate.bodies[body].position(), target);
        assert_eq!(
            bincode::serialize(&control).unwrap(),
            bincode::serialize(&candidate).unwrap()
        );
    }
    #[test]
    fn collider_edits_refresh_parent_mass_in_the_same_collision_only_pass() {
        use crate::prelude::*;

        let scene = || {
            let mut world = PhysicsWorld::new();
            world.gravity = Vector::ZERO;
            let handles = world.insert(
                RigidBodyBuilder::dynamic(),
                ColliderBuilder::new(SharedShape::new(Cuboid::new(Vector::ONE))).density(2.0),
            );
            world.step();
            (world, handles)
        };
        let (mut collision_world, (body, collider)) = scene();
        let (mut physics_world, reference_handles) = scene();
        assert_eq!((body, collider), reference_handles);
        let mut collision_pipeline = CollisionPipeline::new();
        // A collider edit queues its parent while collider changes are processed. Repeated
        // edits also detect a lost queue entry whose IN_MODIFIED_SET flag remains stranded.
        for half in [2.0, 3.0, 1.0] {
            for world in [&mut collision_world, &mut physics_world] {
                world.colliders[collider]
                    .set_shape(SharedShape::new(Cuboid::new(Vector::splat(half))));
            }
            physics_world.step();
            collision_pipeline.step(
                0.01,
                &mut collision_world.islands,
                &mut collision_world.broad_phase,
                &mut collision_world.narrow_phase,
                &mut collision_world.bodies,
                &mut collision_world.colliders,
                &(),
                &(),
            );
            let expected = 2.0 * (2.0 * half).powi(DIM as i32);
            assert_eq!(physics_world.bodies[body].mass(), expected);
            assert_eq!(collision_world.bodies[body].mass(), expected);
            assert_eq!(
                collision_world.bodies[body].mass_properties().local_mprops,
                physics_world.bodies[body].mass_properties().local_mprops
            );
        }
    }

    #[test]
    #[cfg(feature = "dim3")]
    pub fn test_no_rigid_bodies() {
        use crate::prelude::*;
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        /* Create the ground. */
        let collider_a = ColliderBuilder::cuboid(1.0, 1.0, 1.0)
            .active_collision_types(ActiveCollisionTypes::all())
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();

        let a_handle = collider_set.insert(collider_a);

        let collider_b = ColliderBuilder::cuboid(1.0, 1.0, 1.0)
            .active_collision_types(ActiveCollisionTypes::all())
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();

        let _ = collider_set.insert(collider_b);

        let integration_parameters = IntegrationParameters::default();
        let mut islands = IslandManager::new();
        let mut broad_phase = BroadPhaseBvh::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut collision_pipeline = CollisionPipeline::new();
        let physics_hooks = ();

        collision_pipeline.step(
            integration_parameters.prediction_distance(),
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &physics_hooks,
            &(),
        );

        let mut hit = false;

        for (_, _, intersecting) in narrow_phase.intersection_pairs_with(a_handle) {
            if intersecting {
                hit = true;
            }
        }

        assert!(hit, "No hit found");
    }

    #[test]
    #[cfg(feature = "dim2")]
    pub fn test_no_rigid_bodies() {
        use crate::prelude::*;
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        /* Create the ground. */
        let collider_a = ColliderBuilder::cuboid(1.0, 1.0)
            .active_collision_types(ActiveCollisionTypes::all())
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();

        let a_handle = collider_set.insert(collider_a);

        let collider_b = ColliderBuilder::cuboid(1.0, 1.0)
            .active_collision_types(ActiveCollisionTypes::all())
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();

        let _ = collider_set.insert(collider_b);

        let integration_parameters = IntegrationParameters::default();
        let mut islands = IslandManager::new();
        let mut broad_phase = BroadPhaseBvh::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut collision_pipeline = CollisionPipeline::new();
        let physics_hooks = ();

        collision_pipeline.step(
            integration_parameters.prediction_distance(),
            &mut islands,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &physics_hooks,
            &(),
        );

        let mut hit = false;

        for (_, _, intersecting) in narrow_phase.intersection_pairs_with(a_handle) {
            if intersecting {
                hit = true;
            }
        }

        assert!(hit, "No hit found");
    }
}
