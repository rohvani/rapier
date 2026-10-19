//! Compare warm filtering against the full predicate across actual pipeline transitions.
use crate::alloc_prelude::*;
use crate::prelude::*;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Default)]
struct Events(Mutex<Vec<(ColliderHandle, ColliderHandle, bool, bool, bool)>>);
impl EventHandler for Events {
    fn handle_collision_event(
        &self,
        _: &RigidBodySet,
        _: &ColliderSet,
        event: CollisionEvent,
        _: Option<&ContactPair>,
    ) {
        self.0.lock().unwrap().push((
            event.collider1(),
            event.collider2(),
            event.started(),
            event.sensor(),
            event.removed(),
        ));
    }
    fn handle_contact_force_event(
        &self,
        _: Real,
        _: &RigidBodySet,
        _: &ColliderSet,
        _: &ContactPair,
        _: Real,
    ) {
    }
}

fn world() -> (
    PhysicsWorld,
    RigidBodyHandle,
    ColliderHandle,
    RigidBodyHandle,
    ColliderHandle,
) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let (fixed, sensor) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::ball(2.0)
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS),
    );
    let (body, shape) = world.insert(RigidBodyBuilder::dynamic(), ColliderBuilder::ball(0.5));
    (world, fixed, sensor, body, shape)
}

#[test]
fn sensor_filter_cache_matches_cold_filter_after_lifecycle_edits() {
    let (mut warm, fixed, sensor, body, mut shape) = world();
    let (mut cold, other_fixed, other_sensor, other_body, other_shape) = world();
    assert_eq!(
        (fixed, sensor, body, shape),
        (other_fixed, other_sensor, other_body, other_shape)
    );
    let warm_events = Events::default();
    let cold_events = Events::default();
    for step in 0..30 {
        let mut reused = None;
        for world in [&mut warm, &mut cold] {
            match step {
                3 => world.colliders[sensor].set_collision_groups(InteractionGroups::none()),
                4 => world.colliders[sensor].set_collision_groups(InteractionGroups::all()),
                6 => world.bodies[body].set_body_type(RigidBodyType::Fixed, true),
                7 => world.bodies[body].set_body_type(RigidBodyType::Dynamic, true),
                9 => {
                    world.colliders[sensor]
                        .set_active_collision_types(ActiveCollisionTypes::empty());
                    world.colliders[shape]
                        .set_active_collision_types(ActiveCollisionTypes::empty());
                }
                10 => {
                    world.colliders[sensor].set_active_collision_types(ActiveCollisionTypes::all())
                }
                12 => world
                    .colliders
                    .set_parent(shape, Some(fixed), &mut world.bodies),
                13 => world
                    .colliders
                    .set_parent(shape, Some(body), &mut world.bodies),
                15 => world.bodies[body].set_enabled(false),
                16 => world.bodies[body].set_enabled(true),
                18 => {
                    world
                        .colliders
                        .remove(shape, &mut world.islands, &mut world.bodies, true)
                        .unwrap();
                    let fresh = world.colliders.insert_with_parent(
                        ColliderBuilder::ball(0.5),
                        body,
                        &mut world.bodies,
                    );
                    assert_eq!(fresh.into_raw_parts().0, shape.into_raw_parts().0);
                    assert_ne!(fresh, shape);
                    assert!(reused.is_none_or(|other| other == fresh));
                    reused = Some(fresh);
                }
                20 => world.colliders[sensor].set_sensor(false),
                21 => world.colliders[sensor].set_sensor(true),
                23 => world.bodies[body].set_translation(Vector::X * 10.0, true),
                24 => world.bodies[body].set_translation(Vector::ZERO, true),
                26 => world.colliders[sensor].set_enabled(false),
                27 => world.colliders[sensor].set_enabled(true),
                _ => {}
            }
        }
        if let Some(fresh) = reused {
            shape = fresh;
        }
        for edge in &mut cold.narrow_phase.intersection_graph.graph.edges {
            edge.weight.filter_valid = false;
        }
        warm.step_with_events(&(), &warm_events);
        cold.step_with_events(&(), &cold_events);
        assert_eq!(
            warm.narrow_phase.intersection_pair(sensor, shape),
            cold.narrow_phase.intersection_pair(sensor, shape),
            "step {step}"
        );
        assert_eq!(
            *warm_events.0.lock().unwrap(),
            *cold_events.0.lock().unwrap(),
            "events at step {step}"
        );
        #[cfg(feature = "serde-serialize")]
        {
            // Derived cache state is deliberately absent from snapshots. Restoring it cold
            // must preserve complete state bytes and the subsequent event/physics trajectory.
            let bytes = bincode::serialize(&warm).unwrap();
            assert_eq!(
                bytes,
                bincode::serialize(&cold).unwrap(),
                "state at step {step}"
            );
            if step == 11 {
                warm = bincode::deserialize(&bytes).unwrap();
            }
        }
    }
}

#[derive(Default)]
struct Filter {
    allow: AtomicBool,
    calls: AtomicUsize,
}
impl PhysicsHooks for Filter {
    fn filter_intersection_pair(&self, _: &PairFilterContext) -> bool {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.allow.load(Ordering::Relaxed)
    }
}

#[test]
fn sensor_filter_cache_does_not_cache_hooks_or_geometry() {
    let (mut world, _, sensor, body, shape) = world();
    let hook = Filter::default();
    let events = Events::default();
    for step in 0..10 {
        let hooked = step < 8;
        world.colliders[sensor].set_active_hooks(if hooked {
            ActiveHooks::FILTER_INTERSECTION_PAIR
        } else {
            ActiveHooks::empty()
        });
        hook.allow.store(step % 2 == 0, Ordering::Relaxed);
        world.bodies[body].wake_up(true);
        let before = hook.calls.load(Ordering::Relaxed);
        world.step_with_events(&hook, &events);
        assert_eq!(
            hook.calls.load(Ordering::Relaxed) - before,
            usize::from(hooked)
        );
        assert_eq!(
            world.narrow_phase.intersection_pair(sensor, shape),
            Some(!hooked || step % 2 == 0)
        );
    }
    world.bodies[body].set_translation(Vector::X * 10.0, true);
    world.step_with_events(&hook, &events);
    assert_ne!(
        world.narrow_phase.intersection_pair(sensor, shape),
        Some(true)
    );
    world.bodies[body].set_translation(Vector::ZERO, true);
    world.step_with_events(&hook, &events);
    assert_eq!(
        world.narrow_phase.intersection_pair(sensor, shape),
        Some(true)
    );
    assert!(events.0.lock().unwrap().iter().any(|event| event.2));
    assert!(events.0.lock().unwrap().iter().any(|event| !event.2));
}
