//! Bounded sharing of bit-identical builtin cuboid geometry.
use crate::alloc_prelude::*;
use crate::data::arena::Arena;
use crate::geometry::{Collider, SharedShape};
use crate::math::{DIM, Vector};

const CAPACITY: usize = 32;

#[derive(Clone, Debug)]
pub(crate) struct CuboidShapeCache {
    entries: Vec<([u64; DIM], SharedShape)>,
    next: usize,
    initialized: bool,
}

impl Default for CuboidShapeCache {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next: 0,
            initialized: true,
        }
    }
}

impl CuboidShapeCache {
    #[cfg(feature = "serde-serialize")]
    pub(crate) fn cold() -> Self {
        Self {
            initialized: false,
            ..Self::default()
        }
    }

    pub(crate) fn restore(&mut self, colliders: &mut Arena<Collider>) {
        if !self.initialized {
            // Serialization preserves geometry, not Arc identity. Rebuild sharing once without
            // marking any collider changed: every replaced shape has identical type and bits.
            for (_, collider) in colliders.iter_mut() {
                self.share(&mut collider.shape);
            }
            self.initialized = true;
        }
    }

    pub(crate) fn share(&mut self, shape: &mut SharedShape) {
        // as_cuboid downcasts the concrete builtin type; custom shapes retain their identity.
        let Some(cuboid) = shape.as_cuboid() else {
            return;
        };
        let half = cuboid.half_extents;
        if !half.is_finite() || !half.cmpge(Vector::ZERO).all() {
            return;
        }
        // Bit keys preserve signed zero and never quantize nearby dimensions. The u64 storage
        // accommodates both engine precisions; f32's bits occupy only its low half.
        let key = half.to_array().map(|value| value.to_bits() as u64);
        if let Some((_, existing)) = self.entries.iter().find(|(old, _)| *old == key) {
            *shape = existing.clone();
            return;
        }
        let entry = (key, shape.clone());
        if self.entries.len() < CAPACITY {
            self.entries.push(entry);
        } else {
            // Eviction drops only this cache's reference. Live colliders retain their own Arc,
            // and Collider::shape_mut continues to detach before changing shared geometry.
            self.entries[self.next] = entry;
            self.next = (self.next + 1) % CAPACITY;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::geometry::{Ball, ColliderBuilder, ColliderSet, Cuboid};
    use crate::math::Real;

    fn cube(half: Vector) -> SharedShape {
        SharedShape::new(Cuboid::new(half))
    }
    fn same(a: &SharedShape, b: &SharedShape) -> bool {
        core::ptr::eq(a.as_cuboid().unwrap(), b.as_cuboid().unwrap())
    }

    #[test]
    fn shares_exact_geometry_but_not_nearby_dimensions_or_other_shapes() {
        let mut cache = CuboidShapeCache::default();
        let mut first = cube(Vector::ONE);
        let mut second = cube(Vector::ONE);
        assert!(!same(&first, &second));
        cache.share(&mut first);
        cache.share(&mut second);
        assert!(same(&first, &second));
        let mut nearby = Vector::ONE;
        nearby.x = Real::from_bits((1.0 as Real).to_bits() + 1);
        let mut different = cube(nearby);
        cache.share(&mut different);
        assert!(!same(&first, &different));
        let mut ball = SharedShape::new(Ball::new(1.0));
        let before = ball.clone();
        cache.share(&mut ball);
        assert!(core::ptr::eq(
            ball.as_ball().unwrap(),
            before.as_ball().unwrap()
        ));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn preserves_signed_zero_and_rejects_nonfinite_cache_keys() {
        let mut cache = CuboidShapeCache::default();
        let mut positive = cube(Vector::ZERO);
        let mut half = Vector::ZERO;
        half.x = -0.0;
        let mut negative = cube(half);
        cache.share(&mut positive);
        cache.share(&mut negative);
        assert!(!same(&positive, &negative));
        for value in [Real::NAN, Real::INFINITY, -1.0] {
            let mut invalid = SharedShape::new(Cuboid {
                half_extents: Vector::splat(value),
            });
            let before = invalid.clone();
            cache.share(&mut invalid);
            assert!(same(&invalid, &before));
        }
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn eviction_and_copy_on_write_preserve_independent_colliders() {
        let mut colliders = ColliderSet::new();
        let first = colliders.insert(ColliderBuilder::new(cube(Vector::ONE)));
        let second = colliders.insert(ColliderBuilder::new(cube(Vector::ONE)));
        assert!(same(
            colliders[first].shared_shape(),
            colliders[second].shared_shape()
        ));
        colliders[first]
            .shape_mut()
            .as_cuboid_mut()
            .unwrap()
            .half_extents
            .x = 2.0;
        assert_eq!(
            colliders[second].shape().as_cuboid().unwrap().half_extents,
            Vector::ONE
        );
        assert_eq!(
            colliders[first].shape().as_cuboid().unwrap().half_extents.x,
            2.0
        );
        for index in 2..CAPACITY * 4 {
            colliders.insert(ColliderBuilder::new(cube(Vector::splat(index as Real))));
        }
        assert_eq!(colliders.cuboid_shapes.entries.len(), CAPACITY);
        assert_eq!(
            colliders[second].shape().as_cuboid().unwrap().half_extents,
            Vector::ONE
        );
        assert_eq!(
            colliders[first].shape().as_cuboid().unwrap().half_extents.x,
            2.0
        );
    }

    #[test]
    fn both_pipelines_share_shape_edits_and_refresh_parent_mass() {
        use crate::dynamics::RigidBodyBuilder;
        use crate::pipeline::{CollisionPipeline, PhysicsWorld};
        for collision_only in [false, true] {
            let mut world = PhysicsWorld::new();
            world.gravity = Vector::ZERO;
            let (body, first) = world.insert(
                RigidBodyBuilder::dynamic().translation(Vector::splat(20.0)),
                ColliderBuilder::new(cube(Vector::ONE)).density(2.0),
            );
            let second = world
                .colliders
                .insert(ColliderBuilder::new(cube(Vector::ONE * 2.0)));
            world.step();
            world.colliders[first].set_shape(cube(Vector::ONE * 2.0));
            if collision_only {
                CollisionPipeline::new().step(
                    0.01,
                    &mut world.islands,
                    &mut world.broad_phase,
                    &mut world.narrow_phase,
                    &mut world.bodies,
                    &mut world.colliders,
                    &(),
                    &(),
                );
            } else {
                world.step();
            }
            assert!(same(
                world.colliders[first].shared_shape(),
                world.colliders[second].shared_shape()
            ));
            assert_eq!(
                world.bodies[body].mass(),
                2.0 * (4.0 as Real).powi(DIM as i32)
            );
        }
    }

    #[test]
    #[cfg(feature = "serde-serialize")]
    fn cold_restore_changes_only_allocation_sharing() {
        let mut colliders = ColliderSet::new();
        let first = colliders.insert(ColliderBuilder::new(cube(Vector::ONE)));
        let second = colliders.insert(ColliderBuilder::new(cube(Vector::ONE)));
        let before = bincode::serialize(&colliders).unwrap();
        let mut restored: ColliderSet = bincode::deserialize(&before).unwrap();
        assert!(!same(
            restored[first].shared_shape(),
            restored[second].shared_shape()
        ));
        let pending = restored.take_modified();
        restored.set_modified(pending);
        assert!(same(
            restored[first].shared_shape(),
            restored[second].shared_shape()
        ));
        assert_eq!(before, bincode::serialize(&restored).unwrap());
        let clone = restored.clone();
        assert_eq!(before, bincode::serialize(&clone).unwrap());
    }
}
