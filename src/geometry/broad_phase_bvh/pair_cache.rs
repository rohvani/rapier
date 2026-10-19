//! Bounded memoization of existing broad-phase pairs, with exact tags.
use crate::alloc_prelude::*;

const BUCKETS: usize = 1 << 14;
const WAYS: usize = 4;

/// A hit proves membership; collisions and replacement only discard that proof.
/// Generations live in the authoritative pair map. The caller must invalidate every
/// retired collider's tags before an arena slot can be rediscovered as a new generation.
#[derive(Clone, Default)]
pub(super) struct KnownPairCache {
    buckets: Vec<[u64; WAYS]>,
}

impl KnownPairCache {
    pub(super) fn allocated(&self) -> bool {
        !self.buckets.is_empty()
    }

    pub(super) fn allocate(&mut self) {
        self.buckets.resize(BUCKETS, [0; WAYS]);
    }

    #[inline]
    fn key(a: u32, b: u32) -> (u64, u64) {
        debug_assert_ne!(a, b);
        // A real pair has distinct indices, so zero is an unambiguous empty tag.
        let key = ((a.min(b) as u64) << 32) | a.max(b) as u64;
        let mixed = (key ^ key.rotate_left(21)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (key, mixed)
    }

    #[inline]
    pub(super) fn contains(&self, a: u32, b: u32) -> bool {
        if self.buckets.is_empty() {
            return false;
        }
        let (key, mixed) = Self::key(a, b);
        self.buckets[(mixed >> 50) as usize].contains(&key)
    }

    pub(super) fn insert(&mut self, a: u32, b: u32) {
        if self.buckets.is_empty() {
            return;
        }
        let (key, mixed) = Self::key(a, b);
        let bucket = &mut self.buckets[(mixed >> 50) as usize];
        if bucket.contains(&key) {
            return;
        }
        let slot = bucket
            .iter()
            .position(|tag| *tag == 0)
            .unwrap_or((mixed as usize) & (WAYS - 1));
        bucket[slot] = key;
    }

    pub(super) fn remove(&mut self, a: u32, b: u32) {
        if self.buckets.is_empty() {
            return;
        }
        let (key, mixed) = Self::key(a, b);
        let bucket = &mut self.buckets[(mixed >> 50) as usize];
        if let Some(slot) = bucket.iter().position(|tag| *tag == key) {
            bucket[slot] = 0;
        }
    }

    #[cfg(test)]
    pub(super) fn keys(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.buckets
            .iter()
            .flatten()
            .copied()
            .filter(|key| *key != 0)
            .map(|key| ((key >> 32) as u32, key as u32))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn tag_collisions_can_only_drop_membership_proofs() {
        let mut cache = KnownPairCache::default();
        assert!(!cache.contains(0, 1));
        cache.insert(0, 1);
        assert!(!cache.allocated());
        cache.allocate();
        let mut live = BTreeSet::new();
        for i in 1..100_000u32 {
            let a = i / 17;
            let b = i + 7;
            live.insert((a, b));
            cache.insert(b, a);
            assert!(cache.contains(a, b));
            if i % 7 == 0 {
                cache.remove(b, a);
                live.remove(&(a, b));
                assert!(!cache.contains(a, b));
            }
        }
        assert!(cache.keys().all(|key| live.contains(&key)));
        assert_eq!(cache.keys().count(), cache.clone().keys().count());
        for (a, b) in live {
            cache.remove(a, b);
        }
        assert_eq!(cache.keys().count(), 0);
    }
}
