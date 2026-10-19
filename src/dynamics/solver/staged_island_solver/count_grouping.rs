//! Pack body-disjoint contacts by row count without changing color/solve order.
use crate::alloc_prelude::*;
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::ContactRef;
use crate::math::{MAX_MANIFOLD_POINTS, SIMD_WIDTH};

/// Append a count-grouped copy only if it saves SIMD row iterations. The caller
/// must supply a body-disjoint color; overflow contacts cannot be reordered this way.
/// All copies finish before chunk segments borrow their storage.
pub(super) fn append_grouped(
    refs: &[ContactRef],
    store: &ManifoldStore,
    counts: &mut Vec<u8>,
    output: &mut Vec<ContactRef>,
) -> bool {
    if SIMD_WIDTH == 1 || refs.len() <= SIMD_WIDTH {
        return false;
    }
    counts.clear();
    counts.extend(refs.iter().map(|r| {
        store
            .get(*r)
            .data
            .num_active_contacts()
            .min(MAX_MANIFOLD_POINTS) as u8
    }));
    append_by_counts(refs, counts, output)
}

fn append_by_counts(refs: &[ContactRef], counts: &[u8], output: &mut Vec<ContactRef>) -> bool {
    let mut histogram = [0usize; MAX_MANIFOLD_POINTS + 1];
    for &count in counts {
        debug_assert!(count != 0 && count as usize <= MAX_MANIFOLD_POINTS);
        histogram[count as usize] += 1;
    }
    let original_rows: usize = counts
        .chunks(SIMD_WIDTH)
        .map(|chunk| *chunk.iter().max().unwrap() as usize)
        .sum();
    let mut sorted_rows = 0;
    let mut seen = 0;
    let mut starts = [0; MAX_MANIFOLD_POINTS + 1];
    // A chunk costs its maximum row count. Descending order packs each large
    // count into as few chunks as possible; any partial tail contains the
    // smallest counts. Constraints within the color do not share writable bodies.
    for count in (1..=MAX_MANIFOLD_POINTS).rev() {
        starts[count] = output.len() + seen;
        let first_new_chunk = seen.div_ceil(SIMD_WIDTH);
        let after = seen + histogram[count];
        sorted_rows += (after.div_ceil(SIMD_WIDTH) - first_new_chunk) * count;
        seen = after;
    }
    // Uniform/already well-packed colors keep their existing zero-copy layout.
    if sorted_rows >= original_rows {
        return false;
    }
    debug_assert_eq!(seen, refs.len());
    output.resize(output.len() + refs.len(), ContactRef::PADDING);
    for (&r, &count) in refs.iter().zip(counts) {
        output[starts[count as usize]] = r;
        starts[count as usize] += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(counts: &[u8]) -> usize {
        counts
            .chunks(SIMD_WIDTH)
            .map(|c| *c.iter().max().unwrap() as usize)
            .sum()
    }

    #[test]
    fn grouping_preserves_refs_and_matches_sorted_row_cost() {
        // Vary counts, partial tails, repeated colors, and pre-existing output.
        // The oracle sorts a separate count array, independently of the counting pass.
        let mut rng = oorandom::Rand32::new(0x1875_9ab2);
        for len in 1..=257 {
            let counts: Vec<u8> = (0..len)
                .map(|_| 1 + (rng.rand_u32() as usize % MAX_MANIFOLD_POINTS) as u8)
                .collect();
            let refs: Vec<_> = (0..len)
                .map(|i| ContactRef {
                    edge: i as u32,
                    manifold: 7,
                })
                .collect();
            let mut oracle = counts.clone();
            oracle.sort_unstable_by(|a, b| b.cmp(a));
            let mut output = alloc::vec![ContactRef::PADDING; 3];
            let grouped = append_by_counts(&refs, &counts, &mut output);
            assert_eq!(grouped, cost(&oracle) < cost(&counts));
            assert_eq!(&output[..3], &[ContactRef::PADDING; 3]);
            if grouped {
                let reordered: Vec<_> = output[3..]
                    .iter()
                    .map(|r| counts[r.edge as usize])
                    .collect();
                assert_eq!(reordered, oracle);
                let mut restored = output[3..].to_vec();
                restored.sort_unstable_by_key(|r| r.edge);
                assert_eq!(restored, refs);
            } else {
                assert_eq!(output.len(), 3);
            }
        }
    }
}
