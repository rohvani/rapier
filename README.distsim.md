# distsim engine changes

This branch starts at `3e12c2679cb1940a876bde93af9cec0cf2f57944`, the source revision
recorded in the published `rapier3d` 0.35.1 crate. All 212 packaged source and test
files match that revision byte for byte. The archive SHA-256 is
`e1c797fd96e34ae075c7b849741f89968ebedb86b584023420d68a1993bdc5cc`.

The fork's workflow files are retained verbatim from its existing `master`
(`475caa3ff4646fd9a796b67d14116256f4962bc7`). Publishing this older engine branch
does not replace the fork's workflow definitions. Those definitions target the
newer master layout; this branch's validation is the bounded distsim validation
described below, not a claim that the master's binding jobs run on this branch.

The changes preserve the engine's existing collision and solver policies:

- `broad_phase_bvh` omits pair enumeration when all enabled colliders have empty
  `ActiveCollisionTypes`. Bounds, removal, stale-pair cleanup, and query updates
  still run. The proof is maintained from modified colliders and starts
  conservatively after deserialization. `set_active_collision_types` now marks
  filter changes, so stationary pairs reactivate correctly. Fixtures exercise
  movement, removal, parent/collider re-enabling, clone, serialization, and reused
  arena slots.
- `ccd/sweeps.rs` rejects sensor pseudo hits before their sweep when the existing
  continuous pass would necessarily discard them: neither collider is a sensor,
  neither requests collision events, or the exact initial-overlap test succeeds.
  Hooks still run first. Pseudo hits never clamp motion; the final clamped-end
  test remains in place because a solid impact can change the endpoint. Fixtures
  cover crossings, overlap exits, event selection, hooks, and solid clamps for
  default CCD and bullet bodies.
- `Graph::contains_edge` searches both endpoint adjacency lists until either is
  exhausted. Duplicate-pair checks in the narrow phase therefore cost
  O(1 + min(degree(a), degree(b))), avoiding quadratic insertion next to a large
  sensor. Edge identity lookup, ordering, and serialization remain unchanged.
  An independent edge-array oracle covers parallel edges, self-loops, swap
  removals, clone, and serialization. The ignored `edge_existence_star_scaling_diagnostic`
  compares growth costs without imposing a machine-dependent timing assertion.
- Empty cuboid contacts retain a certified separating face. After the ordinary
  pose-recycling interval expires, projecting the current boxes onto that face
  can prove that another full manifold query would still be empty. The shortcut
  advances the reference pose exactly as the empty query would. It preserves
  the original interval before testing the face, so it does not introduce
  speculative contacts earlier. Shape/filter/hook changes and soft CCD fall back
  to the existing update. Increased prediction reach invalidates ordinary reuse
  and requires a fresh separation proof; non-finite values or uncertain separation
  also fall back. Contact-onset/trajectory, random separation, snapshot,
  hook and runtime-edit fixtures cover the boundary cases.
- Recycled contacts keep their solver-qualified manifold count. Sleeping still
  clears the dispatch hint; waking restores the stored count when an awake
  dynamic participant exists. Empty pairs avoid re-walking manifolds and bodies
  to rediscover a zero count. All manifold changes refresh the count after hooks
  and clustering. The sleep/snapshot/reactivation fixture also checks subsequent
  return to sleep. These added cache fields belong to this engine revision's
  snapshot format; use matching builds when exchanging Rapier snapshots.
- The optional `profiler` feature exposes contact-update reason counts for the
  last narrow-phase pass. Parallel workers aggregate locally and reduce after
  joining; the pair loop uses no counter atomics. Counts are diagnostic and do
  not change event or solver ordering.

The distsim repository pins this branch by commit through `third_party/rapier` and
patches `rapier3d` to `crates/rapier3d`. Its separate Bevy bridge pin is unchanged.
Run engine tests from the distsim workspace through its storage guard, compile
outside runtime limits, and execute the printed test binaries through its bounded
runner. The engine's `parallel` and `serde-serialize` configurations also require
validation. Distributed performance claims and remaining issue #7 failures are
recorded in distsim's `docs/distributed-tests.md`; fixture success alone does not
establish its realtime acceptance target.
