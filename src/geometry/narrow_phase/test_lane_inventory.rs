use crate::alloc_prelude::*;
use crate::prelude::*;
use serde_json::json;

// Capture inspection only: no stepping or policy change, and no machine-specific timing gate.
#[test]
#[ignore = "requires DISTSIM_PHYSICS_SNAPSHOT from a matching dense capture"]
fn snapshot_solver_lane_inventory() {
    let path = std::env::var("DISTSIM_PHYSICS_SNAPSHOT").unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() < 256 * 1024 * 1024);
    let world: PhysicsWorld = bincode::deserialize(&bytes).unwrap();
    assert!(world.narrow_phase.solver_graph_valid);
    let mut buckets = Vec::new();
    for (color, refs) in world.narrow_phase.solver_contact_graph.buckets() {
        let counts: Vec<usize> = refs
            .iter()
            .map(|r| {
                let m = &world.narrow_phase.contact_graph.graph.edges[r.edge as usize]
                    .weight
                    .solver_manifolds()[r.manifold as usize];
                m.data
                    .num_active_contacts()
                    .min(crate::math::MAX_MANIFOLD_POINTS)
            })
            .collect();
        assert!(counts.iter().all(|&n| n > 0));
        let mut sorted = counts.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let lanes = |v: &[usize], width| {
            v.chunks(width)
                .map(|chunk| chunk.iter().max().unwrap() * width)
                .sum::<usize>()
        };
        buckets.push(json!({"color":color,"manifolds":counts.len(),
            "active_rows":counts.iter().sum::<usize>(),
            "histogram":(1..=crate::math::MAX_MANIFOLD_POINTS).map(|n| counts.iter().filter(|&&x|x==n).count()).collect::<Vec<_>>(),
            "packed4_lane_rows":lanes(&counts,4),"sorted4_lane_rows":lanes(&sorted,4),
            "packed8_lane_rows":lanes(&counts,8),"sorted8_lane_rows":lanes(&sorted,8),
            "body_disjoint":color<128}));
    }
    std::println!(
        "{}",
        json!({"snapshot":path,"buckets":buckets,
        "generic_manifolds":world.narrow_phase.solver_contact_graph.generic().len(),
        "note":"Overflow color 128 is not body-disjoint; its hypothetical sorting costs are not a safe reordering proposal. Actual multi-group and overflow grouping are not modeled here."})
    );
}

// Diagnose conservatism before changing a contact cache rule. Advancing once consumes pending
// backend edits; the second inspection uses the resulting poses that the next query will see.
#[test]
#[ignore = "requires DISTSIM_PHYSICS_SNAPSHOT from a matching dense capture"]
fn snapshot_contact_drift_inventory() {
    use crate::geometry::contact_pair::{relative_pose_drift, relative_rot_cos};
    use bincode::Options;
    let path = std::env::var("DISTSIM_PHYSICS_SNAPSHOT").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    assert!(file.metadata().unwrap().len() <= 256 * 1024 * 1024);
    let mut world: PhysicsWorld = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(256 * 1024 * 1024)
        .deserialize_from(std::io::BufReader::new(file))
        .unwrap();
    for step in 0..=1 {
        let mut counts = [0usize; 7];
        let mut by_types = std::collections::BTreeMap::<String, [usize; 3]>::new();
        for pair in world.narrow_phase.contact_pairs() {
            counts[0] += 1;
            let Some(state) = pair.recycle_state else {
                continue;
            };
            if !pair.has_any_active_contact() {
                continue;
            }
            counts[1] += 1;
            // Pending removals are still referenced by the pre-step graph until the pipeline
            // consumes the captured removal frontier.
            let (Some(co1), Some(co2)) = (
                world.colliders.get(pair.collider1),
                world.colliders.get(pair.collider2),
            ) else {
                continue;
            };
            let pose = co1.position().inv_mul(co2.position());
            let drift = relative_pose_drift(&state.pos12, &pose, state.max_extent);
            let rotation_safe = relative_rot_cos(&state.rot1, &co1.rotation())
                .min(relative_rot_cos(&state.rot2, &co2.rotation()))
                > 0.98;
            if drift <= state.max_drift {
                continue;
            }
            counts[2] += 1;
            let (Some(box1), Some(box2)) = (co1.shape().as_cuboid(), co2.shape().as_cuboid())
            else {
                continue;
            };
            counts[3] += 1;
            let radius1 = box1.half_extents.length();
            let radius2 = box2.half_extents.length();
            let type_of = |co: &Collider| {
                co.parent()
                    .map(|h| world.bodies[h].body_type())
                    .unwrap_or(RigidBodyType::Fixed)
            };
            let fixed1 = type_of(co1).is_fixed();
            let fixed2 = type_of(co2).is_fixed();
            let key = std::format!("{:?}/{:?}", type_of(co1), type_of(co2));
            let c = by_types.entry(key).or_default();
            c[0] += 1;
            let tighter = relative_pose_drift(&state.pos12, &pose, radius2).min(
                relative_pose_drift(&state.pos12.inverse(), &pose.inverse(), radius1),
            );
            if tighter <= state.max_drift && rotation_safe {
                counts[4] += 1;
                c[1] += 1;
                if fixed1 || fixed2 {
                    counts[5] += 1;
                }
            }
            if radius1.max(radius2) >= radius1.min(radius2) * 4.0 {
                counts[6] += 1;
                c[2] += 1;
            }
        }
        std::println!(
            "{}",
            json!({"snapshot":path,"after_steps":step,
            "labels":["pairs","active_cached","legacy_drift_expired","cuboid_drift_expired","symmetric_radius_reusable","fixed_pair_reusable","extent_ratio_ge_4"],
            "counts":counts,"by_types_labels":["expired","reusable","extent_ratio_ge_4"],"by_types":by_types})
        );
        if step == 0 {
            world.step();
        }
    }
}

// Diagnostic only: inspect the captured local population, without stepping or treating
// its foreign mirrors as an authoritative global scene. Run under the ordinary runtime cap.
#[test]
#[ignore = "requires DISTSIM_PHYSICS_SNAPSHOT from a matching dense capture"]
fn snapshot_spatial_partition_inventory() {
    use bincode::Options;
    let path = std::env::var("DISTSIM_PHYSICS_SNAPSHOT").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    assert!(file.metadata().unwrap().len() <= 256 * 1024 * 1024);
    let world: PhysicsWorld = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(256 * 1024 * 1024)
        .deserialize_from(std::io::BufReader::new(file))
        .unwrap();
    let bodies: Vec<_> = world
        .bodies
        .iter()
        .filter(|(_, body)| body.is_enabled() && body.is_dynamic())
        .collect();
    assert!(!bodies.is_empty());
    let quantiles = |mut values: Vec<Real>| {
        values.sort_by(Real::total_cmp);
        [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
            .map(|q| values[(q * (values.len() - 1) as f64).round() as usize])
    };
    let mut axes = Vec::new();
    for axis in 0..3 {
        let positions = quantiles(bodies.iter().map(|(_, b)| b.translation()[axis]).collect());
        let speeds = quantiles(bodies.iter().map(|(_, b)| b.linvel()[axis].abs()).collect());
        let plane = positions[4];
        let near_plane: Vec<_> = bodies
            .iter()
            .filter(|(_, b)| (b.translation()[axis] - plane).abs() <= 0.5)
            .collect();
        let mut crossing_pairs = 0;
        let mut crossing_active_pairs = 0;
        for pair in world.narrow_phase.contact_pairs() {
            let (Some(first), Some(second)) = (
                world.colliders.get(pair.collider1),
                world.colliders.get(pair.collider2),
            ) else {
                continue;
            };
            let (Some(first), Some(second)) = (first.parent(), second.parent()) else {
                continue;
            };
            let (Some(first), Some(second)) = (world.bodies.get(first), world.bodies.get(second))
            else {
                continue;
            };
            if first.is_dynamic()
                && second.is_dynamic()
                && (first.translation()[axis] < plane) != (second.translation()[axis] < plane)
            {
                crossing_pairs += 1;
                crossing_active_pairs += usize::from(pair.has_any_active_contact());
            }
        }
        axes.push(json!({"axis":axis,"position_quantiles":positions,"absolute_speed_quantiles":speeds,
            "median_plane":plane,"bodies_within_half_metre_of_plane":near_plane.len(),
            "sum_absolute_speed_in_one_metre_slab":near_plane.iter().map(|(_, b)| b.linvel()[axis].abs()).sum::<Real>(),
            "crossing_dynamic_pairs":crossing_pairs,"crossing_active_dynamic_pairs":crossing_active_pairs}));
    }
    std::println!(
        "{}",
        json!({"snapshot":path,"dynamic_bodies":bodies.len(),
        "awake_dynamic_bodies":bodies.iter().filter(|(_, b)| !b.is_sleeping()).count(),
        "quantiles":[0,0.01,0.1,0.25,0.5,0.75,0.9,0.99,1],"axes":axes,
        "limitations":"Local overlapping population, including replicas; no global ownership reconstruction, cut implementation, shape reach, support policy, future trajectories or acceptance claim."})
    );
}
