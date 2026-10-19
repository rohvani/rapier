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

// Read-only attribution of sleep blockers in a captured world. Connected components are
// independently reconstructed from the persistent contact/joint links; no solver policy changes.
#[test]
#[ignore = "requires DISTSIM_PHYSICS_SNAPSHOT from a matching dense capture"]
fn snapshot_sleep_inventory() {
    use bincode::Options;
    use std::collections::BTreeMap;
    let path = std::env::var("DISTSIM_PHYSICS_SNAPSHOT").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    assert!(file.metadata().unwrap().len() <= 256 * 1024 * 1024);
    let world: PhysicsWorld = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(256 * 1024 * 1024)
        .deserialize_from(std::io::BufReader::new(file))
        .unwrap();
    let counts = |handles: &[RigidBodyHandle]| {
        let mut count = [0usize; 10];
        for handle in handles {
            let Some(body) = world.bodies.get(*handle) else {
                continue;
            };
            count[0] += 1;
            if !body.is_fixed() && !body.is_sleeping() {
                count[8] += 1;
                count[9] += usize::from(body.activation().is_eligible_for_sleep());
            }
            if !body.is_dynamic() {
                continue;
            }
            count[1] += 1;
            if body.is_sleeping() {
                continue;
            }
            count[2] += 1;
            let activation = body.activation();
            count[3] += usize::from(activation.is_eligible_for_sleep());
            count[4] += usize::from(
                activation.normalized_linear_threshold < 0.0 || activation.angular_threshold < 0.0,
            );
            let extent = body.mass_properties().max_extent();
            let threshold =
                activation.normalized_linear_threshold * world.integration_parameters.length_unit;
            let drift = crate::geometry::relative_pose_drift(
                &activation.sleep_prev_pose,
                body.position(),
                extent,
            );
            count[5] += usize::from(drift * 0.5 < threshold * world.integration_parameters.dt);
            count[6] +=
                usize::from(body.linvel().length() + body.angvel().length() * extent < threshold);
            count[7] += usize::from(activation.time_since_can_sleep > 0.0);
        }
        count
    };
    fn root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }
    let all: Vec<_> = world.bodies.iter().map(|(handle, _)| handle).collect();
    let mut islands = Vec::new();
    let mut component_count = 0usize;
    let mut split_blocked_eligible = 0usize;
    let mut mixed_blocked_eligible = 0usize;
    let mut static_boundary_links = 0usize;
    let mut unmatched_nonfixed_links = 0usize;
    for (id, island) in world.islands.persistent.islands.iter() {
        let indices: BTreeMap<_, _> = island
            .bodies
            .iter()
            .enumerate()
            .map(|(i, h)| (h.into_raw_parts(), i))
            .collect();
        let mut parents: Vec<_> = (0..island.bodies.len()).collect();
        for (a, b) in island
            .contact_links
            .iter()
            .map(|l| (l.body1, l.body2))
            .chain(island.joint_links.iter().map(|l| (l.body1, l.body2)))
        {
            let (Some(&a), Some(&b)) = (
                indices.get(&a.into_raw_parts()),
                indices.get(&b.into_raw_parts()),
            ) else {
                // Fixed/disabled endpoints are deliberately absent from island membership.
                let external_is_static = [a, b]
                    .into_iter()
                    .filter(|h| !indices.contains_key(&h.into_raw_parts()))
                    .all(|h| {
                        world
                            .bodies
                            .get(h)
                            .is_some_and(|b| b.is_fixed() || !b.is_enabled())
                    });
                if external_is_static {
                    static_boundary_links += 1;
                } else {
                    unmatched_nonfixed_links += 1;
                }
                continue;
            };
            let a = root(&mut parents, a);
            let b = root(&mut parents, b);
            parents[b] = a;
        }
        let mut components: BTreeMap<usize, Vec<RigidBodyHandle>> = BTreeMap::new();
        for (i, handle) in island.bodies.iter().enumerate() {
            components
                .entry(root(&mut parents, i))
                .or_default()
                .push(*handle);
        }
        component_count += components.len();
        let component_counts: Vec<_> = components.values().map(|bodies| counts(bodies)).collect();
        for c in &component_counts {
            if c[8] > 0 && c[8] == c[9] && island.constraint_remove_count > 0 {
                split_blocked_eligible += c[3];
            } else if c[8] != c[9] {
                mixed_blocked_eligible += c[3];
            }
        }
        islands.push(
            json!({"id":id,"sleeping":island.sleeping,"bodies":island.bodies.len(),
            "contact_links":island.contact_links.len(),"joint_links":island.joint_links.len(),
            "unresolved_removals":island.constraint_remove_count,"counts":counts(&island.bodies),
            "components":component_counts}),
        );
    }
    islands.sort_by_key(|r| std::cmp::Reverse(r["bodies"].as_u64().unwrap()));
    let island_count = islands.len();
    islands.truncate(12);
    std::println!(
        "{}",
        json!({"snapshot":path,"read_only":true,
        "labels":["bodies","dynamic","awake_dynamic","eligible_awake","sleep_disabled_awake","drift_under_threshold_awake","speed_under_threshold_awake","sleep_progress_awake","awake_nonfixed","eligible_awake_nonfixed"],
        "counts":counts(&all),"persistent_islands":island_count,"reconstructed_components":component_count,
        "eligible_blocked_by_dirty_split":split_blocked_eligible,"eligible_in_mixed_component":mixed_blocked_eligible,
        "static_boundary_links":static_boundary_links,"unmatched_nonfixed_links":unmatched_nonfixed_links,"largest_islands":islands})
    );
}
