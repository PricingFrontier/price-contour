#![no_main]
use libfuzzer_sys::fuzz_target;
use price_contour_core::QuoteGrid;

/// Parse an f32 from a 4-byte slice at the given offset.
fn read_f32(data: &[u8], offset: usize, fallback: f32) -> f32 {
    if offset + 4 <= data.len() {
        f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
    } else {
        fallback
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    let n_quotes = (data[0] as usize % 30) + 1; // 1-30
    let n_steps_raw = (data[1] as usize % 8) + 1; // 1-8

    // Build scenario_values from fuzz data — ensure finite and positive
    let mut scenario_values: Vec<f32> = (0..n_steps_raw)
        .map(|i| {
            let v = read_f32(data, 8 + i * 4, 0.5 + i as f32 * 0.1);
            if v.is_finite() && v > 0.0 {
                v
            } else {
                0.5 + i as f32 * 0.1
            }
        })
        .collect();
    scenario_values.sort_by(|a, b| a.total_cmp(b));
    scenario_values.dedup();

    let n_steps = scenario_values.len();
    if n_steps == 0 {
        return;
    }
    let total = n_quotes * n_steps;

    // Build objective from fuzz data — may contain NaN/Inf via f32::from_le_bytes
    let obj_offset = 8 + n_steps_raw * 4;
    let objective: Vec<f32> = (0..total)
        .map(|i| read_f32(data, obj_offset + i * 4, 1.0))
        .collect();

    // Build one constraint from fuzz data
    let con_offset = obj_offset + total * 4;
    let constraint: Vec<f32> = (0..total)
        .map(|i| read_f32(data, con_offset + i * 4, 1.0))
        .collect();

    // Track whether all numeric inputs are finite
    let all_obj_finite = objective.iter().all(|v| v.is_finite());
    let all_con_finite = constraint.iter().all(|v| v.is_finite());

    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values,
        objective,
        constraints: vec![constraint],
        constraint_names: vec!["c0".into()],
        quote_ids: (0..n_quotes).map(|q| format!("Q{}", q)).collect(),
    };

    // validate() should not panic
    if grid.validate().is_err() {
        return;
    }

    // baseline_totals and compute_scale_factors should never panic on a valid grid
    let (obj_total, con_totals) = grid.baseline_totals();

    // Only assert finiteness when all inputs are finite (validate() does not
    // check objective/constraint finiteness, so non-finite inputs are allowed
    // through). When inputs ARE finite, outputs must also be finite.
    if all_obj_finite {
        assert!(
            obj_total.is_finite(),
            "baseline_totals returned non-finite objective {} with all-finite inputs",
            obj_total
        );
    }
    if all_con_finite {
        for (k, &ct) in con_totals.iter().enumerate() {
            assert!(
                ct.is_finite(),
                "baseline_totals returned non-finite constraint {}: {} with all-finite inputs",
                k,
                ct
            );
        }
    }

    let (sf_obj, sf_cons, scale_factors) = grid.compute_scale_factors();

    if all_obj_finite {
        assert!(
            sf_obj.is_finite(),
            "compute_scale_factors returned non-finite objective {} with all-finite inputs",
            sf_obj
        );
    }
    if all_obj_finite && all_con_finite {
        for (k, &sf) in scale_factors.iter().enumerate() {
            assert!(
                sf.is_finite(),
                "scale_factor[{}] should be finite with all-finite inputs, got {}",
                k,
                sf
            );
        }
    }

    // Suppress unused variable warning
    let _ = sf_cons;
});
