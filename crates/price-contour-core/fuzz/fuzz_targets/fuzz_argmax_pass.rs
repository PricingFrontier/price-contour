#![no_main]
use libfuzzer_sys::fuzz_target;
use price_contour_core::{
    ConstraintDirection, ConstraintSpec, QuoteGrid,
    compute_lambda_signs_f32, lagrangian_argmax_pass,
};

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

    let n_quotes = (data[0] as usize % 100) + 1; // 1-100
    let n_steps = (data[1] as usize % 10) + 1; // 1-10
    let n_constraints = (data[2] as usize % 3) + 1; // 1-3
    let total = n_quotes * n_steps;

    // Build sorted scenario values
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| 0.8 + 0.4 * i as f32 / n_steps.max(1) as f32)
        .collect();

    // Build objective from arbitrary f32 bytes — intentionally allows NaN/Inf
    // to test that argmax doesn't panic on non-finite values
    let objective: Vec<f32> = (0..total)
        .map(|i| {
            let idx = 8 + (i * 4) % (data.len().saturating_sub(8).max(4));
            read_f32(data, idx, 0.0)
        })
        .collect();

    // Build constraints from fuzz data
    let constraints: Vec<Vec<f32>> = (0..n_constraints)
        .map(|k| {
            (0..total)
                .map(|i| {
                    let idx = 8 + ((k * total + i) * 4) % (data.len().saturating_sub(8).max(4));
                    read_f32(data, idx, 0.0)
                })
                .collect()
        })
        .collect();

    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values,
        objective,
        constraints,
        constraint_names: (0..n_constraints).map(|k| format!("c{}", k)).collect(),
        quote_ids: (0..n_quotes).map(|q| format!("Q{}", q)).collect(),
    };

    // Skip validation — we want to test argmax directly with potentially
    // non-finite data (the argmax pass itself should not panic)
    // But we do need correct dimensions, which our construction guarantees.

    // Build lambda signs from fuzz data
    let lambda_signs: Vec<f32> = (0..n_constraints)
        .map(|k| {
            let idx = 4 + k;
            if idx < data.len() {
                (data[idx] as f32 - 128.0) / 10.0
            } else {
                0.1
            }
        })
        .collect();

    // lagrangian_argmax_pass should never panic
    let _ = lagrangian_argmax_pass(&grid, &lambda_signs, 0, n_quotes);

    // Also test compute_lambda_signs_f32 with fuzz-derived lambdas
    let specs: Vec<ConstraintSpec> = (0..n_constraints)
        .map(|k| ConstraintSpec {
            name: format!("c{}", k),
            direction: if k % 2 == 0 {
                ConstraintDirection::Min
            } else {
                ConstraintDirection::Max
            },
            threshold: 1.0,
        })
        .collect();

    let lambdas: Vec<f64> = (0..n_constraints)
        .map(|k| {
            let idx = 4 + k * 4;
            if idx + 4 <= data.len() {
                f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]) as f64
            } else {
                0.0
            }
        })
        .collect();

    let signed = compute_lambda_signs_f32(&specs, &lambdas);
    let _ = lagrangian_argmax_pass(&grid, &signed, 0, n_quotes);

    // Test sub-range argmax (should not panic with any valid range)
    if n_quotes > 1 {
        let mid = n_quotes / 2;
        let _ = lagrangian_argmax_pass(&grid, &lambda_signs, 0, mid);
        let _ = lagrangian_argmax_pass(&grid, &lambda_signs, mid, n_quotes);
    }
});
