#![no_main]
use libfuzzer_sys::fuzz_target;
use price_contour_core::{
    ConstraintDirection, ConstraintSpec, QuoteGrid, SolverConfig,
    solve_online, apply_lambdas,
};

/// Parse an f32 from a 4-byte slice at the given offset.
/// Returns the value if there are enough bytes, otherwise `fallback`.
fn read_f32(data: &[u8], offset: usize, fallback: f32) -> f32 {
    if offset + 4 <= data.len() {
        f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
    } else {
        fallback
    }
}

fuzz_target!(|data: &[u8]| {
    // Need at least enough bytes to define grid dimensions
    if data.len() < 8 {
        return;
    }

    // Extract dimensions from first bytes (keep small to avoid OOM)
    let n_quotes = (data[0] as usize % 50) + 1; // 1-50
    let n_steps = (data[1] as usize % 10) + 1; // 1-10
    let n_constraints = (data[2] as usize % 3) + 1; // 1-3

    let total = n_quotes * n_steps;

    // Build sorted scenario_values (deterministic, always valid)
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| 0.8 + 0.4 * i as f32 / n_steps.max(1) as f32)
        .collect();

    // Build objective from fuzz data — may contain NaN, Inf, subnormals, -0.0
    let mut offset = 8;
    let objective: Vec<f32> = (0..total)
        .map(|i| {
            let v = read_f32(data, offset + i * 4, 1.0);
            v
        })
        .collect();
    offset += total * 4;

    // Build constraints from fuzz data
    let mut constraints = Vec::new();
    for _ in 0..n_constraints {
        let constraint: Vec<f32> = (0..total)
            .map(|i| {
                let v = read_f32(data, offset + i * 4, 1.0);
                v
            })
            .collect();
        offset += total * 4;
        constraints.push(constraint);
    }

    let quote_ids: Vec<String> = (0..n_quotes).map(|q| format!("Q{}", q)).collect();
    let constraint_names: Vec<String> = (0..n_constraints).map(|k| format!("c{}", k)).collect();

    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values,
        objective,
        constraints,
        constraint_names: constraint_names.clone(),
        quote_ids,
        quote_id_fingerprint: 0,
    };

    // Try validate — should not panic regardless of input
    if grid.validate().is_err() {
        return; // Invalid grid, that's fine
    }

    // Build specs from baseline totals
    let (_, baseline_cons) = grid.baseline_totals();
    let specs: Vec<ConstraintSpec> = constraint_names
        .iter()
        .enumerate()
        .map(|(k, name)| ConstraintSpec {
            name: name.clone(),
            direction: if k % 2 == 0 {
                ConstraintDirection::Min
            } else {
                ConstraintDirection::Max
            },
            threshold: baseline_cons[k]
                * if k % 2 == 0 { 0.9 } else { 1.1 },
        })
        .collect();

    let config = SolverConfig {
        max_iter: 10, // Keep low for fuzzing speed
        ..SolverConfig::default()
    };

    // solve_online should NEVER panic — it should return Ok or Err
    let result = solve_online(&grid, &specs, &config, None);

    // If solve succeeded, also test apply_lambdas with the resulting lambdas
    if let Ok(ref solve_result) = result {
        let _ = apply_lambdas(&grid, &specs, &solve_result.lambdas);
    }
});
