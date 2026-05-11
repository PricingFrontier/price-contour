#![no_main]
use libfuzzer_sys::fuzz_target;
use price_contour_core::{
    ConstraintDirection, ConstraintSpec, GroupMapping, QuoteGrid, SolverConfig,
    solve_grouped,
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
    if data.len() < 16 {
        return;
    }

    let n_quotes = (data[0] as usize % 20) + 1; // 1-20
    let n_steps = (data[1] as usize % 5) + 2; // 2-6
    let n_groups = (data[2] as usize % 5) + 1; // 1-5
    let n_candidates = (data[3] as usize % 10) + 1; // 1-10
    let n_constraints = (data[4] as usize % 2) + 1; // 1-2

    let total = n_quotes * n_steps;

    // Build sorted scenario_values (deterministic, always valid)
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| 0.8 + 0.4 * i as f32 / n_steps.max(1) as f32)
        .collect();

    // Build objective from fuzz data
    let offset = 16;
    let objective: Vec<f32> = (0..total)
        .map(|i| {
            let idx = offset + (i * 4) % (data.len().saturating_sub(offset).max(4));
            read_f32(data, idx, 100.0)
        })
        .collect();

    // Build constraints (use simple deterministic values mixed with some fuzz)
    let mut constraints = Vec::new();
    for k in 0..n_constraints {
        let constraint: Vec<f32> = (0..total)
            .map(|i| {
                let idx = offset + ((k * total + i) * 4) % (data.len().saturating_sub(offset).max(4));
                let v = read_f32(data, idx, 1.0);
                // Ensure finite for constraint values (otherwise validate() will pass
                // but solver may behave unpredictably — we want to test the solver
                // with plausible inputs here)
                if v.is_finite() { v } else { 1.0 }
            })
            .collect();
        constraints.push(constraint);
    }

    let quote_ids: Vec<String> = (0..n_quotes).map(|q| format!("Q{}", q)).collect();
    let constraint_names: Vec<String> = (0..n_constraints).map(|k| format!("c{}", k)).collect();

    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values: scenario_values.clone(),
        objective,
        constraints,
        constraint_names: constraint_names.clone(),
        quote_ids,
        quote_id_fingerprint: 0,
    };

    if grid.validate().is_err() {
        return;
    }

    // Build group mapping
    let group_of: Vec<u32> = (0..n_quotes).map(|q| (q % n_groups) as u32).collect();
    let group_labels: Vec<String> = (0..n_groups).map(|g| format!("G{}", g)).collect();
    let group_mapping = GroupMapping {
        n_groups,
        group_of,
        group_labels,
    };

    // Build residuals from fuzz data — ensure finite
    let residuals: Vec<f32> = (0..n_quotes)
        .map(|i| {
            let v = read_f32(data, 16 + i * 4, 1.0);
            if v.is_finite() && v > 0.0 { v } else { 1.0 }
        })
        .collect();

    // Build sorted candidates within a reasonable range
    let candidates: Vec<f32> = (0..n_candidates)
        .map(|i| 0.7 + 0.7 * i as f32 / n_candidates.max(1) as f32)
        .collect();

    // Build specs
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
            threshold: baseline_cons[k] * if k % 2 == 0 { 0.9 } else { 1.1 },
        })
        .collect();

    let config = SolverConfig {
        max_iter: 5, // Keep low for fuzzing speed
        ..SolverConfig::default()
    };

    // solve_grouped should NEVER panic
    let _ = solve_grouped(
        &grid,
        &group_mapping,
        &residuals,
        &candidates,
        &specs,
        &config,
        None,
    );
});
