use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use price_contour_core::*;

fn make_grid(n_quotes: usize, n_steps: usize) -> QuoteGrid {
    let mut objective = Vec::with_capacity(n_quotes * n_steps);
    let mut constraint = Vec::with_capacity(n_quotes * n_steps);
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| 0.8 + 0.4 * i as f32 / (n_steps - 1).max(1) as f32)
        .collect();

    for q in 0..n_quotes {
        let base = 100.0 + (q % 100) as f32;
        for sv in &scenario_values {
            objective.push(base * *sv);
            constraint.push(1.0 / *sv); // volume-like
        }
    }

    let quote_ids = (0..n_quotes).map(|q| format!("Q{:06}", q)).collect();

    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values,
        objective,
        constraints: vec![constraint],
        constraint_names: vec!["volume".to_string()],
        quote_ids,
        quote_id_fingerprint: 0,
    };
    grid.validate().unwrap();
    grid
}

fn bench_solve_online(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_online");
    for n_quotes in [1_000, 10_000, 100_000] {
        let grid = make_grid(n_quotes, 10);
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: grid.baseline_totals().1[0] * 0.9,
        }];
        let config = SolverConfig::default();

        group.bench_with_input(
            BenchmarkId::new("quotes", n_quotes),
            &(grid, specs, config),
            |b, (grid, specs, config)| {
                b.iter(|| solve_online(grid, specs, config, None).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_argmax_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax_pass");
    for n_quotes in [10_000, 100_000, 1_000_000] {
        let grid = make_grid(n_quotes, 10);
        let lambda_signs = vec![0.1f32];

        group.bench_with_input(
            BenchmarkId::new("quotes", n_quotes),
            &(grid, lambda_signs),
            |b, (grid, signs)| {
                b.iter(|| lagrangian_argmax_pass(grid, signs, 0, grid.n_quotes));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_solve_online, bench_argmax_pass);
criterion_main!(benches);
