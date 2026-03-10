import re
from pathlib import Path

old_content = Path('/tmp/old_decoder.rs').read_text()
new_content = Path('crates/locus-core/src/decoder.rs').read_text()

# 1. Restore multiversion import
if 'use multiversion::multiversion;' not in new_content:
    new_content = new_content.replace('use nalgebra::{SMatrix, SVector};', 'use multiversion::multiversion;\nuse nalgebra::{SMatrix, SVector};')

# 2. Extract the SIMD optimized functions from old_decoder.rs
# We want: project_gradients_optimized, collect_samples_optimized, refine_accumulate_optimized
# And we need to restore the #[multiversion] attribute to sample_grid_values_optimized

# Find the block of optimized functions in the old file (from project_gradients_optimized up to fit_edge_erf)
opt_block_match = re.search(r'(\#\[multiversion\(targets\([^)]+\)\)\]\nfn project_gradients_optimized.*?)\nfn fit_edge_erf', old_content, re.DOTALL)
if opt_block_match:
    opt_block = opt_block_match.group(1)
    
    # We will replace the EdgeFitter methods that were de-optimized
    # scan_initial_d, collect_samples, refine
    
    # Replace scan_initial_d
    new_content = re.sub(
        r'fn scan_initial_d\(&mut self\) \{.*?\n        self\.d \+= best_offset;\n    \}',
        r'''fn scan_initial_d(&mut self) {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        let mut best_offset = 0.0;
        let mut best_grad = 0.0;

        for k in -6..=6 {
            let offset = f64::from(k) * 0.4;
            let scan_d = self.d + offset;

            let (sum_g, count) =
                project_gradients_optimized(self.img, self.nx, self.ny, x0, x1, y0, y1, scan_d);

            if count > 0 && sum_g > best_grad {
                best_grad = sum_g;
                best_offset = offset;
            }
        }
        self.d += best_offset;
    }''', new_content, flags=re.DOTALL)

    # Replace collect_samples
    new_content = re.sub(
        r'fn collect_samples\(&self\) -> Vec<\(f64, f64, f64\)> \{.*?\n        samples\n    \}',
        r'''fn collect_samples(
        &self,
        arena: &'a bumpalo::Bump,
    ) -> bumpalo::collections::Vec<'a, (f64, f64, f64)> {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        collect_samples_optimized(
            self.img, self.nx, self.ny, self.d, self.p1, self.dx, self.dy, self.len, x0, x1, y0,
            y1, window, arena,
        )
    }''', new_content, flags=re.DOTALL)

    # Replace refine
    new_content = re.sub(
        r'fn refine\(&mut self, samples: &\[\(f64, f64, f64\)\], sigma: f64\) \{.*?if step\.abs\(\) < 1e-4 \{\n                break;\n            \}\n        \}\n    \}',
        r'''fn refine(&mut self, samples: &[(f64, f64, f64)], sigma: f64) {
        if samples.len() < 10 {
            return;
        }
        let mut a = 128.0;
        let mut b = 128.0;
        let inv_sigma = 1.0 / sigma;

        for _ in 0..15 {
            let mut dark_sum = 0.0;
            let mut dark_weight = 0.0;
            let mut light_sum = 0.0;
            let mut light_weight = 0.0;

            for &(x, y, _) in samples {
                let dist = self.nx * x + self.ny * y + self.d;
                let val = self.img.sample_bilinear(x, y);
                if dist < -1.0 {
                    let w = (-dist - 0.5).clamp(0.1, 2.0);
                    dark_sum += val * w;
                    dark_weight += w;
                } else if dist > 1.0 {
                    let w = (dist - 0.5).clamp(0.1, 2.0);
                    light_sum += val * w;
                    light_weight += w;
                }
            }

            if dark_weight > 0.0 && light_weight > 0.0 {
                a = dark_sum / dark_weight;
                b = light_sum / light_weight;
            }

            if (b - a).abs() < 5.0 {
                break;
            }

            let (sum_jtj, sum_jt_res) = refine_accumulate_optimized(
                samples, self.img, self.nx, self.ny, self.d, a, b, sigma, inv_sigma,
            );

            if sum_jtj < 1e-6 {
                break;
            }
            let step = sum_jt_res / sum_jtj;
            self.d += step.clamp(-0.5, 0.5);
            if step.abs() < 1e-4 {
                break;
            }
        }
    }''', new_content, flags=re.DOTALL)

    # Insert the optimized block before fit_edge_erf
    new_content = new_content.replace('fn fit_edge_erf(', opt_block + '\nfn fit_edge_erf(')
    
    # Fix fit_edge_erf to take arena
    new_content = new_content.replace('fn fit_edge_erf(\n    img', 'fn fit_edge_erf(\n    arena: &bumpalo::Bump,\n    img')
    new_content = new_content.replace('let samples = fitter.collect_samples();', 'let samples = fitter.collect_samples(arena);')
    
    # Fix refine_corners_erf to pass arena
    new_content = new_content.replace('fn refine_corners_erf(\n    img', 'fn refine_corners_erf(\n    arena: &bumpalo::Bump,\n    img')
    new_content = new_content.replace('if let Some((nx, ny, d)) = fit_edge_erf(img, p1, p2, sigma) {', 'if let Some((nx, ny, d)) = fit_edge_erf(arena, img, p1, p2, sigma) {')

    # Add multiversion back to sample_grid_values_optimized
    multiversion_attr = """#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
fn sample_grid_values_optimized"""
    new_content = new_content.replace('fn sample_grid_values_optimized', multiversion_attr)

Path('crates/locus-core/src/decoder.rs').write_text(new_content)
print("Patched decoder.rs")
