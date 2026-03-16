use crate::batch::{DetectionBatch, FunnelStatus, CandidateState};
use crate::image::ImageView;
use crate::threshold::TileStats;

/// Apply the fast-path funnel to reject candidates early.
/// 
/// This is the O(1) contrast gate that checks for photometric evidence of an edge
/// at the midpoints of the quad's boundaries.
pub fn apply_funnel_gate(
    batch: &mut DetectionBatch,
    n: usize,
    img: &ImageView,
    tile_stats: &[TileStats],
    tile_size: usize,
) {
    let tiles_wide = img.width / tile_size;
    let tiles_high = img.height / tile_size;

    for i in 0..n {
        if batch.status_mask[i] != CandidateState::Active {
            continue;
        }

        let corners = batch.corners[i];
        let mut total_contrast = 0.0;
        let mut valid_samples = 0;

        for j in 0..4 {
            let p1 = corners[j];
            let p2 = corners[(j + 1) % 4];

            // Midpoint
            let mx = (p1.x + p2.x) * 0.5;
            let my = (p1.y + p2.y) * 0.5;

            // Normal vector (inward-facing)
            // Edge vector: (dx, dy) = (p2.x - p1.x, p2.y - p1.y)
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                continue;
            }
            let nx = -dy / len;
            let ny = dx / len;

            // Sample two points: one inside, one outside
            let delta = 2.0; // 2 pixels as per spec
            let pin_x = mx + delta * nx;
            let pin_y = my + delta * ny;
            let pout_x = mx - delta * nx;
            let pout_y = my - delta * ny;

            let w = img.width as f32;
            let h = img.height as f32;

            if pin_x >= 0.0 && pin_x < w && pin_y >= 0.0 && pin_y < h 
               && pout_x >= 0.0 && pout_x < w && pout_y >= 0.0 && pout_y < h {
                let val_in = img.sample_bilinear(f64::from(pin_x), f64::from(pin_y));
                let val_out = img.sample_bilinear(f64::from(pout_x), f64::from(pout_y));
                total_contrast += (val_in - val_out).abs();
                valid_samples += 1;
            }
        }

        if valid_samples > 0 {
            let avg_contrast = total_contrast / f64::from(valid_samples);

            // Get local range from tile_stats
            let tx = (corners[0].x as usize / tile_size).min(tiles_wide - 1);
            let ty = (corners[0].y as usize / tile_size).min(tiles_high - 1);
            let stats = tile_stats[ty * tiles_wide + tx];
            let local_range = f64::from(stats.max.saturating_sub(stats.min));

            // Threshold: derived from local adaptive threshold variance.
            // A simple threshold would be some fraction of the local range.
            // For example, 0.2 * local_range.
            let tau = 0.2 * local_range;

            if avg_contrast < tau {
                batch.funnel_status[i] = FunnelStatus::RejectedContrast;
                batch.status_mask[i] = CandidateState::FailedDecode;
            } else {
                batch.funnel_status[i] = FunnelStatus::PassedContrast;
            }
        }
    }
}
