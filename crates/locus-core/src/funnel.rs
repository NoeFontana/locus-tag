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
    min_contrast: f64,
) {
    let tiles_wide = img.width / tile_size;
    let tiles_high = img.height / tile_size;

    for i in 0..n {
        if batch.status_mask[i] != CandidateState::Active {
            continue;
        }

        let corners = batch.corners[i];
        
        // Skip gate for very small quads where 2px delta might be unreliable
        let mut side_len_sq = 0.0;
        for j in 0..4 {
            let dx = corners[j].x - corners[(j+1)%4].x;
            let dy = corners[j].y - corners[(j+1)%4].y;
            side_len_sq += dx*dx + dy*dy;
        }
        if side_len_sq < 20.0 * 20.0 * 4.0 { // Average side < 20px
             batch.funnel_status[i] = FunnelStatus::PassedContrast;
             continue;
        }

        let mut total_contrast = 0.0;
        let mut valid_samples = 0;

        for j in 0..4 {
            let p1 = corners[j];
            let p2 = corners[(j + 1) % 4];

            // Midpoint
            let mx = (p1.x + p2.x) * 0.5;
            let my = (p1.y + p2.y) * 0.5;

            // Normal vector (inward-facing for CW)
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 1e-6 {
                continue;
            }
            let nx = -dy / len;
            let ny = dx / len;

            // Sample two points: one inside, one outside
            let delta = 2.0; 
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

            // Get local range from tile_stats safely
            let cx = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) * 0.25;
            let cy = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) * 0.25;
            
            let tx = ((cx as f32 / tile_size as f32) as usize).min(tiles_wide.saturating_sub(1));
            let ty = ((cy as f32 / tile_size as f32) as usize).min(tiles_high.saturating_sub(1));
            let stats = tile_stats[ty * tiles_wide + tx];
            let local_range = f64::from(stats.max.saturating_sub(stats.min));

            // Ultra-conservative threshold: 
            // Must have at least some fraction of local range OR a minimum absolute contrast.
            let tau = (0.1 * local_range).min(min_contrast * 0.5).max(2.0);

            if avg_contrast < tau {
                batch.funnel_status[i] = FunnelStatus::RejectedContrast;
                batch.status_mask[i] = CandidateState::FailedDecode;
            } else {
                batch.funnel_status[i] = FunnelStatus::PassedContrast;
            }
        } else {
            // If we couldn't sample safely (e.g. quad at image edge), pass it to be safe.
            batch.funnel_status[i] = FunnelStatus::PassedContrast;
        }
    }
}
