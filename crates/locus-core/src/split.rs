//! Seeded split of a fused component.
//!
//! Ink spread and blur close the diagonal contact where two dark squares meet corner to corner
//! (Kalibr's AprilGrid joins every tag to its neighbours through such X-junctions), so one
//! connected component can hold several markers, and its outline is no single marker's. The
//! necks are thinner than a marker border, so an erosion opens them while each marker keeps a
//! core. Each core dilated back (a morphological opening, labelled per core) is a piece. The
//! piece covers every border pixel the erosion kept, which is all a marker's outer contour needs.
//! It drops what the erosion removed: the neck stubs, which would otherwise bend that contour.
//!
//! The structuring element is 2 × 2, the smallest that opens anything: it keeps every stroke at
//! least 2 px wide. A 3 × 3 one also opened 2 px borders, and fisheye compression thins a
//! 2-cell border to that at the rim: it broke 14 % of the tags at a rendered 70° field angle
//! into fragments, none a marker.
//!
//! Everything runs on the component's runs, so the cost follows its run count, not its bounding
//! box. A background component can span the whole frame.

use crate::segmentation::{ComponentStats, UnionFindSlice};
use crate::simd_ccl_fusion::RleSegment;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

/// Smallest eroded core a marker can leave, in pixels. The erosion keeps a marker only if its
/// border is at least 2 px wide, i.e. its side is at least `2 · cells / border` px; its core is
/// then at least the one-pixel ring of side `side − 1`, `4 · (side − 2)` pixels. Smaller cores
/// (connector squares, noise specks, texture) seed nothing, which is what keeps textured frames
/// cheap.
pub(crate) fn min_marker_core(cells: u32, border: usize) -> u32 {
    let side = 2 * cells as usize / border.max(1);
    u32::try_from(4 * side.saturating_sub(2)).map_or(u32::MAX, |c| c.max(4))
}

/// One piece of a split component: its runs (scan order) and their statistics.
pub(crate) struct Piece<'a> {
    pub runs: BumpVec<'a, RleSegment>,
    pub stat: ComponentStats,
}

/// Half-open column interval `[s, e)` on one row.
#[derive(Clone, Copy)]
struct Span {
    y: u16,
    s: u16,
    e: u16,
}

/// Spans in scan order, indexed by row: row `k` (from the component's top) is
/// `spans[row_start[k]..row_start[k + 1]]`.
struct Rows<'a> {
    spans: BumpVec<'a, Span>,
    row_start: &'a mut [u32],
}

impl Rows<'_> {
    fn row(&self, k: usize) -> &[Span] {
        &self.spans[self.row_start[k] as usize..self.row_start[k + 1] as usize]
    }
}

/// Splits a component at the necks a 2 × 2 erosion opens: strokes one pixel wide, and the
/// diagonal pinch where ink spread joins two squares at an X-junction, which the erosion leaves
/// as cores touching only diagonally (cores are labelled 4-connected).
///
/// `runs` are the component's runs in scan order. One core is enough: a marker fused only to
/// its connector squares comes back as itself, without them. Returns no pieces when no core of
/// at least `min_core` pixels survives the erosion.
pub(crate) fn split_component<'a>(
    arena: &'a Bump,
    runs: &[RleSegment],
    stat: &ComponentStats,
    min_core: u32,
    need_moments: bool,
) -> BumpVec<'a, Piece<'a>> {
    let height = usize::from(stat.max_y - stat.min_y) + 1;
    if height < 2 || runs.is_empty() {
        return BumpVec::new_in(arena);
    }
    let cores = erode(arena, runs, stat.min_y, height);
    let (piece_of, n_pieces) = label_cores(arena, &cores, height, min_core);
    if n_pieces == 0 {
        return BumpVec::new_in(arena);
    }
    open(arena, &cores.spans, piece_of, n_pieces, need_moments)
}

/// The 2 × 2 erosion of the component, as spans: a pixel stays if it and its right, lower and
/// lower-right neighbours are all in the component.
fn erode<'a>(arena: &'a Bump, runs: &[RleSegment], top: u16, height: usize) -> Rows<'a> {
    // Horizontal: every run loses its last pixel.
    let mut shrunk = Rows {
        spans: BumpVec::with_capacity_in(runs.len(), arena),
        row_start: arena.alloc_slice_fill_copy(height + 1, 0u32),
    };
    for run in runs {
        if run.end_x - run.start_x > 1 {
            shrunk.spans.push(Span {
                y: run.y,
                s: run.start_x,
                e: run.end_x - 1,
            });
            shrunk.row_start[usize::from(run.y - top) + 1] += 1;
        }
    }
    prefix_sum(shrunk.row_start);

    // Vertical: a pixel stays if the shrunk rows at and below it both hold it.
    let mut cores = Rows {
        spans: BumpVec::with_capacity_in(shrunk.spans.len(), arena),
        row_start: arena.alloc_slice_fill_copy(height + 1, 0u32),
    };
    for k in 0..height - 1 {
        let before = cores.spans.len();
        intersect2([shrunk.row(k), shrunk.row(k + 1)], &mut cores.spans);
        cores.row_start[k + 1] = (cores.spans.len() - before) as u32;
    }
    prefix_sum(cores.row_start);
    cores
}

/// Labels the 4-connected cores and numbers those of at least `min_core` pixels. Returns, per
/// core span, its piece (`u32::MAX` for none), and the number of pieces.
fn label_cores<'a>(
    arena: &'a Bump,
    cores: &Rows<'_>,
    height: usize,
    min_core: u32,
) -> (&'a mut [u32], u32) {
    let spans = &cores.spans;
    let mut uf = UnionFindSlice::new(arena.alloc_slice_fill_iter(0..spans.len() as u32), 0);
    for k in 1..height - 1 {
        let above = cores.row_start[k - 1] as usize..cores.row_start[k] as usize;
        let here = cores.row_start[k] as usize..cores.row_start[k + 1] as usize;
        let (mut up, mut down) = (above.start, here.start);
        while up < above.end && down < here.end {
            let (a, b) = (spans[up], spans[down]);
            if a.s < b.e && b.s < a.e {
                uf.union(up as u32, down as u32);
            }
            if a.e <= b.e {
                up += 1;
            } else {
                down += 1;
            }
        }
    }
    let size = arena.alloc_slice_fill_copy(spans.len(), 0u32);
    for (idx, span) in spans.iter().enumerate() {
        size[uf.find(idx as u32) as usize] += u32::from(span.e - span.s);
    }
    // Roots come first in scan order (min-index union), so one pass numbers them and maps
    // every span to its root's number.
    let piece_of = arena.alloc_slice_fill_copy(spans.len(), u32::MAX);
    let mut n_pieces = 0u32;
    for idx in 0..spans.len() {
        let root = uf.find(idx as u32) as usize;
        if root == idx {
            if size[idx] >= min_core {
                piece_of[idx] = n_pieces;
                n_pieces += 1;
            }
        } else {
            piece_of[idx] = piece_of[root];
        }
    }
    (piece_of, n_pieces)
}

/// Dilates every numbered core back by the 2 × 2 element and returns the pieces. The element
/// at a core pixel lies in the component by construction, so nothing is clipped.
fn open<'a>(
    arena: &'a Bump,
    cores: &[Span],
    piece_of: &[u32],
    n_pieces: u32,
    need_moments: bool,
) -> BumpVec<'a, Piece<'a>> {
    // Stable counting sort by piece: each piece's spans stay in scan order.
    let n = n_pieces as usize;
    let start = arena.alloc_slice_fill_copy(n + 1, 0u32);
    for &piece in piece_of {
        if piece != u32::MAX {
            start[piece as usize + 1] += 1;
        }
    }
    prefix_sum(start);
    let fill = arena.alloc_slice_copy(&start[..n]);
    let by_piece = arena.alloc_slice_fill_copy(start[n] as usize, Span { y: 0, s: 0, e: 0 });
    for (core, &piece) in cores.iter().zip(piece_of) {
        if piece != u32::MAX {
            by_piece[fill[piece as usize] as usize] = *core;
            fill[piece as usize] += 1;
        }
    }
    let mut pieces = BumpVec::with_capacity_in(n, arena);
    for piece in 0..n {
        let spans = &by_piece[start[piece] as usize..start[piece + 1] as usize];
        pieces.push(dilate(arena, spans, piece as u32 + 1, need_moments));
    }
    pieces
}

/// The 2 × 2 dilation of one core, given as spans in scan order: each output row is the union
/// of the core's rows above and at it, every span widened by a pixel to the right.
fn dilate<'a>(arena: &'a Bump, spans: &[Span], label: u32, need_moments: bool) -> Piece<'a> {
    let mut piece = Piece {
        runs: BumpVec::new_in(arena),
        stat: ComponentStats::default(),
    };
    let top = spans[0].y;
    let height = usize::from(spans[spans.len() - 1].y - top) + 1;
    let row_start = arena.alloc_slice_fill_copy(height + 1, 0u32);
    for span in spans {
        row_start[usize::from(span.y - top) + 1] += 1;
    }
    prefix_sum(row_start);
    let row = |k: usize| -> &[Span] {
        k.checked_sub(1).filter(|&r| r < height).map_or(&[], |r| {
            &spans[row_start[r] as usize..row_start[r + 1] as usize]
        })
    };
    let mut emit = |y: u16, start_x: u16, end_x: u16| {
        let run = RleSegment {
            y,
            start_x,
            end_x,
            label,
        };
        crate::simd_ccl_fusion::accumulate_run(&mut piece.stat, &run, need_moments);
        piece.runs.push(run);
    };
    // Output row `k` is image row `top + k`; it unions core rows `k − 1` and `k`, which `row`
    // indexes from one so that the rows past either end read as empty.
    for k in 0..=height {
        let lists = [row(k), row(k + 1)];
        let mut at = [0usize; 2];
        let mut open_run: Option<(u16, u16)> = None;
        // Two-way merge by start column, coalescing spans that overlap or touch.
        loop {
            let mut next: Option<(usize, u16)> = None;
            for list in 0..2 {
                if let Some(span) = lists[list].get(at[list])
                    && next.is_none_or(|(_, s)| span.s < s)
                {
                    next = Some((list, span.s));
                }
            }
            let Some((list, _)) = next else { break };
            let span = lists[list][at[list]];
            at[list] += 1;
            let (s, e) = (span.s, span.e + 1);
            open_run = match open_run {
                Some((rs, re)) if s <= re => Some((rs, re.max(e))),
                Some((rs, re)) => {
                    emit(top + k as u16, rs, re);
                    Some((s, e))
                },
                None => Some((s, e)),
            };
        }
        if let Some((rs, re)) = open_run {
            emit(top + k as u16, rs, re);
        }
    }
    piece
}

fn prefix_sum(counts: &mut [u32]) {
    for k in 1..counts.len() {
        counts[k] += counts[k - 1];
    }
}

/// Appends to `out` the intersection of two sorted, disjoint span lists of consecutive rows, on
/// the first row.
fn intersect2(rows: [&[Span]; 2], out: &mut BumpVec<'_, Span>) {
    let (mut a, mut b) = (0, 0);
    while a < rows[0].len() && b < rows[1].len() {
        let (upper, lower) = (rows[0][a], rows[1][b]);
        let (start, end) = (upper.s.max(lower.s), upper.e.min(lower.e));
        if start < end {
            out.push(Span {
                y: upper.y,
                s: start,
                e: end,
            });
        }
        // Advance the span that ends first; it cannot meet anything further right.
        if upper.e <= lower.e {
            a += 1;
        } else {
            b += 1;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Runs of a binary picture given as rows of `#` / `.`.
    fn runs_of(rows: &[&str]) -> (Vec<RleSegment>, ComponentStats) {
        let mut runs = Vec::new();
        let mut stat = ComponentStats::default();
        for (y, r) in rows.iter().enumerate() {
            let b = r.as_bytes();
            let mut x = 0;
            while x < b.len() {
                if b[x] == b'#' {
                    let s = x;
                    while x < b.len() && b[x] == b'#' {
                        x += 1;
                    }
                    let run = RleSegment::new(y as u16, s as u16, x as u16);
                    crate::simd_ccl_fusion::accumulate_run(&mut stat, &run, false);
                    runs.push(run);
                } else {
                    x += 1;
                }
            }
        }
        (runs, stat)
    }

    #[test]
    fn two_squares_joined_at_a_corner_split_into_both_squares() {
        // Two 5 × 5 squares joined at their corners by a two-pixel neck.
        let rows = [
            "#####......",
            "#####......",
            "#####......",
            "#####......",
            "######.....",
            ".....######",
            "......#####",
            "......#####",
            "......#####",
            "......#####",
        ];
        let (runs, stat) = runs_of(&rows);
        let arena = Bump::new();
        let pieces = split_component(&arena, &runs, &stat, 1, false);
        assert_eq!(pieces.len(), 2);
        let boxes: Vec<_> = pieces
            .iter()
            .map(|p| {
                (
                    p.stat.min_x,
                    p.stat.max_x,
                    p.stat.min_y,
                    p.stat.max_y,
                    p.stat.pixel_count,
                )
            })
            .collect();
        // Each piece is its square exactly: the neck pixels are gone.
        assert_eq!(boxes, vec![(0, 4, 0, 4, 25), (6, 10, 5, 9, 25)]);
    }

    #[test]
    fn a_square_with_a_stub_comes_back_without_it() {
        // A 6 × 6 square with a one-pixel-wide stub on its right edge.
        let rows = [
            "######..", "######..", "########", "######..", "######..", "######..",
        ];
        let (runs, stat) = runs_of(&rows);
        let arena = Bump::new();
        let pieces = split_component(&arena, &runs, &stat, 1, false);
        assert_eq!(pieces.len(), 1);
        let p = &pieces[0].stat;
        assert_eq!(
            (p.min_x, p.max_x, p.min_y, p.max_y, p.pixel_count),
            (0, 5, 0, 5, 36)
        );
    }

    /// A border 2 px wide, as fisheye compression leaves a 2-cell border at the rim, survives
    /// whole.
    #[test]
    fn a_two_pixel_ring_stays_whole() {
        let rows = [
            "########", "########", "##....##", "##....##", "##....##", "##....##", "########",
            "########",
        ];
        let (runs, stat) = runs_of(&rows);
        let arena = Bump::new();
        let pieces = split_component(&arena, &runs, &stat, 1, false);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].stat.pixel_count, stat.pixel_count);
    }

    #[test]
    fn a_thin_component_has_no_core() {
        let rows = ["##########"];
        let (runs, stat) = runs_of(&rows);
        let arena = Bump::new();
        assert!(split_component(&arena, &runs, &stat, 1, false).is_empty());
    }

    #[test]
    fn cores_below_the_floor_seed_no_piece() {
        // A 3 × 3 block (a four-pixel core) hanging off a 6 × 6 square (a 25-pixel core).
        let rows = [
            "######....",
            "######....",
            "######....",
            "######.###",
            "##########",
            "######.###",
        ];
        let (runs, stat) = runs_of(&rows);
        let arena = Bump::new();
        let square = split_component(&arena, &runs, &stat, 5, false);
        assert_eq!(square.len(), 1);
        assert_eq!(
            square[0].stat.max_x, 5,
            "the block is not part of the square's piece"
        );
        assert_eq!(split_component(&arena, &runs, &stat, 1, false).len(), 2);
    }
}
