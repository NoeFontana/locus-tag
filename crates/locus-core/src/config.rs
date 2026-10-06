//! Configuration types for the detector pipeline.
//!
//! [`DetectorConfig`] is the pipeline-level configuration, immutable after the detector is
//! constructed. Build one with struct-update syntax over [`DetectorConfig::default`] (the
//! `standard` profile) or load a profile with `DetectorConfig::from_profile` /
//! `DetectorConfig::from_profile_json`.

/// Segmentation connectivity mode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SegmentationConnectivity {
    /// 4-connectivity: Pixels connect horizontally and vertically only.
    /// Required for separating checkerboard corners.
    Four,
    /// 8-connectivity: Pixels connect horizontally, vertically, and diagonally.
    /// Better for isolated tags with broken borders.
    Eight,
}

/// How the per-pixel foreground threshold that feeds segmentation is built.
///
/// The connected-component stage marks a pixel as foreground when
/// `pixel < threshold_map[pixel]`, so this enum fully determines which pixels
/// segmentation sees. A threshold of `0` is the canonical "never foreground"
/// value (no `u8` is `< 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ThresholdMode {
    /// A fixed fraction of the min/max range over a 3x3 tile neighbourhood —
    /// [`crate::threshold::CUT_NUM`]/[`crate::threshold::CUT_DEN`] of it, below
    /// the midpoint the name records and this mode originally used. The
    /// historical default; every shipped profile uses it, and it is the only
    /// mode whose output the regression snapshots pin.
    ///
    /// Two fragilities follow from the threshold being a function of the local
    /// *extremes* rather than the local background. A flat tile gets `t` at or
    /// just above its own grey level, so sensor noise speckles uniform regions
    /// with foreground. And a nearby highlight widens the range and lifts the
    /// cut with it, so a dark background can still cross into foreground and
    /// fuse with a marker — cutting below the midpoint shrinks that window but
    /// does not close it, which is what [`Self::LocalMean`] exists for.
    TileMidExtreme,
    /// Per-pixel local mean over a `(2r+1)^2` window minus a noise-calibrated offset
    /// `clamp(round(k · σ̂ₙ), 2, 20)`, where `r` is
    /// [`DetectorConfig::threshold_local_mean_radius`] and `k` is
    /// [`DetectorConfig::threshold_noise_k`]. Computed with a sliding column-sum
    /// accumulator, not an integral image.
    ///
    /// The threshold tracks the local *background level* instead of the local
    /// extremes, which is what keeps a dark textured background from fusing
    /// with a marker; the offset is what keeps sensor noise in a uniform
    /// region below the threshold. Opt-in: it changes detector output on every
    /// frame.
    LocalMean,
}

/// Mode for subpixel corner refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CornerRefinementMode {
    /// Trust the extractor's native corners. With `EdLines` these are
    /// the Gauss-Newton sub-pixel corners (metrology-grade per
    /// `docs/engineering/benchmarking/lessons.md §4.1`); with
    /// `ContourRdp` these are integer-precision midpoints.
    None,
    /// PSF-blurred step function fit via Gauss-Newton on the gradient
    /// profile. Default for `ContourRdp`.
    Erf,
}

/// Quad extraction algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuadExtractionMode {
    /// Legacy contour tracing + Douglas-Peucker + reduce-to-quad (default, backward compatible).
    #[default]
    ContourRdp,
    /// Localized Edge Drawing: anchor routing → line fitting → corner intersection.
    EdLines,
}

/// Policy controlling the EdLines AXIS→DIAG imbalance gate.
///
/// The gate triggers when AXIS-mode boundary segmentation produces one
/// arc above 40 % and another below 16 % of the boundary, indicating
/// two adjacent corners have collapsed onto a single TRBL extremal.
/// When enabled it diverts the candidate to DIAG-mode (NW/NE/SE/SW
/// extremals); when disabled the AXIS partition is kept (which is what
/// distortion-suite aprilgrid sub-tags need — they can legitimately
/// produce min-arc 8–15 % without being collapsed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdLinesImbalanceGatePolicy {
    /// Gate is off — keep the AXIS 4-arc partition unconditionally.
    #[default]
    Disabled,
    /// Gate is on — divert to DIAG-mode when the AXIS partition is severely
    /// unbalanced.
    Enabled,
}

impl EdLinesImbalanceGatePolicy {
    /// Lower the policy to the boolean consumed by the EdLines extractor.
    #[must_use]
    #[inline]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// Per-candidate routing config for [`QuadExtractionPolicy::AdaptivePpb`].
///
/// A pixels-per-bit (PPB) estimate is computed per candidate from its
/// segmentation bounding box and the minimum tag outer dimension across
/// configured decoders. Candidates with `ppb < threshold` take the `low_*`
/// route (typically ContourRdp + Erf for small/blurry tags); candidates with
/// `ppb >= threshold` take the `high_*` route (typically EdLines + None
/// for metrology-grade accuracy). When `ppb == threshold` exactly, the low
/// route wins (deterministic tie-break for snapshot stability).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdaptivePpbConfig {
    /// PPB cutoff separating low and high routes. Validated to fall in `(1.0, 5.0)`.
    pub threshold: f32,
    /// Extraction mode applied when estimated PPB < threshold.
    pub low_extraction: QuadExtractionMode,
    /// Extraction mode applied when estimated PPB >= threshold.
    pub high_extraction: QuadExtractionMode,
    /// Corner refinement mode applied on the low route.
    pub low_refinement: CornerRefinementMode,
    /// Corner refinement mode applied on the high route.
    pub high_refinement: CornerRefinementMode,
}

impl Default for AdaptivePpbConfig {
    fn default() -> Self {
        Self {
            threshold: 2.5,
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::EdLines,
            low_refinement: CornerRefinementMode::Erf,
            high_refinement: CornerRefinementMode::None,
        }
    }
}

/// Per-frame dispatch strategy for quad extraction.
///
/// `Static` (default): every candidate runs
/// `DetectorConfig::quad_extraction_mode` + `DetectorConfig::refinement_mode`.
/// `AdaptivePpb(...)` routes each candidate to one of two configurations
/// based on an on-the-fly pixels-per-bit estimate; it requires
/// `DetectorConfig::refinement_mode == None`, since the routes carry their own
/// refinement modes.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuadExtractionPolicy {
    /// Defer to `DetectorConfig::quad_extraction_mode` and
    /// `DetectorConfig::refinement_mode` (existing, default behavior).
    #[default]
    Static,
    /// Per-candidate routing based on pixels-per-bit estimate.
    AdaptivePpb(AdaptivePpbConfig),
}

/// Pipeline-level configuration for the detector.
///
/// These settings affect the fundamental behavior of the detection pipeline
/// and are immutable after the `Detector` is constructed. Construct one with
/// struct-update syntax over [`DetectorConfig::default`], e.g.
/// `DetectorConfig { quad_min_area: 400, ..DetectorConfig::default() }`, or load
/// a JSON profile.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorConfig {
    // Threshold parameters
    /// Tile size for adaptive thresholding (default: 8).
    /// Larger tiles are faster but less adaptive to local contrast.
    pub threshold_tile_size: usize,

    /// Enable Laplacian sharpening to enhance edges for small tags (default: true).
    pub enable_sharpening: bool,

    /// How the per-pixel foreground threshold is built (default:
    /// [`ThresholdMode::TileMidExtreme`], the historical behaviour).
    pub threshold_mode: ThresholdMode,
    /// Window radius, in pixels, of the local-mean thresholder (default: 7).
    /// Only read by [`ThresholdMode::LocalMean`]; the window is `(2r+1)^2`.
    pub threshold_local_mean_radius: usize,
    /// Noise-calibrated offset `k` of [`ThresholdMode::LocalMean`] (default: 4.0; must be
    /// finite and positive).
    ///
    /// The offset subtracted from the local mean is chosen per frame as
    /// `clamp(round(k · σ̂ₙ), 2, 20)` grey levels, where `σ̂ₙ` is the sensor-noise standard
    /// deviation of the thresholded image (Immerkær estimate on the raw frame, scaled by the
    /// pre-filter white-noise gain). A flat background pixel then becomes foreground with
    /// probability ≈ Φ(−k), whatever the camera's noise level — one physical parameter
    /// instead of a per-camera grey-level constant. Ignored by
    /// [`ThresholdMode::TileMidExtreme`].
    pub threshold_noise_k: f32,

    // Quad filtering parameters
    /// Minimum quad area in pixels (default: 36).
    pub quad_min_area: u32,
    /// Maximum aspect ratio of bounding box (default: 10.0).
    pub quad_max_aspect_ratio: f32,
    /// Minimum fill ratio (pixel count / bbox area); 0.0 (the default) disables it.
    pub quad_min_fill_ratio: f32,
    /// Maximum fill ratio (default: 0.98).
    pub quad_max_fill_ratio: f32,
    /// Minimum edge length in pixels (default: 4.0).
    pub quad_min_edge_length: f64,
    /// Edge-contrast floor of the quad edge gate (default: 4.0): the minimum mean gradient
    /// contrast, in grey levels, across each candidate edge. Not a normalised 0..1 score;
    /// `0.0` disables the gate.
    pub quad_min_edge_score: f64,
    /// PSF blur factor for subpixel refinement (default: 0.6).
    pub subpixel_refinement_sigma: f64,
    /// Segmentation connectivity of dark regions (default: 4-way).
    ///
    /// Where two dark squares touch at a corner (AprilGrid connector squares, any tag touching
    /// dark structure diagonally) 8-way connectivity always joins them through the diagonal
    /// pixels, fusing a whole board into one component (EuRoC `cam_april`: 20 % recall vs 86 %).
    /// 8-way only helps rings thinner than a pixel on the diagonal.
    pub segmentation_connectivity: SegmentationConnectivity,
    /// Factor to upscale the image before detection (1 = no upscaling).
    /// Increasing this to 2 allows detecting smaller tags (e.g., < 15px)
    /// at the cost of processing speed (O(N^2)). Bilinear interpolation is used.
    /// Thresholding, segmentation and quad extraction run on the upscaled grid
    /// (so `quad_min_area`, `quad_min_edge_length` are in upscaled pixels), but
    /// all outputs (corners, homography, pose, covariance) are reported in
    /// original-image coordinates. Ignored when `decimation > 1`.
    pub upscale_factor: usize,

    /// Decimation factor for preprocessing (1 = no decimation).
    pub decimation: usize,

    /// Rayon worker count for **intra-frame** parallelism (0 = global pool).
    ///
    /// `0` (the default) leaves the pipeline on Rayon's global pool, whose
    /// size is governed by `RAYON_NUM_THREADS` or the core count. A non-zero
    /// value makes `LocusEngine` build one scoped [`rayon::ThreadPool`] of
    /// exactly that size at construction and run `detect` /
    /// `detect_concurrent` under `ThreadPool::install`, bounding the detector's
    /// CPU footprint independently of the process-wide pool.
    ///
    /// Output is invariant to this value: every parallel stage writes to
    /// disjoint, index-addressed chunks, so results are bit-identical across
    /// thread counts. Only latency and CPU usage change.
    ///
    /// Not a profile field — set it per call site
    /// ([`crate::DetectorBuilder::with_threads`], or `Detector(threads=…)` in
    /// Python), not in a shipped JSON profile.
    pub nthreads: usize,

    // Decoder parameters
    /// Minimum contrast range for Otsu-based bit classification (default: 20.0).
    /// For checkerboard patterns with densely packed tags, lower values (e.g., 10.0)
    /// can improve recall on small/blurry tags.
    pub decoder_min_contrast: f64,
    /// Largest fraction of the one-cell black border ring that may read bright for a decoded
    /// candidate to be accepted. `None` (the default) gives each family the codeword's own
    /// error density, `max_hamming_error / bit_count`: one ring cell of 28 for tag36h11 at
    /// h = 2, none for tag16h5 at h = 0. `Some(1.0)` disables the check (`high_accuracy`).
    ///
    /// The ring (`4·(d + 1)` cells around a `d×d` payload) is sampled through the candidate
    /// homography; a cell is an error when it reads nearly as bright as the payload's bright
    /// class. It is marker evidence the codeword does not carry, which matters for small
    /// dictionaries (tag16h5) where textured candidates decode by chance; for a strong code a
    /// zero budget would only reject real markers with one glare- or occlusion-hit ring cell.
    pub decoder_max_border_error_rate: Option<f32>,
    /// Sub-pixel corner stage of every decoded marker (default: on).
    ///
    /// Runs after the configured [`CornerRefinementMode`] on candidates that decoded, and the
    /// homography is recomputed from the moved corners. Three steps, none configured:
    /// 1. Each corner `c` moves to the minimiser of `Σ w·(∇I(p)·(c − p))²` over a
    ///    Gaussian-weighted window sized from the marker's cell (the `cv::cornerSubPix`
    ///    model).
    /// 2. It is fused, by inverse covariance, with the intersection of the marker's two
    ///    whole-edge lines. A corner whose seed is no junction at all (quad extraction cut
    ///    across a blurred apex or a touching square) is first re-placed at the crossing of
    ///    its two edges, fitted from the neighbouring corners.
    /// 3. The corners are calibrated against the marker's own bit edges, which removes the
    ///    tone-curve inset every gradient estimator has on gamma-encoded images.
    ///
    /// Applies to undistorted cameras, and skips markers whose cells are under ~3.3 px (their
    /// seed corners are kept).
    pub decoder_corner_subpix: bool,
    /// Strategy for refining corner positions (default: [`CornerRefinementMode::Erf`]).
    ///
    /// Read only under [`QuadExtractionPolicy::Static`]; `AdaptivePpb` requires `None` here
    /// and takes its per-route modes from the policy.
    pub refinement_mode: CornerRefinementMode,
    /// Maximum number of Hamming errors allowed for tag decoding.
    ///
    /// `None` (the default) defers to each registered family's
    /// `TagDecoder::default_max_hamming` — tighter on dense codebooks
    /// (e.g. 16h5 = 0, 4x4_* = 1) and looser on sparse ones
    /// (e.g. 36h11 = 2, 6x6_250 = 2). `Some(n)` is an explicit override
    /// applied uniformly to every family.
    pub max_hamming_error: Option<u32>,

    // Pose estimation tuning parameters
    /// Huber delta for LM reprojection (pixels) in Fast mode.
    /// Residuals beyond this threshold are down-weighted linearly.
    /// 1.5 px is a standard robust threshold for sub-pixel corner detectors.
    pub huber_delta_px: f64,

    /// Maximum Tikhonov regularisation alpha (px^2) for ill-conditioned corners
    /// in Accurate mode. Controls the gain-scheduled regularisation of the
    /// Structure Tensor information matrix on foreshortened tags.
    pub tikhonov_alpha_max: f64,

    /// Pixel noise variance (sigma_n^2) assumed for the Structure Tensor
    /// covariance model in Accurate mode. Typical webcams: ~4.0.
    ///
    /// Also serves as the isotropic noise variance for the Fast-mode pose
    /// consistency gate (`pose_consistency_fpr`) when no per-corner covariance
    /// is available.
    pub sigma_n_sq: f64,

    /// Radius (in pixels) of the window used for Structure Tensor computation
    /// in Accurate mode. A radius of 2 yields a 5x5 window.
    /// Smaller values (1) are better for small tags; larger (3-4) for noisy images.
    /// Validation caps this at 8 to keep the covariance kernel stack-only.
    pub structure_tensor_radius: u8,

    /// Target false-positive rate for the pose-consistency gate.
    ///
    /// `0.0` (the default) disables the gate — `estimate_tag_pose` returns
    /// the LM-refined pose unconditionally and the legacy IPPE branch
    /// selection (lowest reprojection error in ideal corner space) is used.
    ///
    /// A positive value `p ∈ (0, 1)` derives a chi-squared critical value
    /// `χ²(2)` for the aggregate Mahalanobis distance `d² = rᵀ Σ⁻¹ r` over
    /// the four corners (8 obs − 6 DOF) and `χ²(1)` for each per-corner
    /// residual; both must pass or the pose is rejected (`Detection.pose`
    /// becomes `None`). Σ is sourced from the Structure Tensor (Accurate
    /// mode) or `Σ = sigma_n_sq · I` (Fast mode, isotropic fallback). Enabling the gate also activates
    /// observed-space Mahalanobis IPPE branch selection with branch swap.
    ///
    /// Recommended starting values: `1e-3` (good FPR/recall trade-off for
    /// tag36h11 1080p), `1e-4` (stricter), `0.0` (disabled).
    pub pose_consistency_fpr: f64,

    /// Pixel σ assumed by the pose-consistency χ² gate.
    ///
    /// The gate's null distribution is χ²(2) under the assumption that
    /// post-LM residuals are 2-D Gaussian with covariance σ²·I. To keep
    /// that calibration valid, the gate uses isotropic info matrices
    /// `Σ⁻¹ = (1/σ²)·I` independent of the LM's per-corner weighting
    /// (which can be anisotropic / structure-tensor-derived).
    ///
    /// Decoupled from `sigma_n_sq` because the LM and the gate serve
    /// different roles: the LM weights real Gaussian sensor noise (σ_n
    /// is typically 1–2 px on real cameras); the gate rejects
    /// geometrically-impossible residuals — values of 0.5–1 px give
    /// the gate enough resolution to catch sub-2-px false-positive
    /// residuals that the looser LM noise model would let through.
    /// Default: 1.0 px.
    pub pose_consistency_gate_sigma_px: f64,

    /// Outlier-aware corner-drop trigger threshold (squared Mahalanobis).
    ///
    /// After the weighted LM converges, if any corner's reprojection
    /// d²_i = rᵢᵀ Σᵢ⁻¹ rᵢ exceeds this threshold *and* dominates the
    /// second-worst corner by a factor ≥ 2, the solver masks that
    /// corner (zeroes its info matrix), re-runs the LM warm-started,
    /// and keeps the 3-corner pose iff its aggregate d² over the
    /// **three kept corners** is strictly lower than the 4-corner pose's
    /// aggregate d² over those same three corners. This self-rejection
    /// invariant catches catastrophic per-corner outliers (PSF artefacts,
    /// motion-blur spikes, lens vignette edges) that survive the Huber
    /// kernel's down-weighting but bias the rotation tail.
    ///
    /// `0.0` (the default) disables the mechanism — production binaries
    /// stay byte-identical for profiles that have not opted in.
    /// Recommended value: `25.0` (≡ 5σ², `χ²(1; 1.5e-6)`) — only triggers
    /// on the genuinely catastrophic single-corner outliers driving the
    /// rotation p99 tail.
    pub outlier_drop_d2_threshold: f64,

    /// Enable the opt-in **model-edge pose refinement** stage (Accurate mode).
    ///
    /// After the corner-based pose, aligns the decoded tag's internal bit-grid
    /// edges + border to the image and refines the pose against them (rotation
    /// from ~40 distributed edges, translation re-anchored to the 4 corners).
    /// Cuts rotation p99 substantially while preserving translation. Requires
    /// camera intrinsics + `tag_size` (Accurate mode). `false` (default) keeps
    /// production byte-identical for profiles that have not opted in.
    pub pose_edge_refinement_enabled: bool,

    /// Maximum elongation (λ_max / λ_min) allowed for a component before contour tracing.
    /// 0.0 = disabled (the default).
    ///
    /// This gate and [`Self::quad_min_density`] / [`Self::quad_min_fill_ratio`] assume a filled
    /// component. A marker merged with neighbouring structure, or one whose black cells are
    /// wider than the threshold neighbourhood (a hollow ring), fails them although it decodes;
    /// `standard` judges candidates by marker evidence instead (border ring, codeword budget).
    pub quad_max_elongation: f64,

    /// Minimum pixel density (pixel_count / bbox_area) required to pass the moments gate.
    /// 0.0 = disabled (the default).
    pub quad_min_density: f64,

    /// Quad extraction mode: legacy contour tracing (default) or EDLines.
    ///
    /// Read only when `quad_extraction_policy == Static`. Under
    /// `AdaptivePpb(...)` the policy's low/high routes override this field
    /// on a per-candidate basis.
    pub quad_extraction_mode: QuadExtractionMode,

    /// EdLines AXIS→DIAG imbalance-gate policy. See
    /// [`EdLinesImbalanceGatePolicy`].
    pub edlines_imbalance_gate: EdLinesImbalanceGatePolicy,

    /// Per-frame extraction-routing policy.
    ///
    /// DO NOT change the default to `AdaptivePpb` without a planned
    /// snapshot-review campaign: every downstream test that constructs a
    /// default config would silently exercise new code.
    pub quad_extraction_policy: QuadExtractionPolicy,
}

impl Default for DetectorConfig {
    /// Must stay field-for-field identical to `profiles/standard.json`
    /// (`config::schema_parity_tests::default_matches_standard_profile` enforces this) —
    /// `standard` is documented as the implicit default (`Detector()` /
    /// `Detector::new()`), so a silent drift here is a silent behavior
    /// change for every caller that doesn't pass a profile.
    fn default() -> Self {
        Self {
            threshold_tile_size: 8,
            enable_sharpening: true,
            threshold_mode: ThresholdMode::TileMidExtreme,
            threshold_local_mean_radius: 7,
            threshold_noise_k: 4.0,
            // 1 PPB on the smallest supported family's outer grid (6×6 cells:
            // AprilTag16h5, ArUco4x4) — below 36 px² a quad cannot represent
            // 1 pixel per bit on any family, so the decoder cannot succeed.
            // 16 was a placeholder (~0.5–0.7 PPB depending on family) that
            // never decoded anything it accepted that 36 wouldn't. Profiled
            // recall-neutral against ICRA2020/forward (worst-case TP_min
            // workload), tag36h11 1 PPB (= 64) costs 1 pp recall on forward.
            quad_min_area: 36,
            quad_max_aspect_ratio: 10.0,
            quad_min_fill_ratio: 0.0,
            quad_max_fill_ratio: 0.98,
            quad_min_edge_length: 4.0,
            quad_min_edge_score: 4.0,
            subpixel_refinement_sigma: 0.6,

            segmentation_connectivity: SegmentationConnectivity::Four,
            upscale_factor: 1,
            decimation: 1,
            nthreads: 0,
            decoder_min_contrast: 20.0,
            decoder_max_border_error_rate: None,
            decoder_corner_subpix: true,
            refinement_mode: CornerRefinementMode::Erf,
            max_hamming_error: None,
            huber_delta_px: 1.5,
            tikhonov_alpha_max: 0.25,
            sigma_n_sq: 4.0,
            structure_tensor_radius: 2,
            quad_max_elongation: 0.0,
            quad_min_density: 0.0,
            quad_extraction_mode: QuadExtractionMode::ContourRdp,
            edlines_imbalance_gate: EdLinesImbalanceGatePolicy::Disabled,
            pose_consistency_fpr: 0.0,
            pose_consistency_gate_sigma_px: 1.0,
            outlier_drop_d2_threshold: 0.0,
            pose_edge_refinement_enabled: false,
            quad_extraction_policy: QuadExtractionPolicy::Static,
        }
    }
}

impl DetectorConfig {
    /// Validate the configuration, returning an error if any parameter is out of range.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ConfigError`] if any parameter violates its constraints.
    pub fn validate(&self) -> Result<(), crate::error::ConfigError> {
        use crate::error::ConfigError;

        if self.threshold_tile_size < 2 {
            return Err(ConfigError::TileSizeTooSmall(self.threshold_tile_size));
        }
        if !(1..=crate::threshold::MAX_LOCAL_MEAN_RADIUS)
            .contains(&self.threshold_local_mean_radius)
        {
            return Err(ConfigError::InvalidLocalMeanRadius(
                self.threshold_local_mean_radius,
            ));
        }
        if let Some(rate) = self.decoder_max_border_error_rate
            && !(0.0..=1.0).contains(&rate)
        {
            return Err(ConfigError::InvalidBorderErrorRate(rate));
        }
        if !(self.threshold_noise_k.is_finite() && self.threshold_noise_k > 0.0) {
            return Err(ConfigError::InvalidNoiseK(self.threshold_noise_k));
        }
        if self.decimation < 1 {
            return Err(ConfigError::InvalidDecimation(self.decimation));
        }
        if self.upscale_factor < 1 {
            return Err(ConfigError::InvalidUpscaleFactor(self.upscale_factor));
        }
        if self.quad_min_fill_ratio < 0.0
            || self.quad_max_fill_ratio > 1.0
            || self.quad_min_fill_ratio >= self.quad_max_fill_ratio
        {
            return Err(ConfigError::InvalidFillRatio {
                min: self.quad_min_fill_ratio,
                max: self.quad_max_fill_ratio,
            });
        }
        if self.quad_min_edge_length <= 0.0 {
            return Err(ConfigError::InvalidEdgeLength(self.quad_min_edge_length));
        }
        if !(1..=8).contains(&self.structure_tensor_radius) {
            return Err(ConfigError::InvalidStructureTensorRadius(
                self.structure_tensor_radius,
            ));
        }
        if !(0.0..1.0).contains(&self.pose_consistency_fpr) || self.pose_consistency_fpr.is_nan() {
            return Err(ConfigError::InvalidPoseConsistencyFpr(
                self.pose_consistency_fpr,
            ));
        }
        if !self.outlier_drop_d2_threshold.is_finite() || self.outlier_drop_d2_threshold < 0.0 {
            return Err(ConfigError::InvalidOutlierDropD2Threshold(
                self.outlier_drop_d2_threshold,
            ));
        }
        if self.quad_extraction_mode == QuadExtractionMode::EdLines
            && self.refinement_mode == CornerRefinementMode::Erf
        {
            return Err(ConfigError::EdLinesIncompatibleWithErf);
        }
        if let QuadExtractionPolicy::AdaptivePpb(ref p) = self.quad_extraction_policy {
            if self.refinement_mode != CornerRefinementMode::None {
                return Err(ConfigError::AdaptivePolicyStaticRefinement(
                    self.refinement_mode,
                ));
            }
            if p.low_extraction == p.high_extraction {
                return Err(ConfigError::AdaptivePolicyDegenerate);
            }
            if !(p.threshold > 1.0 && p.threshold < 5.0) {
                return Err(ConfigError::AdaptivePolicyThresholdOutOfRange(p.threshold));
            }
            for (ext, refine) in [
                (p.low_extraction, p.low_refinement),
                (p.high_extraction, p.high_refinement),
            ] {
                if ext == QuadExtractionMode::EdLines && refine == CornerRefinementMode::Erf {
                    return Err(ConfigError::EdLinesIncompatibleWithErf);
                }
            }
        }
        Ok(())
    }

    /// Whether decode-first ordering applies: candidates reach the decoder with their contour
    /// corners and only those that decode (or nearly do) are refined. It needs the decoder's
    /// ERF refinement on every candidate (the `Static` policy with `Erf`), and the refinement
    /// grid must be the input image: an upscaled grid would be refined at one resolution by the
    /// quad stage and another by the decoder. Every other configuration refines during quad
    /// extraction, before decoding.
    #[must_use]
    pub(crate) fn decode_first(&self) -> bool {
        self.refinement_mode == CornerRefinementMode::Erf
            && self.upscale_factor <= 1
            && matches!(self.quad_extraction_policy, QuadExtractionPolicy::Static)
    }

    /// Returns `true` if the **static** extraction mode selects `EdLines`.
    ///
    /// Used by the distortion gate in `run_detection_pipeline` to reject
    /// only the `Static` `EdLines` configuration on distorted intrinsics
    /// (the user-explicit misconfiguration case). `AdaptivePpb` policies
    /// gracefully degrade to `ContourRdp` on the distortion path inside
    /// `extract_single_quad_with_camera` (in `crate::quad`), so they don't
    /// trip the gate even when one of their routes is `EdLines`.
    #[must_use]
    pub fn static_uses_edlines(&self) -> bool {
        matches!(self.quad_extraction_policy, QuadExtractionPolicy::Static)
            && self.quad_extraction_mode == QuadExtractionMode::EdLines
    }

    /// Whether any candidate can be routed to `EdLines` extraction, the one consumer of the
    /// full-frame label image.
    #[must_use]
    pub fn may_use_edlines(&self) -> bool {
        match self.quad_extraction_policy {
            QuadExtractionPolicy::Static => {
                self.quad_extraction_mode == QuadExtractionMode::EdLines
            },
            QuadExtractionPolicy::AdaptivePpb(cfg) => {
                cfg.low_extraction == QuadExtractionMode::EdLines
                    || cfg.high_extraction == QuadExtractionMode::EdLines
            },
        }
    }
}

/// Tag family identifier for per-call decoder selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TagFamily {
    /// AprilTag 16h5 family.
    AprilTag16h5,
    /// AprilTag 36h11 family (587 codes, 11-bit hamming distance).
    AprilTag36h11,
    /// ArUco 4x4_50 dictionary.
    ArUco4x4_50,
    /// ArUco 4x4_100 dictionary.
    ArUco4x4_100,
    /// ArUco 6x6_250 dictionary.
    ArUco6x6_250,
    /// ArUco MIP 36h12 dictionary (250 codes, 6x6 bits, 12-bit hamming distance;
    /// `cv2.aruco.DICT_ARUCO_MIP_36h12`).
    ArUcoMip36h12,
}

impl TagFamily {
    /// Returns all available tag families.
    #[must_use]
    pub const fn all() -> &'static [TagFamily] {
        &[
            TagFamily::AprilTag16h5,
            TagFamily::AprilTag36h11,
            TagFamily::ArUco4x4_50,
            TagFamily::ArUco4x4_100,
            TagFamily::ArUco6x6_250,
            TagFamily::ArUcoMip36h12,
        ]
    }

    /// Returns the number of unique tag IDs in this family's dictionary.
    ///
    /// Use this to validate board configurations before use: the number of
    /// markers on the board must not exceed this count.
    #[must_use]
    pub fn max_id_count(self) -> usize {
        crate::dictionaries::get_dictionary(self).len()
    }
}

// The three shipped JSON profiles live in `crates/locus-core/profiles/`
// and are embedded into Rust via `include_str!`; the Python wheel reads the
// exact same bytes through the `_shipped_profile_json` FFI hook. If the
// Rust defaults here and the JSON ever disagree, the JSON wins. The grouping
// below exists only at this serde boundary — `DetectorConfig` stays flat for
// hot-path access.
//
// Every nested group is `#[serde(default)]` at the container level, and its
// `Default` projects `DetectorConfig::default()`: a key omitted from a profile
// takes the canonical default, the same value the Pydantic model fills in.
#[cfg(feature = "profiles")]
mod profile_json {
    use super::{
        CornerRefinementMode, DetectorConfig, EdLinesImbalanceGatePolicy, QuadExtractionMode,
        QuadExtractionPolicy, SegmentationConnectivity, ThresholdMode,
    };
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct ProfileJson {
        /// Profile label; metadata only, not a detector setting.
        pub name: Option<String>,
        pub threshold: ThresholdJson,
        pub quad: QuadJson,
        pub decoder: DecoderJson,
        pub pose: PoseJson,
        pub segmentation: SegmentationJson,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct ThresholdJson {
        pub tile_size: usize,
        pub enable_sharpening: bool,
        pub mode: ThresholdMode,
        pub local_mean_radius: usize,
        pub noise_k: f32,
    }

    impl ThresholdJson {
        /// Project the flat hot-path struct into this nested group. Single source
        /// of the flat→nested field correspondence, shared by `Default` and
        /// `From<&DetectorConfig> for ProfileJson`.
        fn from_config(c: &DetectorConfig) -> Self {
            Self {
                tile_size: c.threshold_tile_size,
                enable_sharpening: c.enable_sharpening,
                mode: c.threshold_mode,
                local_mean_radius: c.threshold_local_mean_radius,
                noise_k: c.threshold_noise_k,
            }
        }
    }

    impl Default for ThresholdJson {
        fn default() -> Self {
            Self::from_config(&DetectorConfig::default())
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct QuadJson {
        pub min_area: u32,
        pub max_aspect_ratio: f32,
        pub min_fill_ratio: f32,
        pub max_fill_ratio: f32,
        pub min_edge_length: f64,
        pub min_edge_score: f64,
        pub subpixel_refinement_sigma: f64,
        pub upscale_factor: usize,
        pub max_elongation: f64,
        pub min_density: f64,
        pub extraction_mode: QuadExtractionMode,
        pub edlines_imbalance_gate: EdLinesImbalanceGatePolicy,
        pub extraction_policy: QuadExtractionPolicy,
    }

    impl QuadJson {
        fn from_config(c: &DetectorConfig) -> Self {
            Self {
                min_area: c.quad_min_area,
                max_aspect_ratio: c.quad_max_aspect_ratio,
                min_fill_ratio: c.quad_min_fill_ratio,
                max_fill_ratio: c.quad_max_fill_ratio,
                min_edge_length: c.quad_min_edge_length,
                min_edge_score: c.quad_min_edge_score,
                subpixel_refinement_sigma: c.subpixel_refinement_sigma,
                upscale_factor: c.upscale_factor,
                max_elongation: c.quad_max_elongation,
                min_density: c.quad_min_density,
                extraction_mode: c.quad_extraction_mode,
                edlines_imbalance_gate: c.edlines_imbalance_gate,
                extraction_policy: c.quad_extraction_policy,
            }
        }
    }

    impl Default for QuadJson {
        fn default() -> Self {
            Self::from_config(&DetectorConfig::default())
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct DecoderJson {
        pub min_contrast: f64,
        pub refinement_mode: CornerRefinementMode,
        pub max_hamming_error: Option<u32>,
        pub max_border_error_rate: Option<f32>,
        pub corner_subpix: bool,
    }

    impl DecoderJson {
        fn from_config(c: &DetectorConfig) -> Self {
            Self {
                min_contrast: c.decoder_min_contrast,
                refinement_mode: c.refinement_mode,
                max_hamming_error: c.max_hamming_error,
                max_border_error_rate: c.decoder_max_border_error_rate,
                corner_subpix: c.decoder_corner_subpix,
            }
        }
    }

    impl Default for DecoderJson {
        fn default() -> Self {
            Self::from_config(&DetectorConfig::default())
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct PoseJson {
        pub huber_delta_px: f64,
        pub tikhonov_alpha_max: f64,
        pub sigma_n_sq: f64,
        pub structure_tensor_radius: u8,
        pub pose_consistency_fpr: f64,
        pub pose_consistency_gate_sigma_px: f64,
        pub outlier_drop_d2_threshold: f64,
        pub pose_edge_refinement_enabled: bool,
    }

    impl PoseJson {
        fn from_config(c: &DetectorConfig) -> Self {
            Self {
                huber_delta_px: c.huber_delta_px,
                tikhonov_alpha_max: c.tikhonov_alpha_max,
                sigma_n_sq: c.sigma_n_sq,
                structure_tensor_radius: c.structure_tensor_radius,
                pose_consistency_fpr: c.pose_consistency_fpr,
                pose_consistency_gate_sigma_px: c.pose_consistency_gate_sigma_px,
                outlier_drop_d2_threshold: c.outlier_drop_d2_threshold,
                pose_edge_refinement_enabled: c.pose_edge_refinement_enabled,
            }
        }
    }

    impl Default for PoseJson {
        fn default() -> Self {
            Self::from_config(&DetectorConfig::default())
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(default, deny_unknown_fields)]
    pub(super) struct SegmentationJson {
        pub connectivity: SegmentationConnectivity,
    }

    impl SegmentationJson {
        fn from_config(c: &DetectorConfig) -> Self {
            Self {
                connectivity: c.segmentation_connectivity,
            }
        }
    }

    impl Default for SegmentationJson {
        fn default() -> Self {
            Self::from_config(&DetectorConfig::default())
        }
    }

    impl From<ProfileJson> for DetectorConfig {
        fn from(p: ProfileJson) -> Self {
            // `decimation` and `nthreads` are per-call orchestration, not
            // profile fields. Keep them at `DetectorConfig::default()`.
            let d = DetectorConfig::default();
            DetectorConfig {
                threshold_tile_size: p.threshold.tile_size,
                enable_sharpening: p.threshold.enable_sharpening,
                threshold_mode: p.threshold.mode,
                threshold_local_mean_radius: p.threshold.local_mean_radius,
                threshold_noise_k: p.threshold.noise_k,
                quad_min_area: p.quad.min_area,
                quad_max_aspect_ratio: p.quad.max_aspect_ratio,
                quad_min_fill_ratio: p.quad.min_fill_ratio,
                quad_max_fill_ratio: p.quad.max_fill_ratio,
                quad_min_edge_length: p.quad.min_edge_length,
                quad_min_edge_score: p.quad.min_edge_score,
                subpixel_refinement_sigma: p.quad.subpixel_refinement_sigma,
                segmentation_connectivity: p.segmentation.connectivity,
                upscale_factor: p.quad.upscale_factor,
                decimation: d.decimation,
                nthreads: d.nthreads,
                decoder_min_contrast: p.decoder.min_contrast,
                decoder_max_border_error_rate: p.decoder.max_border_error_rate,
                decoder_corner_subpix: p.decoder.corner_subpix,
                refinement_mode: p.decoder.refinement_mode,
                max_hamming_error: p.decoder.max_hamming_error,
                huber_delta_px: p.pose.huber_delta_px,
                tikhonov_alpha_max: p.pose.tikhonov_alpha_max,
                sigma_n_sq: p.pose.sigma_n_sq,
                structure_tensor_radius: p.pose.structure_tensor_radius,
                quad_max_elongation: p.quad.max_elongation,
                quad_min_density: p.quad.min_density,
                quad_extraction_mode: p.quad.extraction_mode,
                edlines_imbalance_gate: p.quad.edlines_imbalance_gate,
                pose_consistency_fpr: p.pose.pose_consistency_fpr,
                pose_consistency_gate_sigma_px: p.pose.pose_consistency_gate_sigma_px,
                outlier_drop_d2_threshold: p.pose.outlier_drop_d2_threshold,
                pose_edge_refinement_enabled: p.pose.pose_edge_refinement_enabled,
                quad_extraction_policy: p.quad.extraction_policy,
            }
        }
    }

    impl From<&DetectorConfig> for ProfileJson {
        /// Inverse of `From<ProfileJson>`: project the flat hot-path struct back
        /// into the nested profile shape for serialization. `decimation` /
        /// `nthreads` are per-call orchestration and are intentionally omitted
        /// (they are not profile fields). Total over every profile field, so the
        /// serialized document round-trips losslessly through Pydantic.
        fn from(c: &DetectorConfig) -> Self {
            ProfileJson {
                name: None,
                threshold: ThresholdJson::from_config(c),
                quad: QuadJson::from_config(c),
                decoder: DecoderJson::from_config(c),
                pose: PoseJson::from_config(c),
                segmentation: SegmentationJson::from_config(c),
            }
        }
    }
}

#[cfg(feature = "profiles")]
const STANDARD_JSON: &str = include_str!("../profiles/standard.json");
#[cfg(feature = "profiles")]
const GRID_JSON: &str = include_str!("../profiles/grid.json");
#[cfg(feature = "profiles")]
const HIGH_ACCURACY_JSON: &str = include_str!("../profiles/high_accuracy.json");

/// Return the raw embedded JSON for a shipped profile, or `None` if the name
/// is unknown. Exposed so FFI consumers (the Python wheel) can read the exact
/// bytes Rust embeds at compile time, keeping one source of truth.
#[cfg(feature = "profiles")]
#[must_use]
pub fn shipped_profile_json(name: &str) -> Option<&'static str> {
    match name {
        "standard" => Some(STANDARD_JSON),
        "grid" => Some(GRID_JSON),
        "high_accuracy" => Some(HIGH_ACCURACY_JSON),
        _ => None,
    }
}

#[cfg(feature = "profiles")]
impl DetectorConfig {
    /// Load a user-supplied profile from a JSON string.
    ///
    /// Returns [`crate::error::ConfigError::ProfileParse`] for malformed JSON or unknown
    /// fields (the serde deserializer rejects unknown keys), and any
    /// validation error from [`DetectorConfig::validate`] for configurations
    /// that fail cross-group compatibility checks (e.g. EdLines + Erf). Keys
    /// omitted from the document take their [`DetectorConfig::default`] value.
    ///
    /// # Errors
    ///
    /// See above: parse failure and post-parse validation failure.
    pub fn from_profile_json(json: &str) -> Result<Self, crate::error::ConfigError> {
        use crate::error::ConfigError;
        let parsed: profile_json::ProfileJson =
            serde_json::from_str(json).map_err(|e| ConfigError::ProfileParse(e.to_string()))?;
        let config: DetectorConfig = parsed.into();
        config.validate()?;
        Ok(config)
    }

    /// Serialize this config into the nested profile-JSON shape.
    ///
    /// The inverse of [`DetectorConfig::from_profile_json`]. This is the format
    /// that crosses the Python FFI boundary for `Detector.config()` readback:
    /// Python re-parses it into the Pydantic model, so the round-trip is total
    /// over every profile field. `decimation` / `nthreads` are per-call
    /// orchestration and are not part of the profile document.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ConfigError::ProfileParse`] if serde_json fails to
    /// serialize (not expected for the closed set of `Copy` scalar fields).
    pub fn to_profile_json(&self) -> Result<String, crate::error::ConfigError> {
        use crate::error::ConfigError;
        let profile = profile_json::ProfileJson::from(self);
        serde_json::to_string(&profile).map_err(|e| ConfigError::ProfileParse(e.to_string()))
    }

    /// Load a shipped profile by name.
    ///
    /// Accepts `"standard"`, `"grid"`, or `"high_accuracy"`.
    ///
    /// # Panics
    ///
    /// Panics on an unknown profile name — this is a programming error
    /// against a closed set of compile-time-embedded profiles.
    /// Panics on a malformed embedded JSON, which would be a build error
    /// caught by the `profile_loading` integration test.
    #[must_use]
    #[expect(
        clippy::panic,
        reason = "closed set of compile-time-embedded profiles; an unknown name is a programming error"
    )]
    pub fn from_profile(name: &str) -> Self {
        let Some(json) = shipped_profile_json(name) else {
            panic!(
                "Unknown shipped profile {name:?}; expected one of \
                 [\"standard\", \"grid\", \"high_accuracy\"]"
            )
        };
        Self::from_profile_json(json).unwrap_or_else(|e| {
            panic!("shipped profile {name:?} failed to load: {e}; this is a build bug")
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = DetectorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_bad_tile_size() {
        let config = DetectorConfig {
            threshold_tile_size: 1,
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_bad_fill_ratio() {
        let config = DetectorConfig {
            quad_min_fill_ratio: 0.9,
            quad_max_fill_ratio: 0.5,
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_negative_edge_length() {
        let config = DetectorConfig {
            quad_min_edge_length: -1.0,
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_out_of_range_structure_tensor_radius() {
        for radius in [0, 9] {
            let config = DetectorConfig {
                structure_tensor_radius: radius,
                ..DetectorConfig::default()
            };
            assert!(
                matches!(
                    config.validate(),
                    Err(crate::error::ConfigError::InvalidStructureTensorRadius(r)) if r == radius
                ),
                "structure_tensor_radius {radius} should be rejected"
            );
        }
    }

    #[test]
    fn test_validation_rejects_non_positive_noise_k() {
        for k in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let config = DetectorConfig {
                threshold_noise_k: k,
                ..DetectorConfig::default()
            };
            assert!(
                matches!(
                    config.validate(),
                    Err(crate::error::ConfigError::InvalidNoiseK(_))
                ),
                "threshold_noise_k {k} should be rejected"
            );
        }
    }

    #[cfg(feature = "serde")]
    mod imbalance_gate_serde {
        use super::super::EdLinesImbalanceGatePolicy;

        #[test]
        fn deserializes_string_variants() {
            let enabled: EdLinesImbalanceGatePolicy = serde_json::from_str("\"Enabled\"").unwrap();
            let disabled: EdLinesImbalanceGatePolicy =
                serde_json::from_str("\"Disabled\"").unwrap();
            assert_eq!(enabled, EdLinesImbalanceGatePolicy::Enabled);
            assert_eq!(disabled, EdLinesImbalanceGatePolicy::Disabled);
        }

        #[test]
        fn rejects_legacy_bool_form() {
            assert!(serde_json::from_str::<EdLinesImbalanceGatePolicy>("true").is_err());
            assert!(serde_json::from_str::<EdLinesImbalanceGatePolicy>("false").is_err());
        }

        #[test]
        fn rejects_unknown_string_variant() {
            let err =
                serde_json::from_str::<EdLinesImbalanceGatePolicy>("\"AutoMagic\"").unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("AutoMagic"), "error message: {msg}");
            assert!(msg.contains("Enabled"), "error message: {msg}");
        }

        #[test]
        fn round_trip_via_profile_json() {
            // Patch the gate field of a shipped profile to exercise the full
            // ProfileJson → QuadJson → DetectorConfig path.
            let template = super::super::shipped_profile_json("high_accuracy").unwrap();
            for (input, expected) in [
                ("\"Enabled\"", EdLinesImbalanceGatePolicy::Enabled),
                ("\"Disabled\"", EdLinesImbalanceGatePolicy::Disabled),
            ] {
                let mut value: serde_json::Value = serde_json::from_str(template).unwrap();
                value["quad"]["edlines_imbalance_gate"] = serde_json::from_str(input).unwrap();
                let json = serde_json::to_string(&value).unwrap();
                let cfg = super::super::DetectorConfig::from_profile_json(&json).unwrap();
                assert_eq!(
                    cfg.edlines_imbalance_gate, expected,
                    "input {input:?} should deserialize to {expected:?}"
                );
            }
        }
    }

    /// An `AdaptivePpb` config over the default, with the static refinement mode at `None`
    /// as validation requires.
    fn adaptive(policy: AdaptivePpbConfig) -> DetectorConfig {
        DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(policy),
            refinement_mode: CornerRefinementMode::None,
            ..DetectorConfig::default()
        }
    }

    #[test]
    fn test_adaptive_ppb_default_valid() {
        assert!(adaptive(AdaptivePpbConfig::default()).validate().is_ok());
    }

    #[test]
    fn test_adaptive_ppb_rejects_static_refinement() {
        let config = DetectorConfig {
            refinement_mode: CornerRefinementMode::Erf,
            ..adaptive(AdaptivePpbConfig::default())
        };
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::AdaptivePolicyStaticRefinement(
                CornerRefinementMode::Erf
            ))
        ));
        // AdaptivePpb never takes the decode-first path.
        assert!(!config.decode_first());
    }

    #[test]
    fn test_adaptive_ppb_rejects_degenerate() {
        let config = adaptive(AdaptivePpbConfig {
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::ContourRdp,
            ..AdaptivePpbConfig::default()
        });
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::AdaptivePolicyDegenerate)
        ));
    }

    #[test]
    fn test_adaptive_ppb_rejects_threshold_out_of_range() {
        for bad in [0.5_f32, 1.0, 5.0, 10.0] {
            let config = adaptive(AdaptivePpbConfig {
                threshold: bad,
                ..AdaptivePpbConfig::default()
            });
            assert!(
                matches!(
                    config.validate(),
                    Err(crate::error::ConfigError::AdaptivePolicyThresholdOutOfRange(_))
                ),
                "threshold {bad} should be rejected"
            );
        }
    }

    #[test]
    fn test_adaptive_ppb_per_route_edlines_erf_rejected() {
        let config = adaptive(AdaptivePpbConfig {
            low_extraction: QuadExtractionMode::ContourRdp,
            high_extraction: QuadExtractionMode::EdLines,
            low_refinement: CornerRefinementMode::None,
            high_refinement: CornerRefinementMode::Erf,
            threshold: 2.5,
        });
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::EdLinesIncompatibleWithErf)
        ));
    }

    #[test]
    fn test_static_uses_edlines() {
        let base = DetectorConfig::default();
        assert!(!base.static_uses_edlines());

        let edlines_static = DetectorConfig {
            quad_extraction_mode: QuadExtractionMode::EdLines,
            refinement_mode: CornerRefinementMode::None,
            ..DetectorConfig::default()
        };
        assert!(edlines_static.static_uses_edlines());

        // AdaptivePpb with EdLines on a route is NOT a static-EdLines config —
        // the distortion gate doesn't fire on it because the AdaptivePpb path
        // gracefully degrades to ContourRdp on distorted frames.
        let adaptive_with_edlines = adaptive(AdaptivePpbConfig::default());
        assert!(!adaptive_with_edlines.static_uses_edlines());
    }
}

/// Field-set parity tripwire.
///
/// The serde profile shim (`ProfileJson` + its nested `*Json` structs) and the
/// referee schema `schemas/profile.schema.json` must expose the identical JSON
/// key set. The schema is CI-locked to the Pydantic model
/// (`tools/export_profile_schema.py --check`), so pinning the Rust shim to the
/// schema transitively pins **Rust ↔ Python** config field-set agreement — the
/// one seam none of the value-oriented tripwires (`profile_loading.rs`,
/// `test_profiles.py`, the schema-diff job) asserts. Adding a knob on one side
/// and forgetting the other turns from a silent bug into a loud red build.
#[cfg(all(test, feature = "profiles"))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod schema_parity_tests {
    use super::profile_json::ProfileJson;
    use super::{AdaptivePpbConfig, QuadExtractionPolicy};
    use std::collections::BTreeSet;
    use std::path::Path;

    /// Recursively collect dotted leaf paths from a serialized JSON value.
    /// Objects recurse; every scalar (string enum, number, bool, `null`) is a leaf.
    fn json_leaf_paths(prefix: &str, value: &serde_json::Value, out: &mut BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    json_leaf_paths(&path, child, out);
                }
            },
            _ => {
                out.insert(prefix.to_string());
            },
        }
    }

    /// The full JSON key set the Rust serde shim round-trips. `extraction_policy`
    /// is an externally-tagged enum, so a `Static` default hides the adaptive
    /// sub-fields; union it with an `AdaptivePpb` variant to surface them.
    fn rust_key_paths() -> BTreeSet<String> {
        let mut paths = BTreeSet::new();

        let static_default = ProfileJson::default();
        json_leaf_paths(
            "",
            &serde_json::to_value(&static_default).expect("serialize default ProfileJson"),
            &mut paths,
        );

        let mut adaptive = ProfileJson::default();
        adaptive.quad.extraction_policy =
            QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig::default());
        json_leaf_paths(
            "",
            &serde_json::to_value(&adaptive).expect("serialize adaptive ProfileJson"),
            &mut paths,
        );

        paths
    }

    /// Resolve a `{"$ref": "#/$defs/Name"}` node against the schema's `$defs`;
    /// returns the node unchanged when it is not a ref.
    fn resolve<'a>(
        node: &'a serde_json::Value,
        defs: &'a serde_json::Value,
    ) -> &'a serde_json::Value {
        match node.get("$ref").and_then(serde_json::Value::as_str) {
            Some(reference) => {
                let name = reference.rsplit('/').next().expect("non-empty $ref");
                &defs[name]
            },
            None => node,
        }
    }

    /// Collect dotted leaf paths from a JSON-Schema node, resolving `$ref`s and
    /// expanding `anyOf` (a scalar/`const`/`null` branch makes the property a
    /// leaf; an object branch recurses — mirroring the serde enum's two forms).
    fn schema_leaf_paths(
        prefix: &str,
        node: &serde_json::Value,
        defs: &serde_json::Value,
        out: &mut BTreeSet<String>,
    ) {
        let node = resolve(node, defs);

        if let Some(props) = node
            .get("properties")
            .and_then(serde_json::Value::as_object)
        {
            for (key, child) in props {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                schema_leaf_paths(&path, child, defs, out);
            }
            return;
        }

        if let Some(branches) = node.get("anyOf").and_then(serde_json::Value::as_array) {
            for branch in branches {
                if resolve(branch, defs).get("properties").is_some() {
                    schema_leaf_paths(prefix, branch, defs, out);
                } else {
                    out.insert(prefix.to_string());
                }
            }
            return;
        }

        out.insert(prefix.to_string());
    }

    fn schema_key_paths() -> BTreeSet<String> {
        let schema_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas/profile.schema.json");
        let text = std::fs::read_to_string(&schema_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", schema_path.display()));
        let schema: serde_json::Value = serde_json::from_str(&text).expect("parse referee schema");
        let defs = schema
            .get("$defs")
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        let mut paths = BTreeSet::new();
        schema_leaf_paths("", &schema, &defs, &mut paths);
        paths
    }

    #[test]
    fn serde_shim_matches_referee_schema() {
        let rust = rust_key_paths();
        let schema = schema_key_paths();

        let only_in_rust: Vec<&String> = rust.difference(&schema).collect();
        let only_in_schema: Vec<&String> = schema.difference(&rust).collect();

        assert!(
            only_in_rust.is_empty() && only_in_schema.is_empty(),
            "profile field-set drift between the Rust serde shim and \
             schemas/profile.schema.json\n  only in Rust shim: {only_in_rust:?}\n  \
             only in schema:    {only_in_schema:?}\n\nAdd/remove the field on both \
             sides (and re-run tools/export_profile_schema.py).",
        );
    }

    /// Value-level counterpart to `serde_shim_matches_referee_schema` (which
    /// only pins the JSON *key* set). `standard` is documented as the
    /// implicit default (`Detector()` in Python, `Detector::new()` in Rust —
    /// see the doc comment on `impl Default for DetectorConfig`), so
    /// `DetectorConfig::default()` must equal `from_profile("standard")`
    /// field-for-field or that promise silently breaks. Caught this drifting
    /// on three fields (`enable_sharpening`, `quad_max_elongation`,
    /// `quad_min_density`) before this test existed.
    /// Every key of a profile is optional: an omitted key takes its
    /// `DetectorConfig::default()` value, the same value the Pydantic model fills in
    /// (`test_profile_values.py` pins the Python side of the same documents).
    #[test]
    fn omitted_keys_take_the_default() {
        let default = super::DetectorConfig::default();
        for json in [
            "{}",
            r#"{"name": "x"}"#,
            r#"{"threshold": {}, "quad": {}, "decoder": {}, "pose": {}, "segmentation": {}}"#,
        ] {
            let parsed = super::DetectorConfig::from_profile_json(json).expect("parse");
            assert_eq!(parsed, default, "{json}");
        }
        let partial = super::DetectorConfig::from_profile_json(
            r#"{"decoder": {"min_contrast": 12.0}, "quad": {"min_area": 100}}"#,
        )
        .expect("parse");
        assert_eq!(
            partial,
            super::DetectorConfig {
                decoder_min_contrast: 12.0,
                quad_min_area: 100,
                ..default
            }
        );
        assert!(partial.decoder_corner_subpix);
    }

    #[test]
    fn default_matches_standard_profile() {
        let default = super::DetectorConfig::default();
        let standard = super::DetectorConfig::from_profile("standard");
        assert_eq!(
            default, standard,
            "DetectorConfig::default() has drifted from profiles/standard.json \
             — standard is JSON-authoritative (docs/engineering/core.md); sync \
             `impl Default for DetectorConfig` to match, not the other way round.",
        );
    }
}
