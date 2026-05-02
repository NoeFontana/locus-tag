//! Configuration types for the detector pipeline.
//!
//! This module provides two configuration types:
//! - [`DetectorConfig`]: Pipeline-level configuration (immutable after construction)
//! - [`DetectOptions`]: Per-call options (e.g., which tag families to decode)

// ============================================================================
// DetectorConfig: Pipeline-level configuration
// ============================================================================

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

/// Mode for subpixel corner refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CornerRefinementMode {
    /// No subpixel refinement (integer pixel precision).
    None,
    /// Edge-based refinement using gradient maxima (Default).
    Edge,
    /// Erf: Fits a Gaussian to the gradient profile for sub-pixel edge alignment.
    Erf,
    /// Gwlf: Gradient-Weighted Line Fitting (PCA on gradients).
    Gwlf,
}

/// Mode for decoding strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DecodeMode {
    /// Hard-decision decoding using Hamming distance (fastest).
    Hard,
    /// Soft-decision decoding using Log-Likelihood Ratios (better for noise/blur).
    Soft,
}

/// Mode for 3D pose estimation quality.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PoseEstimationMode {
    /// Standard IPPE + Levenberg-Marquardt with identity weights (Fast).
    Fast,
    /// Structure Tensor + Weighted Levenberg-Marquardt (Accurate, Slower).
    ///
    /// This models corner uncertainty using image gradients and weights the
    /// PnP optimization to prefer "sharp" directions, significantly improving
    /// accuracy (RMSE) at the cost of ~0.5ms per tag.
    Accurate,
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
///
/// The legacy boolean form is still accepted on the JSON / Python
/// boundary for backward compatibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for EdLinesImbalanceGatePolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept legacy `true` / `false` alongside the tagged-string form.
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Str(String),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Bool(true) => Ok(Self::Enabled),
            Raw::Bool(false) => Ok(Self::Disabled),
            Raw::Str(s) => match s.as_str() {
                "Enabled" => Ok(Self::Enabled),
                "Disabled" => Ok(Self::Disabled),
                other => Err(serde::de::Error::custom(format!(
                    "EdLinesImbalanceGatePolicy: {other:?} is not a valid \
                     variant (allowed: \"Disabled\", \"Enabled\", or boolean \
                     true/false)"
                ))),
            },
        }
    }
}

/// Per-candidate routing config for [`QuadExtractionPolicy::AdaptivePpb`].
///
/// A pixels-per-bit (PPB) estimate is computed per candidate from its
/// segmentation bounding box and the minimum tag outer dimension across
/// configured decoders. Candidates with `ppb < threshold` take the `low_*`
/// route (typically ContourRdp + Erf for small/blurry tags); candidates with
/// `ppb >= threshold` take the `high_*` route (typically EdLines + None/Gwlf
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
/// `Static` (default) preserves existing behavior: every candidate runs
/// `DetectorConfig::quad_extraction_mode` + `DetectorConfig::refinement_mode`.
/// `AdaptivePpb(...)` routes each candidate to one of two configurations
/// based on an on-the-fly pixels-per-bit estimate.
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
/// and are immutable after the `Detector` is constructed. Use the builder
/// pattern for ergonomic construction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectorConfig {
    // Threshold parameters
    /// Tile size for adaptive thresholding (default: 4).
    /// Larger tiles are faster but less adaptive to local contrast.
    pub threshold_tile_size: usize,
    /// Minimum intensity range in a tile to be considered valid (default: 10).
    /// Tiles with lower range are treated as uniform (no edges).
    pub threshold_min_range: u8,

    /// Enable Laplacian sharpening to enhance edges for small tags (default: true).
    pub enable_sharpening: bool,

    /// Enable adaptive threshold window sizing based on gradient (default: true).
    pub enable_adaptive_window: bool,
    /// Minimum threshold window radius for high-gradient regions (default: 2 = 5x5).
    pub threshold_min_radius: usize,
    /// Maximum threshold window radius for low-gradient regions (default: 7 = 15x15).
    pub threshold_max_radius: usize,
    /// Constant subtracted from local mean in adaptive thresholding (default: 3).
    pub adaptive_threshold_constant: i16,
    /// Gradient magnitude threshold above which the minimum window radius is used (default: 40).
    pub adaptive_threshold_gradient_threshold: u8,

    // Quad filtering parameters
    /// Minimum quad area in pixels (default: 16).
    pub quad_min_area: u32,
    /// Maximum aspect ratio of bounding box (default: 3.0).
    pub quad_max_aspect_ratio: f32,
    /// Minimum fill ratio (pixel count / bbox area) (default: 0.3).
    pub quad_min_fill_ratio: f32,
    /// Maximum fill ratio (default: 0.95).
    pub quad_max_fill_ratio: f32,
    /// Minimum edge length in pixels (default: 4.0).
    pub quad_min_edge_length: f64,
    /// Minimum edge alignment score (0.0 to 1.0)
    pub quad_min_edge_score: f64,
    /// PSF blur factor for subpixel refinement (e.g., 0.6)
    pub subpixel_refinement_sigma: f64,
    /// Minimum deviation from threshold for a pixel to be connected in threshold-model CCL (default: 2).
    pub segmentation_margin: i16,
    /// Segmentation connectivity (4-way or 8-way).
    pub segmentation_connectivity: SegmentationConnectivity,
    /// Factor to upscale the image before detection (1 = no upscaling).
    /// Increasing this to 2 allows detecting smaller tags (e.g., < 15px)
    /// at the cost of processing speed (O(N^2)). Nearest-neighbor interpolation is used.
    pub upscale_factor: usize,

    /// Decimation factor for preprocessing (1 = no decimation).
    pub decimation: usize,

    /// Number of threads for parallel processing (0 = auto).
    pub nthreads: usize,

    // Decoder parameters
    /// Minimum contrast range for Otsu-based bit classification (default: 20.0).
    /// For checkerboard patterns with densely packed tags, lower values (e.g., 10.0)
    /// can improve recall on small/blurry tags.
    pub decoder_min_contrast: f64,
    /// Strategy for refining corner positions (default: Edge).
    pub refinement_mode: CornerRefinementMode,
    /// Decoding mode (Hard vs Soft).
    pub decode_mode: DecodeMode,
    /// Maximum number of Hamming errors allowed for tag decoding.
    ///
    /// `None` (the default) defers to each registered family's
    /// `TagDecoder::default_max_hamming` — tighter on dense codebooks
    /// (e.g. 16h5 = 1, 4x4_* = 1) and looser on sparse ones
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
    /// becomes `None`). Σ is sourced from GWLF covariances (when present),
    /// the Structure Tensor (Accurate mode), or `Σ = sigma_n_sq · I` (Fast
    /// mode, isotropic fallback). Enabling the gate also activates
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
    /// (which can be anisotropic / structure-tensor / GWLF-derived).
    ///
    /// Decoupled from `sigma_n_sq` because the LM and the gate serve
    /// different roles: the LM weights real Gaussian sensor noise (σ_n
    /// is typically 1–2 px on real cameras); the gate rejects
    /// geometrically-impossible residuals — values of 0.5–1 px give
    /// the gate enough resolution to catch sub-2-px false-positive
    /// residuals that the looser LM noise model would let through.
    /// Default: 1.0 px.
    pub pose_consistency_gate_sigma_px: f64,

    /// Branch-ratio escape clause for the pose-consistency gate.
    ///
    /// `alternate_d2 / primary_d2` from the IPPE branch selector. When
    /// this ratio exceeds the configured value the chosen branch is
    /// considered decisive and the χ² gate is bypassed even when the
    /// post-LM aggregate / per-corner d² exceeds the threshold.
    ///
    /// The χ² gate's job is to catch IPPE branch ambiguity: cases where
    /// the chosen branch was a coin-flip and the LM converged to a
    /// geometrically-wrong solution. When the IPPE selector had
    /// overwhelming evidence (alternate ≫ primary), a high post-LM
    /// residual is more likely to reflect scene-specific noise (PSF
    /// artefacts, lighting gradients) than a wrong branch — and the
    /// pose should not be discarded. A spurious-corner false positive
    /// has *both* IPPE candidates with high d² *and* similar magnitudes,
    /// so its ratio stays near 1 and the gate still catches it.
    ///
    /// Default: 5.0 (alternate ≥ 5× primary). Set to `f64::INFINITY` to
    /// disable the escape clause and use only the χ² test.
    pub pose_consistency_min_decisive_ratio: f64,

    /// Alpha parameter for GWLF adaptive transversal windowing.
    /// The search band is set to +/- max(2, alpha * edge_length).
    pub gwlf_transversal_alpha: f64,

    /// Maximum elongation (λ_max / λ_min) allowed for a component before contour tracing.
    /// 0.0 = disabled. Recommended: 15.0 to reject thin lines and non-square blobs.
    pub quad_max_elongation: f64,

    /// Minimum pixel density (pixel_count / bbox_area) required to pass the moments gate.
    /// 0.0 = disabled. Recommended: 0.2 to reject sparse/noisy regions.
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

    /// Run an edge-fit corner re-refit after decoding (Phase C.5).
    ///
    /// When `true`, each Valid candidate's 4 outer edges are independently
    /// fitted with the shared `ErfEdgeFitter` (PSF step model), the four
    /// adjacent edge pairs are intersected to recover sub-pixel corners,
    /// and the homography is re-solved. `corner_covariances` is left
    /// untouched — Phase A's GWLF / structure-tensor covariance is the
    /// calibrated prior for the weighted LM solver, and the line fit's
    /// Cramér-Rao bound is too tight for synthetic-PSF imagery (see
    /// `docs/engineering/post_decode_refinement_20260426.md`).
    ///
    /// `false` (the default) preserves byte-identity for all profiles that
    /// don't opt in.
    pub post_decode_refinement: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            threshold_tile_size: 8,
            threshold_min_range: 10,
            enable_sharpening: false,
            enable_adaptive_window: false,
            threshold_min_radius: 2,
            threshold_max_radius: 15,
            adaptive_threshold_constant: 0,
            adaptive_threshold_gradient_threshold: 10,
            // 1 PPB on the smallest supported family's outer grid (6×6 cells:
            // AprilTag16h5, ArUco4x4) — below 36 px² a quad cannot represent
            // 1 pixel per bit on any family, so the decoder cannot succeed.
            // 16 was a placeholder (~0.5–0.7 PPB depending on family) that
            // never decoded anything it accepted that 36 wouldn't. Profiled
            // recall-neutral against ICRA2020/forward (worst-case TP_min
            // workload), tag36h11 1 PPB (= 64) costs 1 pp recall on forward.
            quad_min_area: 36,
            quad_max_aspect_ratio: 10.0,
            quad_min_fill_ratio: 0.10,
            quad_max_fill_ratio: 0.98,
            quad_min_edge_length: 4.0,
            quad_min_edge_score: 4.0,
            subpixel_refinement_sigma: 0.6,

            segmentation_margin: 1,
            segmentation_connectivity: SegmentationConnectivity::Eight,
            upscale_factor: 1,
            decimation: 1,
            nthreads: 0,
            decoder_min_contrast: 20.0,
            refinement_mode: CornerRefinementMode::Erf,
            decode_mode: DecodeMode::Hard,
            max_hamming_error: None,
            huber_delta_px: 1.5,
            tikhonov_alpha_max: 0.25,
            sigma_n_sq: 4.0,
            structure_tensor_radius: 2,
            gwlf_transversal_alpha: 0.01,
            quad_max_elongation: 0.0,
            quad_min_density: 0.0,
            quad_extraction_mode: QuadExtractionMode::ContourRdp,
            edlines_imbalance_gate: EdLinesImbalanceGatePolicy::Disabled,
            pose_consistency_fpr: 0.0,
            pose_consistency_gate_sigma_px: 1.0,
            pose_consistency_min_decisive_ratio: 5.0,
            quad_extraction_policy: QuadExtractionPolicy::Static,
            post_decode_refinement: false,
        }
    }
}

impl DetectorConfig {
    /// Create a new builder for `DetectorConfig`.
    #[must_use]
    pub fn builder() -> DetectorConfigBuilder {
        DetectorConfigBuilder::default()
    }

    /// Validate the configuration, returning an error if any parameter is out of range.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter violates its constraints.
    pub fn validate(&self) -> Result<(), crate::error::ConfigError> {
        use crate::error::ConfigError;

        if self.threshold_tile_size < 2 {
            return Err(ConfigError::TileSizeTooSmall(self.threshold_tile_size));
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
        if self.structure_tensor_radius > 8 {
            return Err(ConfigError::InvalidStructureTensorRadius(
                self.structure_tensor_radius,
            ));
        }
        if !(0.0..1.0).contains(&self.pose_consistency_fpr) || self.pose_consistency_fpr.is_nan() {
            return Err(ConfigError::InvalidPoseConsistencyFpr(
                self.pose_consistency_fpr,
            ));
        }
        if self.pose_consistency_min_decisive_ratio < 1.0
            || self.pose_consistency_min_decisive_ratio.is_nan()
        {
            return Err(ConfigError::InvalidPoseConsistencyMinDecisiveRatio(
                self.pose_consistency_min_decisive_ratio,
            ));
        }
        if self.quad_extraction_mode == QuadExtractionMode::EdLines {
            if self.refinement_mode == CornerRefinementMode::Erf {
                return Err(ConfigError::EdLinesIncompatibleWithErf);
            }
            if self.decode_mode == DecodeMode::Soft {
                return Err(ConfigError::EdLinesIncompatibleWithSoftDecode);
            }
        }
        if let QuadExtractionPolicy::AdaptivePpb(ref p) = self.quad_extraction_policy {
            if p.low_extraction == p.high_extraction {
                return Err(ConfigError::AdaptivePolicyDegenerate);
            }
            if !(p.threshold > 1.0 && p.threshold < 5.0) {
                return Err(ConfigError::AdaptivePolicyThresholdOutOfRange(p.threshold));
            }
            // Re-apply per-route EdLines incompatibilities.
            for (ext, refine) in [
                (p.low_extraction, p.low_refinement),
                (p.high_extraction, p.high_refinement),
            ] {
                if ext == QuadExtractionMode::EdLines {
                    if refine == CornerRefinementMode::Erf {
                        return Err(ConfigError::EdLinesIncompatibleWithErf);
                    }
                    if self.decode_mode == DecodeMode::Soft {
                        return Err(ConfigError::EdLinesIncompatibleWithSoftDecode);
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns `true` if any active extraction path — either the static
    /// `quad_extraction_mode` or (under `AdaptivePpb`) either of the low/high
    /// routes — selects `EdLines`.
    ///
    /// Used by the distortion gate in `run_detection_pipeline` to reject
    /// incompatible configurations once camera intrinsics carry distortion
    /// coefficients.
    #[must_use]
    pub fn any_route_uses_edlines(&self) -> bool {
        match self.quad_extraction_policy {
            QuadExtractionPolicy::Static => {
                self.quad_extraction_mode == QuadExtractionMode::EdLines
            },
            QuadExtractionPolicy::AdaptivePpb(ref p) => {
                p.low_extraction == QuadExtractionMode::EdLines
                    || p.high_extraction == QuadExtractionMode::EdLines
            },
        }
    }
}

/// Builder for [`DetectorConfig`].
#[derive(Default)]
pub struct DetectorConfigBuilder {
    threshold_tile_size: Option<usize>,
    threshold_min_range: Option<u8>,
    enable_sharpening: Option<bool>,
    enable_adaptive_window: Option<bool>,
    threshold_min_radius: Option<usize>,
    threshold_max_radius: Option<usize>,
    adaptive_threshold_constant: Option<i16>,
    adaptive_threshold_gradient_threshold: Option<u8>,
    quad_min_area: Option<u32>,
    quad_max_aspect_ratio: Option<f32>,
    quad_min_fill_ratio: Option<f32>,
    quad_max_fill_ratio: Option<f32>,
    quad_min_edge_length: Option<f64>,
    /// Minimum gradient magnitude along edges (rejects weak candidates).
    pub quad_min_edge_score: Option<f64>,
    /// Sigma for Gaussian in subpixel refinement.
    pub subpixel_refinement_sigma: Option<f64>,
    /// Margin for threshold-model segmentation.
    pub segmentation_margin: Option<i16>,
    /// Connectivity mode for segmentation (4 or 8).
    pub segmentation_connectivity: Option<SegmentationConnectivity>,
    /// Upscale factor for low-res images (1 = no upscale).
    pub upscale_factor: Option<usize>,
    /// Minimum contrast for decoder to accept a tag.
    pub decoder_min_contrast: Option<f64>,
    /// Refinement mode.
    pub refinement_mode: Option<CornerRefinementMode>,
    /// Decoding mode.
    pub decode_mode: Option<DecodeMode>,
    /// Maximum Hamming errors.
    pub max_hamming_error: Option<u32>,
    /// GWLF transversal alpha.
    pub gwlf_transversal_alpha: Option<f64>,
    /// Maximum elongation for the moments culling gate.
    pub quad_max_elongation: Option<f64>,
    /// Minimum density for the moments culling gate.
    pub quad_min_density: Option<f64>,
    /// Quad extraction mode.
    pub quad_extraction_mode: Option<QuadExtractionMode>,
    /// Quad extraction policy (Static or AdaptivePpb).
    pub quad_extraction_policy: Option<QuadExtractionPolicy>,
    /// Huber delta for LM reprojection (pixels).
    pub huber_delta_px: Option<f64>,
    /// Maximum Tikhonov regularisation alpha for Accurate mode.
    pub tikhonov_alpha_max: Option<f64>,
    /// Pixel noise variance for Structure Tensor covariance model.
    pub sigma_n_sq: Option<f64>,
    /// Radius of the Structure Tensor window in Accurate mode.
    pub structure_tensor_radius: Option<u8>,
}

impl DetectorConfigBuilder {
    /// Set the tile size for adaptive thresholding.
    #[must_use]
    pub fn threshold_tile_size(mut self, size: usize) -> Self {
        self.threshold_tile_size = Some(size);
        self
    }

    /// Set the minimum intensity range for valid tiles.
    #[must_use]
    pub fn threshold_min_range(mut self, range: u8) -> Self {
        self.threshold_min_range = Some(range);
        self
    }

    /// Set the minimum quad area.
    #[must_use]
    pub fn quad_min_area(mut self, area: u32) -> Self {
        self.quad_min_area = Some(area);
        self
    }

    /// Set the maximum aspect ratio.
    #[must_use]
    pub fn quad_max_aspect_ratio(mut self, ratio: f32) -> Self {
        self.quad_max_aspect_ratio = Some(ratio);
        self
    }

    /// Set the minimum fill ratio.
    #[must_use]
    pub fn quad_min_fill_ratio(mut self, ratio: f32) -> Self {
        self.quad_min_fill_ratio = Some(ratio);
        self
    }

    /// Set the maximum fill ratio.
    #[must_use]
    pub fn quad_max_fill_ratio(mut self, ratio: f32) -> Self {
        self.quad_max_fill_ratio = Some(ratio);
        self
    }

    /// Set the minimum edge length.
    #[must_use]
    pub fn quad_min_edge_length(mut self, length: f64) -> Self {
        self.quad_min_edge_length = Some(length);
        self
    }

    /// Set the minimum edge gradient score.
    #[must_use]
    pub fn quad_min_edge_score(mut self, score: f64) -> Self {
        self.quad_min_edge_score = Some(score);
        self
    }

    /// Enable or disable Laplacian sharpening.
    #[must_use]
    pub fn enable_sharpening(mut self, enable: bool) -> Self {
        self.enable_sharpening = Some(enable);
        self
    }

    /// Enable or disable adaptive threshold window sizing.
    #[must_use]
    pub fn enable_adaptive_window(mut self, enable: bool) -> Self {
        self.enable_adaptive_window = Some(enable);
        self
    }

    /// Set minimum threshold window radius.
    #[must_use]
    pub fn threshold_min_radius(mut self, radius: usize) -> Self {
        self.threshold_min_radius = Some(radius);
        self
    }

    /// Set maximum threshold window radius.
    #[must_use]
    pub fn threshold_max_radius(mut self, radius: usize) -> Self {
        self.threshold_max_radius = Some(radius);
        self
    }

    /// Set the constant subtracted from local mean in adaptive thresholding.
    #[must_use]
    pub fn adaptive_threshold_constant(mut self, c: i16) -> Self {
        self.adaptive_threshold_constant = Some(c);
        self
    }

    /// Set the gradient threshold for adaptive window sizing.
    #[must_use]
    pub fn adaptive_threshold_gradient_threshold(mut self, threshold: u8) -> Self {
        self.adaptive_threshold_gradient_threshold = Some(threshold);
        self
    }

    /// Build the configuration, using defaults for unset fields.
    #[must_use]
    pub fn build(self) -> DetectorConfig {
        let d = DetectorConfig::default();
        DetectorConfig {
            threshold_tile_size: self.threshold_tile_size.unwrap_or(d.threshold_tile_size),
            threshold_min_range: self.threshold_min_range.unwrap_or(d.threshold_min_range),
            enable_sharpening: self.enable_sharpening.unwrap_or(d.enable_sharpening),
            enable_adaptive_window: self
                .enable_adaptive_window
                .unwrap_or(d.enable_adaptive_window),
            threshold_min_radius: self.threshold_min_radius.unwrap_or(d.threshold_min_radius),
            threshold_max_radius: self.threshold_max_radius.unwrap_or(d.threshold_max_radius),
            adaptive_threshold_constant: self
                .adaptive_threshold_constant
                .unwrap_or(d.adaptive_threshold_constant),
            adaptive_threshold_gradient_threshold: self
                .adaptive_threshold_gradient_threshold
                .unwrap_or(d.adaptive_threshold_gradient_threshold),
            quad_min_area: self.quad_min_area.unwrap_or(d.quad_min_area),
            quad_max_aspect_ratio: self
                .quad_max_aspect_ratio
                .unwrap_or(d.quad_max_aspect_ratio),
            quad_min_fill_ratio: self.quad_min_fill_ratio.unwrap_or(d.quad_min_fill_ratio),
            quad_max_fill_ratio: self.quad_max_fill_ratio.unwrap_or(d.quad_max_fill_ratio),
            quad_min_edge_length: self.quad_min_edge_length.unwrap_or(d.quad_min_edge_length),
            quad_min_edge_score: self.quad_min_edge_score.unwrap_or(d.quad_min_edge_score),
            subpixel_refinement_sigma: self
                .subpixel_refinement_sigma
                .unwrap_or(d.subpixel_refinement_sigma),
            segmentation_margin: self.segmentation_margin.unwrap_or(d.segmentation_margin),
            segmentation_connectivity: self
                .segmentation_connectivity
                .unwrap_or(d.segmentation_connectivity),
            upscale_factor: self.upscale_factor.unwrap_or(d.upscale_factor),
            decimation: 1, // Default to 1, as it's typically set via builder
            nthreads: 0,   // Default to 0
            decoder_min_contrast: self.decoder_min_contrast.unwrap_or(d.decoder_min_contrast),
            refinement_mode: self.refinement_mode.unwrap_or(d.refinement_mode),
            decode_mode: self.decode_mode.unwrap_or(d.decode_mode),
            max_hamming_error: self.max_hamming_error.or(d.max_hamming_error),
            huber_delta_px: self.huber_delta_px.unwrap_or(d.huber_delta_px),
            tikhonov_alpha_max: self.tikhonov_alpha_max.unwrap_or(d.tikhonov_alpha_max),
            sigma_n_sq: self.sigma_n_sq.unwrap_or(d.sigma_n_sq),
            structure_tensor_radius: self
                .structure_tensor_radius
                .unwrap_or(d.structure_tensor_radius),
            gwlf_transversal_alpha: self
                .gwlf_transversal_alpha
                .unwrap_or(d.gwlf_transversal_alpha),
            quad_max_elongation: self.quad_max_elongation.unwrap_or(d.quad_max_elongation),
            quad_min_density: self.quad_min_density.unwrap_or(d.quad_min_density),
            quad_extraction_mode: self.quad_extraction_mode.unwrap_or(d.quad_extraction_mode),
            edlines_imbalance_gate: d.edlines_imbalance_gate,
            pose_consistency_fpr: d.pose_consistency_fpr,
            pose_consistency_gate_sigma_px: d.pose_consistency_gate_sigma_px,
            pose_consistency_min_decisive_ratio: d.pose_consistency_min_decisive_ratio,
            quad_extraction_policy: self
                .quad_extraction_policy
                .unwrap_or(d.quad_extraction_policy),
            post_decode_refinement: d.post_decode_refinement,
        }
    }

    /// Build the configuration and validate all parameter ranges.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter is out of its valid range.
    pub fn validated_build(self) -> Result<DetectorConfig, crate::error::ConfigError> {
        let config = self.build();
        config.validate()?;
        Ok(config)
    }

    /// Set the segmentation connectivity.
    #[must_use]
    pub fn segmentation_connectivity(mut self, connectivity: SegmentationConnectivity) -> Self {
        self.segmentation_connectivity = Some(connectivity);
        self
    }

    /// Set the segmentation margin for threshold-model CCL.
    #[must_use]
    pub fn segmentation_margin(mut self, margin: i16) -> Self {
        self.segmentation_margin = Some(margin);
        self
    }

    /// Set the upscale factor (1 = no upscaling, 2 = 2x, etc.).
    #[must_use]
    pub fn upscale_factor(mut self, factor: usize) -> Self {
        self.upscale_factor = Some(factor);
        self
    }

    /// Set the minimum contrast for decoder bit classification.
    /// Lower values (e.g., 10.0) improve recall on small/blurry checkerboard tags.
    #[must_use]
    pub fn decoder_min_contrast(mut self, contrast: f64) -> Self {
        self.decoder_min_contrast = Some(contrast);
        self
    }

    /// Set the corner refinement mode.
    #[must_use]
    pub fn refinement_mode(mut self, mode: CornerRefinementMode) -> Self {
        self.refinement_mode = Some(mode);
        self
    }

    /// Set the decoding mode (Hard or Soft).
    #[must_use]
    pub fn decode_mode(mut self, mode: DecodeMode) -> Self {
        self.decode_mode = Some(mode);
        self
    }

    /// Set the maximum number of Hamming errors allowed.
    #[must_use]
    pub fn max_hamming_error(mut self, errors: u32) -> Self {
        self.max_hamming_error = Some(errors);
        self
    }

    /// Set the GWLF transversal alpha.
    #[must_use]
    pub fn gwlf_transversal_alpha(mut self, alpha: f64) -> Self {
        self.gwlf_transversal_alpha = Some(alpha);
        self
    }

    /// Set the maximum elongation for the moments-based culling gate.
    /// Set to 0.0 to disable (default). Recommended: 15.0.
    #[must_use]
    pub fn quad_max_elongation(mut self, max_elongation: f64) -> Self {
        self.quad_max_elongation = Some(max_elongation);
        self
    }

    /// Set the minimum density for the moments-based culling gate.
    /// Set to 0.0 to disable (default). Recommended: 0.2.
    #[must_use]
    pub fn quad_min_density(mut self, min_density: f64) -> Self {
        self.quad_min_density = Some(min_density);
        self
    }

    /// Set the quad extraction mode (ContourRdp or EdLines).
    ///
    /// Read only when `quad_extraction_policy == Static`. Under
    /// `AdaptivePpb(...)` this field is ignored in favor of the policy's
    /// per-route modes.
    #[must_use]
    pub fn quad_extraction_mode(mut self, mode: QuadExtractionMode) -> Self {
        self.quad_extraction_mode = Some(mode);
        self
    }

    /// Set the quad extraction policy (Static or AdaptivePpb).
    #[must_use]
    pub fn quad_extraction_policy(mut self, policy: QuadExtractionPolicy) -> Self {
        self.quad_extraction_policy = Some(policy);
        self
    }

    /// Set the Huber delta for LM reprojection (pixels).
    #[must_use]
    pub fn huber_delta_px(mut self, delta: f64) -> Self {
        self.huber_delta_px = Some(delta);
        self
    }

    /// Set the maximum Tikhonov regularisation alpha for Accurate pose mode.
    #[must_use]
    pub fn tikhonov_alpha_max(mut self, alpha: f64) -> Self {
        self.tikhonov_alpha_max = Some(alpha);
        self
    }

    /// Set the pixel noise variance for the Structure Tensor covariance model.
    #[must_use]
    pub fn sigma_n_sq(mut self, sigma_n_sq: f64) -> Self {
        self.sigma_n_sq = Some(sigma_n_sq);
        self
    }

    /// Set the Structure Tensor window radius for Accurate pose mode.
    /// Valid range: `0..=8`.
    #[must_use]
    pub fn structure_tensor_radius(mut self, radius: u8) -> Self {
        self.structure_tensor_radius = Some(radius);
        self
    }
}

// ============================================================================
// DetectOptions: Per-call detection options
// ============================================================================

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

/// Per-call detection options.
///
/// These allow customizing which tag families to decode for a specific call,
/// enabling performance optimization when you know which tags to expect.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectOptions {
    /// Tag families to attempt decoding. Empty means use detector defaults.
    pub families: Vec<TagFamily>,
    /// Camera intrinsics for 3D pose estimation. If None, pose is not computed.
    pub intrinsics: Option<crate::pose::CameraIntrinsics>,
    /// Physical size of the tag in world units (e.g. meters) for 3D pose estimation.
    pub tag_size: Option<f64>,
    /// Decimation factor for preprocessing (1 = no decimation).
    /// Preprocessing and segmentation operate on a downsampled image of size (W/D, H/D).
    pub decimation: usize,
    /// Mode for 3D pose estimation (Fast vs Accurate).
    pub pose_estimation_mode: PoseEstimationMode,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            families: Vec::new(),
            intrinsics: None,
            tag_size: None,
            decimation: 1,
            pose_estimation_mode: PoseEstimationMode::Fast,
        }
    }
}

impl DetectOptions {
    /// Create a new builder for `DetectOptions`.
    #[must_use]
    pub fn builder() -> DetectOptionsBuilder {
        DetectOptionsBuilder::default()
    }
    /// Create options that decode only the specified tag families.
    #[must_use]
    pub fn with_families(families: &[TagFamily]) -> Self {
        Self {
            families: families.to_vec(),
            intrinsics: None,
            tag_size: None,
            decimation: 1,
            pose_estimation_mode: PoseEstimationMode::Fast,
        }
    }

    /// Create options that decode all known tag families.
    #[must_use]
    pub fn all_families() -> Self {
        Self {
            families: TagFamily::all().to_vec(),
            intrinsics: None,
            tag_size: None,
            decimation: 1,
            pose_estimation_mode: PoseEstimationMode::Fast,
        }
    }
}

/// Builder for [`DetectOptions`].
pub struct DetectOptionsBuilder {
    families: Vec<TagFamily>,
    intrinsics: Option<crate::pose::CameraIntrinsics>,
    tag_size: Option<f64>,
    decimation: usize,
    pose_estimation_mode: PoseEstimationMode,
}

impl Default for DetectOptionsBuilder {
    fn default() -> Self {
        Self {
            families: Vec::new(),
            intrinsics: None,
            tag_size: None,
            decimation: 1,
            pose_estimation_mode: PoseEstimationMode::Fast,
        }
    }
}

impl DetectOptionsBuilder {
    /// Set the tag families to decode.
    #[must_use]
    pub fn families(mut self, families: &[TagFamily]) -> Self {
        self.families = families.to_vec();
        self
    }

    /// Set camera intrinsics for pose estimation.
    #[must_use]
    pub fn intrinsics(mut self, fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        self.intrinsics = Some(crate::pose::CameraIntrinsics::new(fx, fy, cx, cy));
        self
    }

    /// Set physical tag size for pose estimation.
    #[must_use]
    pub fn tag_size(mut self, size: f64) -> Self {
        self.tag_size = Some(size);
        self
    }

    /// Set the decimation factor (1 = no decimation).
    #[must_use]
    pub fn decimation(mut self, decimation: usize) -> Self {
        self.decimation = decimation.max(1);
        self
    }

    /// Set the pose estimation mode.
    #[must_use]
    pub fn pose_estimation_mode(mut self, mode: PoseEstimationMode) -> Self {
        self.pose_estimation_mode = mode;
        self
    }

    /// Build the options.
    #[must_use]
    pub fn build(self) -> DetectOptions {
        DetectOptions {
            families: self.families,
            intrinsics: self.intrinsics,
            tag_size: self.tag_size,
            decimation: self.decimation,
            pose_estimation_mode: self.pose_estimation_mode,
        }
    }
}

// The three shipped JSON profiles live in `crates/locus-core/profiles/`
// and are embedded into Rust via `include_str!`; the Python wheel reads the
// exact same bytes through the `_shipped_profile_json` FFI hook. If the
// Rust defaults here and the JSON ever disagree, the JSON wins. The grouping
// below exists only at this serde boundary — `DetectorConfig` stays flat for
// hot-path access.
#[cfg(feature = "profiles")]
mod profile_json {
    use super::{
        CornerRefinementMode, DecodeMode, DetectorConfig, EdLinesImbalanceGatePolicy,
        QuadExtractionMode, SegmentationConnectivity,
    };
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct ProfileJson {
        #[allow(dead_code)]
        #[serde(default)]
        pub name: Option<String>,
        #[serde(default)]
        pub extends: Option<String>,
        #[serde(default)]
        pub threshold: ThresholdJson,
        #[serde(default)]
        pub quad: QuadJson,
        #[serde(default)]
        pub decoder: DecoderJson,
        #[serde(default)]
        pub pose: PoseJson,
        #[serde(default)]
        pub segmentation: SegmentationJson,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct ThresholdJson {
        pub tile_size: usize,
        pub min_range: u8,
        pub enable_sharpening: bool,
        pub enable_adaptive_window: bool,
        pub min_radius: usize,
        pub max_radius: usize,
        pub constant: i16,
        pub gradient_threshold: u8,
    }

    impl Default for ThresholdJson {
        fn default() -> Self {
            let d = DetectorConfig::default();
            Self {
                tile_size: d.threshold_tile_size,
                min_range: d.threshold_min_range,
                enable_sharpening: d.enable_sharpening,
                enable_adaptive_window: d.enable_adaptive_window,
                min_radius: d.threshold_min_radius,
                max_radius: d.threshold_max_radius,
                constant: d.adaptive_threshold_constant,
                gradient_threshold: d.adaptive_threshold_gradient_threshold,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
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
        #[serde(default)]
        pub edlines_imbalance_gate: EdLinesImbalanceGatePolicy,
        #[serde(default)]
        pub extraction_policy: super::QuadExtractionPolicy,
    }

    impl Default for QuadJson {
        fn default() -> Self {
            let d = DetectorConfig::default();
            Self {
                min_area: d.quad_min_area,
                max_aspect_ratio: d.quad_max_aspect_ratio,
                min_fill_ratio: d.quad_min_fill_ratio,
                max_fill_ratio: d.quad_max_fill_ratio,
                min_edge_length: d.quad_min_edge_length,
                min_edge_score: d.quad_min_edge_score,
                subpixel_refinement_sigma: d.subpixel_refinement_sigma,
                upscale_factor: d.upscale_factor,
                max_elongation: d.quad_max_elongation,
                min_density: d.quad_min_density,
                extraction_mode: d.quad_extraction_mode,
                edlines_imbalance_gate: d.edlines_imbalance_gate,
                extraction_policy: d.quad_extraction_policy,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct DecoderJson {
        pub min_contrast: f64,
        pub refinement_mode: CornerRefinementMode,
        pub decode_mode: DecodeMode,
        #[serde(default)]
        pub max_hamming_error: Option<u32>,
        pub gwlf_transversal_alpha: f64,
        // Optional so existing profile JSONs (and any third-party files
        // shipped before this field existed) deserialize unchanged. Missing
        // → `false` → Phase C.5 disabled, byte-identical to today.
        #[serde(default)]
        pub post_decode_refinement: bool,
    }

    impl Default for DecoderJson {
        fn default() -> Self {
            let d = DetectorConfig::default();
            Self {
                min_contrast: d.decoder_min_contrast,
                refinement_mode: d.refinement_mode,
                decode_mode: d.decode_mode,
                max_hamming_error: d.max_hamming_error,
                gwlf_transversal_alpha: d.gwlf_transversal_alpha,
                post_decode_refinement: d.post_decode_refinement,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct PoseJson {
        pub huber_delta_px: f64,
        pub tikhonov_alpha_max: f64,
        pub sigma_n_sq: f64,
        pub structure_tensor_radius: u8,
        // Optional so existing profile JSONs (and any third-party files
        // shipped before this field existed) deserialize unchanged. Missing
        // → 0.0 → gate disabled, identical to today's behavior.
        #[serde(default)]
        pub pose_consistency_fpr: f64,
        // Optional pixel σ for the χ² consistency gate (independent of
        // `sigma_n_sq` so the gate's null distribution stays calibrated
        // against Gaussian noise even when LM uses anisotropic info
        // matrices). Missing → 1.0 px (default).
        #[serde(default = "default_gate_sigma_px")]
        pub pose_consistency_gate_sigma_px: f64,
        // Branch-ratio escape clause for the χ² gate. Missing → 5.0
        // (alternate IPPE d² ≥ 5× primary IPPE d² bypasses the gate).
        #[serde(default = "default_min_decisive_ratio")]
        pub pose_consistency_min_decisive_ratio: f64,
    }

    fn default_gate_sigma_px() -> f64 {
        1.0
    }

    fn default_min_decisive_ratio() -> f64 {
        5.0
    }

    impl Default for PoseJson {
        fn default() -> Self {
            let d = DetectorConfig::default();
            Self {
                huber_delta_px: d.huber_delta_px,
                tikhonov_alpha_max: d.tikhonov_alpha_max,
                sigma_n_sq: d.sigma_n_sq,
                structure_tensor_radius: d.structure_tensor_radius,
                pose_consistency_fpr: d.pose_consistency_fpr,
                pose_consistency_gate_sigma_px: d.pose_consistency_gate_sigma_px,
                pose_consistency_min_decisive_ratio: d.pose_consistency_min_decisive_ratio,
            }
        }
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct SegmentationJson {
        pub connectivity: SegmentationConnectivity,
        pub margin: i16,
    }

    impl Default for SegmentationJson {
        fn default() -> Self {
            let d = DetectorConfig::default();
            Self {
                connectivity: d.segmentation_connectivity,
                margin: d.segmentation_margin,
            }
        }
    }

    impl From<ProfileJson> for DetectorConfig {
        fn from(p: ProfileJson) -> Self {
            // `decimation` and `nthreads` are per-call orchestration, not
            // profile fields. Keep them at `DetectorConfig::default()`.
            let d = DetectorConfig::default();
            DetectorConfig {
                threshold_tile_size: p.threshold.tile_size,
                threshold_min_range: p.threshold.min_range,
                enable_sharpening: p.threshold.enable_sharpening,
                enable_adaptive_window: p.threshold.enable_adaptive_window,
                threshold_min_radius: p.threshold.min_radius,
                threshold_max_radius: p.threshold.max_radius,
                adaptive_threshold_constant: p.threshold.constant,
                adaptive_threshold_gradient_threshold: p.threshold.gradient_threshold,
                quad_min_area: p.quad.min_area,
                quad_max_aspect_ratio: p.quad.max_aspect_ratio,
                quad_min_fill_ratio: p.quad.min_fill_ratio,
                quad_max_fill_ratio: p.quad.max_fill_ratio,
                quad_min_edge_length: p.quad.min_edge_length,
                quad_min_edge_score: p.quad.min_edge_score,
                subpixel_refinement_sigma: p.quad.subpixel_refinement_sigma,
                segmentation_margin: p.segmentation.margin,
                segmentation_connectivity: p.segmentation.connectivity,
                upscale_factor: p.quad.upscale_factor,
                decimation: d.decimation,
                nthreads: d.nthreads,
                decoder_min_contrast: p.decoder.min_contrast,
                refinement_mode: p.decoder.refinement_mode,
                decode_mode: p.decoder.decode_mode,
                max_hamming_error: p.decoder.max_hamming_error,
                huber_delta_px: p.pose.huber_delta_px,
                tikhonov_alpha_max: p.pose.tikhonov_alpha_max,
                sigma_n_sq: p.pose.sigma_n_sq,
                structure_tensor_radius: p.pose.structure_tensor_radius,
                gwlf_transversal_alpha: p.decoder.gwlf_transversal_alpha,
                quad_max_elongation: p.quad.max_elongation,
                quad_min_density: p.quad.min_density,
                quad_extraction_mode: p.quad.extraction_mode,
                edlines_imbalance_gate: p.quad.edlines_imbalance_gate,
                pose_consistency_fpr: p.pose.pose_consistency_fpr,
                pose_consistency_gate_sigma_px: p.pose.pose_consistency_gate_sigma_px,
                pose_consistency_min_decisive_ratio: p.pose.pose_consistency_min_decisive_ratio,
                quad_extraction_policy: p.quad.extraction_policy,
                post_decode_refinement: p.decoder.post_decode_refinement,
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
#[cfg(feature = "profiles")]
const RENDER_TAG_HUB_JSON: &str = include_str!("../profiles/render_tag_hub.json");
#[cfg(feature = "profiles")]
const MAX_RECALL_ADAPTIVE_JSON: &str = include_str!("../profiles/max_recall_adaptive.json");

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
        "render_tag_hub" => Some(RENDER_TAG_HUB_JSON),
        "max_recall_adaptive" => Some(MAX_RECALL_ADAPTIVE_JSON),
        _ => None,
    }
}

#[cfg(feature = "profiles")]
impl DetectorConfig {
    /// Load a user-supplied profile from a JSON string.
    ///
    /// Returns [`ConfigError::ProfileParse`] for malformed JSON or unknown
    /// fields (the serde deserializer rejects unknown keys), and any
    /// validation error from [`DetectorConfig::validate`] for configurations
    /// that fail cross-group compatibility checks (e.g. EdLines + Erf).
    ///
    /// # Errors
    ///
    /// See above: parse failure and post-parse validation failure.
    pub fn from_profile_json(json: &str) -> Result<Self, crate::error::ConfigError> {
        use crate::error::ConfigError;
        let parsed: profile_json::ProfileJson =
            serde_json::from_str(json).map_err(|e| ConfigError::ProfileParse(e.to_string()))?;
        if let Some(name) = parsed.extends.as_deref() {
            return Err(ConfigError::ProfileParse(format!(
                "profile inheritance (extends={name:?}) is declared in the schema but \
                 not yet resolved by the Rust loader; inline the parent profile's values"
            )));
        }
        let config: DetectorConfig = parsed.into();
        config.validate()?;
        Ok(config)
    }

    /// Load a shipped profile by name.
    ///
    /// Accepts `"standard"`, `"grid"`, `"high_accuracy"`, `"render_tag_hub"`,
    /// or `"max_recall_adaptive"`.
    ///
    /// # Panics
    ///
    /// Panics on an unknown profile name — this is a programming error
    /// against a closed set of compile-time-embedded profiles.
    /// Panics on a malformed embedded JSON, which would be a build error
    /// caught by the `profile_loading` integration test.
    #[must_use]
    #[allow(clippy::panic)] // Closed set; unknown-name is a programming error.
    pub fn from_profile(name: &str) -> Self {
        let json = match name {
            "standard" => STANDARD_JSON,
            "grid" => GRID_JSON,
            "high_accuracy" => HIGH_ACCURACY_JSON,
            "render_tag_hub" => RENDER_TAG_HUB_JSON,
            "max_recall_adaptive" => MAX_RECALL_ADAPTIVE_JSON,
            other => panic!(
                "Unknown shipped profile {other:?}; expected one of \
                 [\"standard\", \"grid\", \"high_accuracy\", \"render_tag_hub\", \
                 \"max_recall_adaptive\"]"
            ),
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
    #[allow(clippy::float_cmp)]
    fn test_detector_config_builder() {
        let config = DetectorConfig::builder()
            .threshold_tile_size(16)
            .quad_min_area(1000)
            .build();
        assert_eq!(config.threshold_tile_size, 16);
        assert_eq!(config.quad_min_area, 1000);
        // Check defaults
        assert_eq!(config.threshold_min_range, 10);
        assert_eq!(config.quad_min_edge_score, 4.0);
        assert_eq!(config.max_hamming_error, None);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_detector_config_defaults() {
        let config = DetectorConfig::default();
        assert_eq!(config.threshold_tile_size, 8);
        assert_eq!(config.quad_min_area, 36);
        assert_eq!(config.quad_min_edge_length, 4.0);
        assert_eq!(config.max_hamming_error, None);
    }

    #[test]
    fn test_detect_options_families() {
        let opt = DetectOptions::with_families(&[TagFamily::AprilTag36h11]);
        assert_eq!(opt.families.len(), 1);
        assert_eq!(opt.families[0], TagFamily::AprilTag36h11);
    }

    #[test]
    fn test_detect_options_default_empty() {
        let opt = DetectOptions::default();
        assert!(opt.families.is_empty());
    }

    #[test]
    fn test_all_families() {
        let opt = DetectOptions::all_families();
        assert_eq!(opt.families.len(), 5);
    }

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
    fn test_validation_rejects_large_structure_tensor_radius() {
        let config = DetectorConfig {
            structure_tensor_radius: 9,
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validated_build_catches_errors() {
        let result = DetectorConfig::builder()
            .threshold_tile_size(0)
            .validated_build();
        assert!(result.is_err());
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
        fn deserializes_legacy_bool_form() {
            let enabled: EdLinesImbalanceGatePolicy = serde_json::from_str("true").unwrap();
            let disabled: EdLinesImbalanceGatePolicy = serde_json::from_str("false").unwrap();
            assert_eq!(enabled, EdLinesImbalanceGatePolicy::Enabled);
            assert_eq!(disabled, EdLinesImbalanceGatePolicy::Disabled);
        }

        #[test]
        fn rejects_unknown_string_variant() {
            let err =
                serde_json::from_str::<EdLinesImbalanceGatePolicy>("\"AutoMagic\"").unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("AutoMagic"), "error message: {msg}");
        }

        #[test]
        fn round_trip_via_profile_json() {
            // Patch the gate field of a shipped profile to exercise the full
            // ProfileJson → QuadJson → DetectorConfig path.
            let template = super::super::shipped_profile_json("high_accuracy").unwrap();
            for (input, expected) in [
                ("\"Enabled\"", EdLinesImbalanceGatePolicy::Enabled),
                ("\"Disabled\"", EdLinesImbalanceGatePolicy::Disabled),
                ("true", EdLinesImbalanceGatePolicy::Enabled),
                ("false", EdLinesImbalanceGatePolicy::Disabled),
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

    #[test]
    fn test_adaptive_ppb_default_valid() {
        let config = DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig::default()),
            ..DetectorConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_adaptive_ppb_rejects_degenerate() {
        let config = DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                low_extraction: QuadExtractionMode::ContourRdp,
                high_extraction: QuadExtractionMode::ContourRdp,
                ..AdaptivePpbConfig::default()
            }),
            ..DetectorConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::AdaptivePolicyDegenerate)
        ));
    }

    #[test]
    fn test_adaptive_ppb_rejects_threshold_out_of_range() {
        for bad in [0.5_f32, 1.0, 5.0, 10.0] {
            let config = DetectorConfig {
                quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                    threshold: bad,
                    ..AdaptivePpbConfig::default()
                }),
                ..DetectorConfig::default()
            };
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
        let config = DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                low_extraction: QuadExtractionMode::ContourRdp,
                high_extraction: QuadExtractionMode::EdLines,
                low_refinement: CornerRefinementMode::Edge,
                high_refinement: CornerRefinementMode::Erf,
                threshold: 2.5,
            }),
            ..DetectorConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::EdLinesIncompatibleWithErf)
        ));
    }

    #[test]
    fn test_adaptive_ppb_per_route_edlines_soft_rejected() {
        let config = DetectorConfig {
            decode_mode: DecodeMode::Soft,
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                low_extraction: QuadExtractionMode::EdLines,
                high_extraction: QuadExtractionMode::ContourRdp,
                low_refinement: CornerRefinementMode::None,
                high_refinement: CornerRefinementMode::Edge,
                threshold: 2.5,
            }),
            ..DetectorConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(crate::error::ConfigError::EdLinesIncompatibleWithSoftDecode)
        ));
    }

    #[test]
    fn test_any_route_uses_edlines() {
        let base = DetectorConfig::default();
        assert!(!base.any_route_uses_edlines());

        let edlines_static = DetectorConfig {
            quad_extraction_mode: QuadExtractionMode::EdLines,
            refinement_mode: CornerRefinementMode::None,
            ..DetectorConfig::default()
        };
        assert!(edlines_static.any_route_uses_edlines());

        let adaptive_with_edlines = DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig::default()),
            ..DetectorConfig::default()
        };
        assert!(adaptive_with_edlines.any_route_uses_edlines());

        let adaptive_no_edlines = DetectorConfig {
            quad_extraction_policy: QuadExtractionPolicy::AdaptivePpb(AdaptivePpbConfig {
                low_extraction: QuadExtractionMode::ContourRdp,
                high_extraction: QuadExtractionMode::ContourRdp,
                low_refinement: CornerRefinementMode::Erf,
                high_refinement: CornerRefinementMode::Gwlf,
                threshold: 2.5,
            }),
            ..DetectorConfig::default()
        };
        // Degenerate policy — validate would reject, but any_route_uses_edlines
        // is a pure accessor that doesn't validate.
        assert!(!adaptive_no_edlines.any_route_uses_edlines());
    }
}
