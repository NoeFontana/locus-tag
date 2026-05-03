# Algorithms & Mathematical Foundations

This document provides rigorous mathematical descriptions of the core solvers in Locus. For the pipeline data flow, see [Pipeline](pipeline.md). For memory layout, see [Memory Model](memory_model.md).

---

## 1. Homography Estimation (DLT)

**Module:** `homography.rs`

A homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ is the projective transformation mapping points from canonical tag space to image pixels:

$$\tilde{\mathbf{p}}_{\text{img}} = \mathbf{H} \, \tilde{\mathbf{p}}_{\text{tag}}$$

where $\tilde{\mathbf{p}} = [x, y, 1]^T$ are homogeneous coordinates.

### 1.1 General DLT (`from_pairs`)

Given 4 point correspondences $(\mathbf{s}_i, \mathbf{d}_i)$, the Direct Linear Transform constructs an $8 \times 9$ matrix $\mathbf{A}$ where each correspondence contributes two rows:

$$\mathbf{A}_{2i} = \begin{bmatrix} -s_x & -s_y & -1 & 0 & 0 & 0 & s_x d_x & s_y d_x & d_x \end{bmatrix}$$

$$\mathbf{A}_{2i+1} = \begin{bmatrix} 0 & 0 & 0 & -s_x & -s_y & -1 & s_x d_y & s_y d_y & d_y \end{bmatrix}$$

Setting $h_{33} = 1$ (dehomogenization), the remaining 8 parameters are solved via LU decomposition of the reduced $8 \times 8$ system:

$$\mathbf{M} \mathbf{h} = \mathbf{b}$$

A post-hoc reprojection check validates $\|\mathbf{H} \tilde{\mathbf{s}}_i - \tilde{\mathbf{d}}_i\|^2 < 10^{-4}$ for all 4 points, rejecting degenerate configurations.

### 1.2 Optimized Square-to-Quad (`square_to_quad`)

When the source points are the canonical unit square $\{(-1,-1), (1,-1), (1,1), (-1,1)\}$, the coefficient matrix $\mathbf{M}$ has a fixed sparsity pattern. The implementation hardcodes these coefficients, eliminating the generic source-point loop while producing an identical $8 \times 8$ LU solve.

### 1.3 Digital Differential Analyzer (DDA)

For grid-based sampling (decoding), recomputing $\mathbf{H} \tilde{\mathbf{p}}$ per pixel is wasteful. The DDA exploits the linearity of the homogeneous numerators:

$$n_x(u,v) = h_{00} u + h_{01} v + h_{02}, \quad n_y(u,v) = h_{10} u + h_{11} v + h_{12}, \quad d(u,v) = h_{20} u + h_{21} v + h_{22}$$

Stepping by $(\Delta u, \Delta v)$ in tag space requires only 3 additions per pixel:

$$n_x \mathrel{+}= h_{00} \Delta u, \quad n_y \mathrel{+}= h_{10} \Delta u, \quad d \mathrel{+}= h_{20} \Delta u$$

The perspective divide $\mathbf{p}_{\text{img}} = (n_x/d, \; n_y/d)$ is computed using a hardware reciprocal approximation (`rcp_nr`) with one Newton-Raphson refinement step for ~23-bit accuracy.

---

## 2. ERF Sub-pixel Edge Refinement

**Module:** `edge_refinement.rs`

### 2.1 Intensity Model

The observed intensity along the normal to a tag edge is modeled as a PSF-blurred step function:

$$I(x,y) = \frac{A+B}{2} + \frac{B-A}{2} \cdot \operatorname{erf}\!\left(\frac{d}{\sigma}\right)$$

where:

- $A, B$ are the dark and light intensities on either side of the edge.
- $d$ is the signed perpendicular distance from the sample point to the edge line.
- $\sigma$ is the Gaussian blur parameter (PSF width).
- $\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$ is the error function.

### 2.2 Edge Parameterization

The edge line is parameterized by its perpendicular offset $\rho$ from an initial estimate:

$$d_i = \mathbf{n} \cdot (\mathbf{p}_i - \mathbf{p}_0) - \rho$$

where $\mathbf{n}$ is the unit normal to the edge and $\mathbf{p}_0$ is a point on the initial edge estimate.

### 2.3 Gauss-Newton Optimization

The scalar parameter $\rho$ is refined by minimizing the sum of squared residuals:

$$\min_\rho \sum_i \left[ I_i - \frac{A+B}{2} - \frac{B-A}{2} \cdot \operatorname{erf}\!\left(\frac{d_i - \rho}{\sigma}\right) \right]^2$$

The Jacobian for each sample is:

$$J_i = \frac{\partial r_i}{\partial \rho} = \frac{B-A}{\sigma} \cdot \frac{1}{\sqrt{\pi}} \exp\!\left(-\left(\frac{d_i - \rho}{\sigma}\right)^2\right)$$

The Gauss-Newton update is:

$$\Delta\rho = \frac{\sum_i J_i \, r_i}{\sum_i J_i^2}$$

with convergence when $|\Delta\rho| < 10^{-4}$ or after 15 iterations.

### 2.4 Configuration Variants

| Variant | A/B Estimation | Min Contrast | Use Case |
| :--- | :--- | :--- | :--- |
| **Quad-style** | Once before GN loop | None | Quad corner refinement |
| **Decoder-style** | Per iteration | Early exit if $|B-A|$ too low | Edge refinement during decoding |

The `erf_approx` function is shared across both variants and has a SIMD-vectorized 4-wide version (`erf_approx_v4`) for batch evaluation.

---

## 3. Gradient-Weighted Line Fitting (GWLF)

**Module:** `gwlf.rs`

### 3.1 Moment Accumulation

For each of the 4 quad edges, image gradient magnitudes serve as weights in a spatial moment accumulation:

$$\bar{x} = \frac{\sum w_i x_i}{\sum w_i}, \quad \bar{y} = \frac{\sum w_i y_i}{\sum w_i}$$

The $2 \times 2$ gradient-weighted covariance matrix is:

$$\mathbf{C} = \begin{bmatrix} \sum w_i (x_i - \bar{x})^2 & \sum w_i (x_i - \bar{x})(y_i - \bar{y}) \\ \sum w_i (x_i - \bar{x})(y_i - \bar{y}) & \sum w_i (y_i - \bar{y})^2 \end{bmatrix} \cdot \frac{1}{\sum w_i}$$

### 3.2 Line Fitting via PCA

The edge direction is the eigenvector of $\mathbf{C}$ corresponding to $\lambda_{\max}$ (the tangent direction). The eigenvector corresponding to $\lambda_{\min}$ gives the edge normal $\mathbf{n}$.

The line is represented in homogeneous form $\mathbf{l} = [n_x, n_y, d]^T$ where $d = -(n_x \bar{x} + n_y \bar{y})$.

The eigendecomposition of the $2 \times 2$ symmetric matrix is solved analytically:

$$\lambda_{\max/\min} = \frac{\text{tr}(\mathbf{C}) \pm \sqrt{\text{tr}(\mathbf{C})^2 - 4 \det(\mathbf{C})}}{2}$$

### 3.3 Line Covariance Propagation

The $3 \times 3$ covariance $\boldsymbol{\Sigma}_{\mathbf{l}}$ of the homogeneous line parameters is propagated through the PCA fitting procedure, encoding the uncertainty in the edge direction and offset.

### 3.4 Corner as Line Intersection

The refined corner is the intersection of two adjacent homogeneous lines:

$$\tilde{\mathbf{c}} = \mathbf{l}_1 \times \mathbf{l}_2$$

The covariance of the homogeneous intersection point is propagated via the cross-product Jacobians:

$$\boldsymbol{\Sigma}_{\tilde{\mathbf{c}}} = [\mathbf{l}_2]_\times \, \boldsymbol{\Sigma}_{\mathbf{l}_1} \, [\mathbf{l}_2]_\times^T + [\mathbf{l}_1]_\times \, \boldsymbol{\Sigma}_{\mathbf{l}_2} \, [\mathbf{l}_1]_\times^T$$

where $[\mathbf{l}]_\times$ is the skew-symmetric cross-product matrix.

### 3.5 Perspective Division to Affine Coordinates

The Cartesian corner $\mathbf{c} = (\tilde{c}_x / \tilde{c}_w, \; \tilde{c}_y / \tilde{c}_w)$ and its $2 \times 2$ covariance are obtained via the Jacobian of perspective division:

$$\mathbf{J}_\pi = \frac{1}{w} \begin{bmatrix} 1 & 0 & -x/w \\ 0 & 1 & -y/w \end{bmatrix}, \qquad \boldsymbol{\Sigma}_{\mathbf{c}} = \mathbf{J}_\pi \, \boldsymbol{\Sigma}_{\tilde{\mathbf{c}}} \, \mathbf{J}_\pi^T$$

This $2 \times 2$ corner covariance feeds directly into the Accurate mode pose estimator as per-corner uncertainty.

### 3.6 Adaptive Transversal Windowing

The perpendicular search band for gradient sampling scales with edge length $L$:

$$\text{half-width} = \max(2, \; 0.01 \cdot L) \text{ pixels}$$

This prevents over-smoothing on short edges while capturing sufficient gradient evidence on long edges.

---

## 4. IPPE-Square Pose Estimation

**Module:** `pose.rs`

### 4.1 Homography Normalization

The pixel-space homography $\mathbf{H}_\text{pixel}$ from `square_to_quad` is normalized by the inverse intrinsics:

$$\mathbf{H}_\text{norm} = \mathbf{K}^{-1} \mathbf{H}_\text{pixel}$$

The metric homography accounting for tag size $s$ and the modern OpenCV top-left origin convention:

$$\mathbf{H}_\text{metric} = \left[\frac{2}{s}\mathbf{h}_1, \; \frac{2}{s}\mathbf{h}_2, \; \mathbf{h}_3 - \mathbf{h}_1 - \mathbf{h}_2\right]$$

### 4.2 Jacobian SVD Decomposition

The Jacobian $\mathbf{J} = [\mathbf{h}_1, \mathbf{h}_2]$ encodes the image-plane stretch. Its $2 \times 2$ Gram matrix:

$$\mathbf{B} = \mathbf{J}^T \mathbf{J} = \begin{bmatrix} a & c \\ c & b \end{bmatrix}$$

has eigenvalues $\sigma_1^2, \sigma_2^2$ (singular values squared) computed analytically:

$$\sigma_{1,2}^2 = \frac{\text{tr}(\mathbf{B}) \pm \sqrt{\text{tr}(\mathbf{B})^2 - 4 \det(\mathbf{B})}}{2}$$

### 4.3 Dual Pose Solutions

IPPE-Square produces **two** candidate poses from the SVD, corresponding to the two minima of the planar PnP error surface (the "Necker reversal" ambiguity). The candidate with the lower reprojection error is selected.

**Frontal degeneracy:** When $|\sigma_1 - \sigma_2| < 10^{-4} \sigma_1$, the two solutions collapse. In this case, Gram-Schmidt orthonormalization of $[\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_1 \times \mathbf{h}_2]$ produces the rotation, and the translation scale $\gamma = (\sigma_1 + \sigma_2)/2$.

### 4.4 Orthogonalization

The raw rotation estimate from the SVD is projected onto $SO(3)$ via polar decomposition:

$$\mathbf{R} = \mathbf{U} \mathbf{V}^T, \quad \text{from SVD}(\mathbf{R}_\text{raw}) = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

ensuring $\det(\mathbf{R}) = +1$.

---

## 5. Levenberg-Marquardt Pose Refinement

### 5.1 Fast Mode (Geometric Error)

**Module:** `pose.rs` | **Function:** `refine_pose_lm`

#### Objective

Minimize Huber-robust geometric reprojection error:

$$\min_{\boldsymbol{\xi}} \sum_{i=1}^{4} \rho_H\!\left(\|\mathbf{r}_i\|, \delta\right)$$

where $\mathbf{r}_i = \mathbf{p}_i^\text{obs} - \pi(\mathbf{R} \mathbf{P}_i + \mathbf{t})$ is the 2D reprojection residual and $\pi$ is the pinhole projection.

#### Huber Loss

$$\rho_H(r, \delta) = \begin{cases} \frac{1}{2} r^2 & r \leq \delta \\ \delta(r - \frac{1}{2}\delta) & r > \delta \end{cases}$$

Applied per corner as an IRLS weight: $w_i = \min(1, \; \delta / \|\mathbf{r}_i\|)$.

#### Jacobian Structure

The $2 \times 6$ Jacobian for corner $i$ with respect to the $\mathfrak{se}(3)$ twist $\boldsymbol{\xi} = [\mathbf{t}, \boldsymbol{\omega}]^T$:

$$\mathbf{J}_i = \frac{\partial \pi}{\partial \mathbf{P}_\text{cam}} \cdot \frac{\partial \mathbf{P}_\text{cam}}{\partial \boldsymbol{\xi}}$$

where:

$$\frac{\partial \pi}{\partial \mathbf{P}_\text{cam}} = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{bmatrix}$$

$$\frac{\partial \mathbf{P}_\text{cam}}{\partial \boldsymbol{\xi}} = \begin{bmatrix} \mathbf{I}_3 & -[\mathbf{P}_\text{cam}]_\times \end{bmatrix}$$

#### Normal Equations with Marquardt Scaling

$$({\mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \cdot \text{diag}(\mathbf{J}^T \mathbf{W} \mathbf{J})}) \, \boldsymbol{\delta} = \mathbf{J}^T \mathbf{W} \mathbf{r}$$

Marquardt diagonal scaling damps each DOF proportionally to its own curvature, correcting the scale mismatch between translational and rotational gradient magnitudes. Solved via Cholesky decomposition.

#### Trust-Region (Nielsen Schedule)

- **Accept:** Gain ratio $\rho = \text{actual}/\text{predicted} > 0$ triggers $\lambda \leftarrow \lambda \cdot \max(1/3, \; 1 - (2\rho-1)^3)$, $\nu \leftarrow 2$.
- **Reject:** $\lambda \leftarrow \lambda \cdot \nu$, $\nu \leftarrow 2\nu$.

#### Manifold Update

The $\mathfrak{se}(3)$ increment is applied via the exponential map:

$$\mathbf{R} \leftarrow \exp([\boldsymbol{\omega}]_\times) \cdot \mathbf{R}, \qquad \mathbf{t} \leftarrow \mathbf{t} + \boldsymbol{\delta}_t$$

The rotation update uses `UnitQuaternion::from_scaled_axis` for numerically stable $SO(3)$ integration.

#### Convergence

- Gradient convergence: $\|\mathbf{J}^T \mathbf{W} \mathbf{r}\|_\infty < 10^{-8}$
- Step convergence: $\|\boldsymbol{\delta}\| < 10^{-8}$
- Maximum 20 iterations (typically exits in 3-6).

---

### 5.2 Accurate Mode (Mahalanobis Distance)

**Module:** `pose_weighted.rs` | **Function:** `refine_pose_lm_weighted`

#### Objective

Minimize Huber-robust Mahalanobis distance using per-corner information matrices:

$$\min_{\boldsymbol{\xi}} \sum_{i=1}^{4} \rho_H\!\left(s_i, k\right)$$

where:

$$s_i = \sqrt{\mathbf{r}_i^T \mathbf{W}_i \mathbf{r}_i}$$

is the Mahalanobis distance and $\mathbf{W}_i = \boldsymbol{\Sigma}_i^{-1}$ is the information matrix (inverse of the $2 \times 2$ corner covariance from the Structure Tensor or GWLF).

#### Huber-on-Mahalanobis IRLS

The IRLS weight in the Mahalanobis metric:

$$w(s_i) = \min\!\left(1, \; \frac{k}{s_i}\right), \quad k = 1.345$$

The augmented information matrix: $\tilde{\mathbf{W}}_i = w(s_i) \cdot \mathbf{W}_i$

The threshold $k = 1.345$ provides 95% asymptotic efficiency under Gaussian noise.

#### Normal Equations

$$\left(\sum_i \mathbf{J}_i^T \tilde{\mathbf{W}}_i \mathbf{J}_i + \lambda \mathbf{D}\right) \boldsymbol{\delta} = \sum_i \mathbf{J}_i^T \tilde{\mathbf{W}}_i \mathbf{r}_i$$

The Jacobian $\mathbf{J}_i$ has the same structure as in Fast mode.

#### Corner Uncertainty Sources

| Source | Method | Module |
| :--- | :--- | :--- |
| **Structure Tensor** | $\boldsymbol{\Sigma}_c \approx \sigma_n^2 \mathbf{S}^{-1}$ where $\mathbf{S}$ is the Sobel-based structure tensor | `pose_weighted.rs` |
| **GWLF Propagation** | Formal covariance propagation through PCA line fitting and homogeneous intersection | `gwlf.rs` |

#### Gain-Scheduled Tikhonov Regularization

For severely foreshortened tags (grazing angles), the structure tensor becomes ill-conditioned. A gain-scheduled regularizer prevents the information matrix from exploding:

$$\boldsymbol{\Sigma}_{\text{reg}} = \sigma_n^2 \mathbf{S}^{-1} + \alpha(R) \cdot \mathbf{I}$$

where the anisotropy ratio $R = \lambda_{\min} / \lambda_{\max}$ and:

$$\alpha(R) = \alpha_{\max} \cdot (1 - R)^2$$

The quadratic transfer function keeps $\alpha \approx 0$ for well-conditioned frontal tags ($R \to 1$) and smoothly ramps to $\alpha_{\max}$ for degenerate configurations ($R \to 0$).

#### Covariance Output

At convergence, the $6 \times 6$ pose covariance (Cramer-Rao lower bound) is extracted from the final normal equations:

$$\boldsymbol{\Sigma}_\text{pose} = \left(\sum_i \mathbf{J}_i^T \tilde{\mathbf{W}}_i \mathbf{J}_i\right)^{-1}$$

This encodes the full translational and rotational uncertainty and is returned alongside the refined pose.

---

## 6. Decoding Strategies

**Module:** `decoder.rs`, `strategy.rs`

Each bit cell is sampled at its grid center via the homography DDA. The sampled intensity is compared against the local adaptive threshold to produce a binary code. Dictionary lookup is $O(1)$ via precomputed Hamming distance tables.

---

## 7. Board-Level Pose Estimation

**Module:** `board.rs`, `charuco.rs`

Board pose estimation aggregates evidence from multiple detected tags into a single, more precise 6-DOF board pose. The solver backend (`RobustPoseSolver`) is shared by both board types; the two paths differ only in how they assemble the point correspondences.

---

### 7.1 AprilGrid: Tag-Corner Correspondences

**Struct:** `BoardEstimator`

Each visible tag contributes 4 point correspondences: its refined image corners paired with pre-computed 3D board-frame coordinates from `AprilGridTopology::obj_points`.

The `group_size=4` parameter tells `RobustPoseSolver` that these 4 points belong to a rigid group — RANSAC hypotheses are drawn from whole tags, not individual corners, preventing degenerate single-tag hypotheses.

**Seed poses** are derived from each tag's individual Stage-5 pose by subtracting the tag's board-frame origin:

$$\mathbf{t}_{\text{board}} = \mathbf{t}_{\text{tag}} - \mathbf{R}_{\text{tag}} \, \mathbf{o}_{\text{tag}}$$

where $\mathbf{o}_{\text{tag}}$ is the tag's top-left corner in board coordinates.

---

### 7.2 ChAruco: Saddle-Point Correspondences

**Struct:** `CharucoRefiner`

ChAruco boards embed ArUco markers in alternating cells of a checkerboard. The interior checkerboard corners — *saddle points* — are sharper, more photometrically stable features than tag corners and are used as the primary measurement for board pose.

#### Board Geometry (Two-Layer Model)

| Layer | Feature | Coordinate Source |
| :--- | :--- | :--- |
| **A — Tags** | ArUco markers occupying cells where $(r+c)$ is even. Each tag fills `marker_length × marker_length` within its `square_length × square_length` cell. | `CharucoTopology::obj_points` |
| **B — Saddles** | Interior checkerboard corners at the outer corners of each square. Indexed as `sr*(cols-1)+sc` for grid position `(sr, sc)`. | `CharucoTopology::saddle_points` |

The white padding margin between a tag edge and its enclosing square corner is:

$$\delta = \frac{\text{square\_length} - \text{marker\_length}}{2}$$

This means saddle points are **outside** the tag's physical boundary — they cannot be obtained by interpolating the tag's own corners.

#### Saddle Prediction via Homography Extrapolation

Each tag stores a $3 \times 3$ homography mapping canonical coordinates $[-1, 1]^2$ to the tag's image corners. The canonical coordinates of a saddle at board position $(s_x, s_y)$ relative to a tag with top-left corner $(t_x, t_y)$ are:

$$u = \frac{2(s_x - t_x)}{\text{marker\_length}} - 1, \qquad v = \frac{2(s_y - t_y)}{\text{marker\_length}} - 1$$

Because the saddle is at the outer corner of the *enclosing square* (not the marker), $|u| > 1$ or $|v| > 1$ — the homography is evaluated *outside* its training domain. This is a valid linear extrapolation for the projective model and correctly maps to the saddle's image location.

The predicted image coordinates are:

$$\begin{bmatrix} x_h \\ y_h \\ w_h \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad (p_x, p_y) = (x_h / w_h, \; y_h / w_h)$$

#### Gauss-Newton Saddle Refinement

Starting from the predicted position, a Gauss-Newton iteration drives the gradient to zero using the structure tensor as a surrogate Hessian. This is the standard sub-pixel saddle localization technique (Harris/Shi-Tomasi variant):

$$\mathbf{p}^{(k+1)} = \mathbf{p}^{(k)} - \mathbf{S}^{-1}(\mathbf{p}^{(k)}) \, \nabla I(\mathbf{p}^{(k)})$$

where the structure tensor $\mathbf{S}$ is accumulated over a $\text{radius} \times \text{radius}$ window using 3×3 Sobel kernels:

$$\mathbf{S} = \sum_{\mathbf{q} \in \mathcal{W}} \begin{bmatrix} g_x^2 & g_x g_y \\ g_x g_y & g_y^2 \end{bmatrix}\bigg|_\mathbf{q}$$

Convergence is declared after at most 5 iterations. A saddle is rejected if:

- $\|\mathbf{p}^{(k+1)} - \mathbf{p}^{(0)}\| > \text{max\_drift\_px}$ (default 5.0 px), or
- $\det(\mathbf{S}) < 10^{-3}$ (degenerate, non-corner region).

#### LO-RANSAC + AW-LM

Accepted saddles feed `RobustPoseSolver` with `group_size=1` (each saddle is independent). The solver runs LO-RANSAC followed by anisotropically-weighted Levenberg-Marquardt using the inverse structure tensor as the per-point information matrix. Returns `None` if fewer than 4 saddles survive.

**Memory model**: all scratch buffers (`img`, `obj`, `info`, `seeds`, `seen`, `touched`) are pre-allocated in `CharucoRefiner::new()`. `estimate()` performs zero heap allocations.
