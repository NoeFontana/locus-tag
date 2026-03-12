# Future Accuracy Improvements

After successfully diagnosing and fixing the sub-pixel coordinate bias (aligning our detections to the "0.5-is-center" convention) and seeing our corner error drop significantly in the 640x480 tests, there are still a few avenues for driving the Corner RMSE closer to theoretical perfection and unlocking improvements in our Reprojection RMSE:

## 1. Joint Corner Optimization (Template Fitting)

**Current State:** The pipeline refines a corner by independently fitting two straight 1D lines to the adjacent edges and intersecting them.
**The Problem:** Right at the vertex, the pixel intensities are a complex 2D mixture of both edges meeting. Independent 1D line fits ignore this mixed corner region, losing critical information where the curve is sharpest.
**Next Step:** Implement a true 2D corner refinement algorithm (like a Gauss-Newton optimization of a 2D L-shape template) that optimizes the `(x, y)` position of the vertex simultaneously using the pixels *exactly* at the corner.

## 2. Dynamic Point Spread Function (PSF) Estimation

**Current State:** The `Erf` sub-pixel refinement models camera blur using a static `sigma` (likely tuned for a generic camera).
**The Problem:** Blur varies wildly due to motion, out-of-focus optics, or depth-of-field variations across the image plane. A static `sigma` will under-fit blurry tags and over-fit sharp ones.
**Next Step:** Add a quick gradient-spread analysis pass to estimate the local `sigma` for each edge dynamically before applying the `Erf` refinement.

## 3. Pose Estimator Tuning (The Reprojection Mystery)

**Current State:** Corner RMSE (direct 2D distance to ground truth) dropped by 17%, but the Reprojection RMSE stayed stubbornly static (~2.86px). 
**The Problem:** This indicates a bottleneck in Phase 4 (Pose Estimation). The PnP solver (likely IPPE or a custom homography decomposition) isn't fully capitalizing on the more accurate 2D coordinates.
**Next Step:** Investigate the pose estimation stage. It may be prematurely terminating its non-linear refinement, lacking robust outlier rejection, or using an outdated focal length assumption that the updated corners are now highlighting.

## 4. Lens Distortion & Ray-Space Refinement

**Current State:** Refinement operates purely in 2D distorted pixel space.
**The Problem:** If real-world cameras have radial/tangential distortion, straight edges in the physical world are curved in the image. Fitting straight lines to curved edges introduces inherent geometric error.
**Next Step:** Push the sub-pixel refinement into undistorted space, or refine using the camera intrinsics directly by projecting 3D rays.
