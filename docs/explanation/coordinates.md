# Coordinate Systems

This page describes the coordinate systems used by **Locus** for detections and pose estimation. Consistency in these conventions is critical for downstream applications in robotics and computer vision.

## 1. 2D Image Coordinates

Locus follows the standard image coordinate system used by OpenCV and major computer vision benchmarks.

- **Origin (0,0)**: The top-left corner of the image.
- **X-axis**: Increases from left to right.
- **Y-axis**: Increases from top to bottom.

### Pixel Center Convention
Locus uses the **pixel-center at 0.5** convention.
- The center of the top-left pixel is $(0.5, 0.5)$.
- The top-left corner of the image (the boundary of the first pixel) is $(0.0, 0.0)$.

This convention is strictly followed to ensure that sub-pixel corner refinement and RMSE calculations are compatible with standard computer vision datasets like **ICRA 2020**.

!!! info "ICRA 2020 Parity"
    While Locus uses the same pixel coordinate system as the ICRA 2020 benchmark, the **corner indexing** differs due to winding conventions:
    - **ICRA 2020**: Uses UMich-style Counter-Clockwise (CCW) winding starting from the bottom-left $[BL, BR, TR, TL]$.
    - **Locus**: Uses Modern OpenCV Clockwise (CW) winding starting from the top-left $[TL, TR, BR, BL]$.
    - **Orientation**: The ICRA dataset has a 180-degree bit-rotation offset relative to Locus's row-major 0-degree orientation.

## 2. Tag Local Coordinates (Object Space)

When estimating 3D pose, the tag is defined in a right-handed local coordinate system where the tag lies on the $Z=0$ plane. This follows the **modern OpenCV (4.6+)** convention (`ARUCO_CW_TOP_LEFT_CORNER`).

For a tag of physical size $s$ (e.g., in meters):

- **Origin (0,0,0)**: The **Top-Left** corner of the tag.
- **X-axis**: Points to the right, along the top edge.
- **Y-axis**: Points downward, along the left edge.
- **Z-axis**: Points **into** the scene (away from the camera), established by the right-hand rule ($X \times Y$).

### Corner Ordering
Detections return corners in **clockwise order** (when looking at the tag), matching the following object-space coordinates:

| Index | Location | Object Space Coordinates $(X, Y, Z)$ |
| :--- | :--- | :--- |
| **0** | Top-Left | $(0, 0, 0)$ |
| **1** | Top-Right | $(s, 0, 0)$ |
| **2** | Bottom-Right | $(s, s, 0)$ |
| **3** | Bottom-Left | $(0, s, 0)$ |

## 3. Tag Layout and Bit Order

Locus strictly adheres to **modern OpenCV (cv2.aruco)** conventions for dictionary layout, bit ordering, and canonical orientation.

### Row-Major Bit Ordering
For all supported families (AprilTag 16h5, 36h11, and ArUco 4x4), the binary payload is extracted in **row-major order**:
- **Bit 0**: Top-left data cell.
- **Bit N-1**: Bottom-right data cell.

This matches the internal memory layout of OpenCV's `bytesList`.

### Canonical Orientation (0 Degrees)
The 0-degree ("canonical") orientation of a tag is defined such that the bits are read from top-left to bottom-right without any rotation.

!!! warning "Migration Warning: Orientation Shift"
    Users migrating from the legacy **UMich C library** or `apriltag` Python package should note that Locus's 0-degree orientation is shifted by **90 degrees clockwise** relative to UMich. 
    - **UMich 0-deg**: Origin at bottom-left (spiral bit order).
    - **Locus/OpenCV 0-deg**: Origin at top-left (row-major bit order).

    If you are comparing pose outputs ($R, t$) against UMich-based ground truth, you may need to apply a 90-degree Z-axis rotation to the results.

## 4. 3D Camera Coordinate System

Locus utilizes a **right-handed** coordinate system for the camera, consistent with the standard pinhole camera model.

- **Origin**: The optical center (focal point) of the camera.
- **Z-axis**: Points forward along the optical axis (depth).
- **X-axis**: Points to the right.
- **Y-axis**: Points downwards.

### Projection Model
A 3D point $P = (X, Y, Z)$ in the camera frame is projected to image coordinates $(u, v)$ using the focal lengths $(f_x, f_y)$ and the principal point $(c_x, c_y)$:

$$
u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y
$$

This model assumes a rectified image (zero distortion). If your camera has significant lens distortion, you should undistort the image or the corner points before passing them to the pose estimator.

## 5. Pose Representation

A pose $(R, t)$ returned by Locus transforms a point $P_{object}$ from **Tag Local Coordinates** to **Camera Coordinates** $P_{camera}$:

$$
P_{camera} = R \cdot P_{object} + t
$$

- $R$: A $3 \times 3$ rotation matrix.
- $t$: A $3 \times 1$ translation vector (representing the tag's **top-left corner** in the camera frame).
