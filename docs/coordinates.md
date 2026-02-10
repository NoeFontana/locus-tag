# Coordinate Systems

This page describes the coordinate systems used by **Locus** for detections and pose estimation. Consistency in these conventions is critical for downstream applications in robotics and computer vision.

## 1. 2D Image Coordinates

Locus follows the standard image coordinate system used by OpenCV and the `UMich/ICRA 2020` benchmark.

- **Origin (0,0)**: The top-left corner of the image.
- **X-axis**: Increases from left to right.
- **Y-axis**: Increases from top to bottom.

### Pixel Center Convention
Locus uses the **pixel-center at 0.5** convention. 
- The center of the top-left pixel is $(0.5, 0.5)$.
- The top-left corner of the image (the boundary of the first pixel) is $(0.0, 0.0)$.

This convention is strictly followed to ensure that sub-pixel corner refinement and RMSE calculations are compatible with standard computer vision datasets.

## 2. Tag Local Coordinates (Object Space)

When estimating 3D pose, the tag is defined in a right-handed local coordinate system where the tag lies on the $Z=0$ plane. For a tag of physical size $s$ (e.g., in meters):

- **Origin (0,0,0)**: The geometric center of the tag.
- **Z-axis**: Points outward from the tag face (the "normal" of the tag).
- **X-axis**: Points to the right relative to the tag's orientation.
- **Y-axis**: Points downward relative to the tag's orientation.

### Corner Ordering
Detections return corners in a specific counter-clockwise order (when looking at the tag), matching the following object-space coordinates:

| Index | Location | Object Space Coordinates $(X, Y, Z)$ |
| :--- | :--- | :--- |
| **0** | Top-Left | $(-s/2, -s/2, 0)$ |
| **1** | Top-Right | $(s/2, -s/2, 0)$ |
| **2** | Bottom-Right | $(s/2, s/2, 0)$ |
| **3** | Bottom-Left | $(-s/2, s/2, 0)$ |

## 3. 3D Camera Coordinate System

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

## 4. Pose Representation

A pose $(R, t)$ returned by Locus transforms a point $P_{object}$ from **Tag Local Coordinates** to **Camera Coordinates** $P_{camera}$:

$$
P_{camera} = R \cdot P_{object} + t
$$

- $R$: A $3 \times 3$ rotation matrix.
- $t$: A $3 \times 1$ translation vector (representing the tag's center in the camera frame).
