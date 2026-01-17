# Coordinate Systems

This page describes the coordinate systems used by **Locus** for detections and pose estimation. Understanding these conventions is critical for downstream applications in robotics, computer vision, and augmented reality.

## 1. 2D Image Coordinates

Locus follows the standard image coordinate system used by OpenCV and most computer vision libraries.

- **Origin (0,0)**: The top-left corner of the image.
- **X-axis**: Increases from left to right.
- **Y-axis**: Increases from top to bottom.

### Pixel Center Convention
Locus uses the **pixel-center at 0.5** convention for its final detection output. 
- The center of the top-left pixel is `(0.5, 0.5)`.
- The range of the top-left pixel is `[0.0, 1.0)` in both dimensions.

This convention ensures compatibility with the **UMich/ICRA 2020 AprilTag benchmark** and other standard datasets.

## 2. Tag Local Coordinates (Object Space)

When estimating a 3D pose, the tag is defined in its own local coordinate system (object space) on the $Z=0$ plane. For a tag of physical size $S$ (e.g., in meters):

- **Origin (0,0,0)**: The geometric center of the tag.
- **Z-axis**: Points outward from the tag (away from the surface it is mounted on).
- **X/Y Plane**: The tag face lies on the $Z=0$ plane.

### Corner Ordering
Detections return corners in the following order (matches object space points):
1. **Corner 0**: Top-Left (`-S/2, -S/2`)
2. **Corner 1**: Top-Right (`S/2, -S/2`)
3. **Corner 2**: Bottom-Right (`S/2, S/2`)
4. **Corner 3**: Bottom-Left (`-S/2, S/2`)

## 3. 3D Camera Coordinate System

Locus uses a right-handed coordinate system for the camera, matching the pinhole camera model.

- **Origin**: The optical center (projection center) of the camera.
- **Z-axis**: Points forward (optical axis, depth).
- **X-axis**: Points to the right.
- **Y-axis**: Points downwards.

### Projection Model
A 3D point $(X, Y, Z)$ in camera coordinates projects to image coordinates $(u, v)$ as:

$$
u = \frac{X}{Z} \cdot f_x + c_x
$$
$$
v = \frac{Y}{Z} \cdot f_y + c_y
$$

Where $(f_x, f_y)$ are the focal lengths and $(c_x, c_y)$ is the principal point, all provided via `CameraIntrinsics`.
