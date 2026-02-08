#![allow(missing_docs)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]

use divan::bench;
use locus_core::pose::{CameraIntrinsics, estimate_tag_pose};
use nalgebra::{Matrix3, Vector3};

use locus_core::config::PoseEstimationMode;

fn main() {
    divan::main();
}

#[bench]
fn bench_pose_estimation(bencher: divan::Bencher) {
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 400.0, 300.0);
    let tag_size = 0.16;
    let s = tag_size * 0.5;

    // Construct a synthetic pose
    let r_gt = Matrix3::identity(); // Facing camera
    // T: 2m away, slightly offset
    let t_gt = Vector3::new(0.1, -0.1, 2.0);

    // Project points manually to get corners
    let obj_pts = [
        Vector3::new(-s, -s, 0.0),
        Vector3::new(s, -s, 0.0),
        Vector3::new(s, s, 0.0),
        Vector3::new(-s, s, 0.0),
    ];

    let mut corners = [[0.0; 2]; 4];
    for i in 0..4 {
        let p_cam = r_gt * obj_pts[i] + t_gt;
        let u = (p_cam.x / p_cam.z) * intrinsics.fx + intrinsics.cx;
        let v = (p_cam.y / p_cam.z) * intrinsics.fy + intrinsics.cy;
        corners[i] = [u, v];
    }

    bencher.bench_local(move || {
        divan::black_box(estimate_tag_pose(
            divan::black_box(&intrinsics),
            divan::black_box(&corners),
            divan::black_box(tag_size),
            divan::black_box(None),
            divan::black_box(PoseEstimationMode::Fast),
        ))
    });
}
