import argparse
import json

import datasets
import locus
import numpy as np
import rerun as rr
from huggingface_hub import hf_hub_download


def get_manifest(repo_id, subset):
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id, filename=f"{subset}/manifest.json", repo_type="dataset"
        )
        with open(manifest_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load manifest: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize Locus Tag Hub Dataset with Rerun")
    parser.add_argument("--subset", type=str, default="locus_v1_tag36h11_640x480")
    parser.add_argument("--repo-id", type=str, default="NoeFontana/locus-tag-bench")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--save", type=str, help="Save to .rrd file instead of spawning viewer")
    args = parser.parse_args()

    rr.init("locus_visualizer")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()  # Spawn the Rerun viewer

    print(f"Loading dataset {args.subset}...")
    ds = datasets.load_dataset(args.repo_id, args.subset, split="train", streaming=True)

    manifest = get_manifest(args.repo_id, args.subset)
    intrinsics = None
    tag_size = 0.16

    if manifest:
        if "camera_intrinsics" in manifest:
            intrinsics = locus.CameraIntrinsics(
                fx=manifest["camera_intrinsics"]["fx"],
                fy=manifest["camera_intrinsics"]["fy"],
                cx=manifest["camera_intrinsics"]["cx"],
                cy=manifest["camera_intrinsics"]["cy"],
            )
            print(
                f"Loaded intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, cx={intrinsics.cx}, cy={intrinsics.cy}"
            )
        if "tag_specification" in manifest:
            tag_size = manifest["tag_specification"]["tag_size_m"]
            print(f"Loaded tag size: {tag_size}m")

    detector = locus.Detector(
        families=[locus.TagFamily.AprilTag36h11],
        refinement_mode=locus.CornerRefinementMode.Erf,  # pyright: ignore[reportCallIssue]
    )

    if intrinsics:
        width = manifest["camera_intrinsics"].get("width", 640)
        height = manifest["camera_intrinsics"].get("height", 480)
        rr.log(
            "world/camera",
            rr.Pinhole(
                focal_length=[intrinsics.fx, intrinsics.fy],
                principal_point=[intrinsics.cx, intrinsics.cy],
                resolution=[width, height],
            ),
            static=True,
        )
    else:
        rr.log("world/camera", rr.Pinhole(resolution=[640, 480]), static=True)

    count = 0
    for item in ds:
        if count >= args.max_images:
            break

        rr.set_time(sequence=count, timeline="frame")

        pil_img = item["image"]
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        img_np = np.array(pil_img, dtype=np.uint8)

        # Log grayscale image as Image/Tensor
        rr.log("world/camera/image", rr.Image(img_np))

        tag_ids = item["tag_id"]
        corners_list = item["corners"]
        positions = item.get("position")
        rotations = item.get("rotation_quaternion")

        if not isinstance(tag_ids, list):
            tag_ids = [tag_ids]
            corners_list = [corners_list]
            if positions is not None:
                positions = [positions]
            if rotations is not None:
                rotations = [rotations]

        gt_corner_strips = []
        for corners in corners_list:
            gt_corner_strips.append(corners + [corners[0]])

        rr.log(
            "world/camera/gt_corners",
            rr.LineStrips2D(
                gt_corner_strips, colors=[[0, 255, 0]] * len(gt_corner_strips), radii=1.0
            ),
        )

        if (
            positions is not None
            and rotations is not None
            and all(p is not None for p in positions)
        ):
            for tid, pos, rot in zip(tag_ids, positions, rotations, strict=False):
                rr.log(
                    f"world/camera/gt_tag_{tid}",
                    rr.Transform3D(translation=pos, rotation=rr.Quaternion(xyzw=rot)),
                )
                rr.log(
                    f"world/camera/gt_tag_{tid}/axes",
                    rr.Arrows3D(
                        vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ),
                )

        # Locus Predictions
        batch = detector.detect(
            img_np,
            intrinsics=intrinsics,
            tag_size=tag_size,
            pose_estimation_mode=locus.PoseEstimationMode.Fast,
        )

        pred_corner_strips = []
        for i in range(len(batch)):
            c = batch.corners[i].tolist()
            pred_corner_strips.append(c + [c[0]])

        if pred_corner_strips:
            rr.log(
                "world/camera/pred_corners",
                rr.LineStrips2D(
                    pred_corner_strips, colors=[[255, 0, 0]] * len(pred_corner_strips), radii=1.0
                ),
            )

        for i in range(len(batch)):
            tid = batch.ids[i]
            pose = batch.poses[i]
            if pose is not None:
                # pose is a numpy array of shape (7,) tx, ty, tz, qx, qy, qz, qw
                tx, ty, tz, qx, qy, qz, qw = pose
                rr.log(
                    f"world/camera/pred_tag_{tid}",
                    rr.Transform3D(
                        translation=[tx, ty, tz], rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    ),
                )
                rr.log(
                    f"world/camera/pred_tag_{tid}/axes",
                    rr.Arrows3D(
                        vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
                        colors=[[255, 128, 0], [128, 255, 0], [0, 128, 255]],
                    ),
                )

        for i in range(len(batch)):
            det_id = batch.ids[i]
            if det_id in tag_ids:
                gt_idx = tag_ids.index(det_id)
                det_corners = batch.corners[i]
                gt_corners_arr = np.array(corners_list[gt_idx])

                min_err = float("inf")
                for rot in range(4):
                    rotated = np.roll(det_corners, rot, axis=0)
                    err = np.sqrt(np.mean(np.sum((rotated - gt_corners_arr) ** 2, axis=1)))
                    min_err = min(min_err, err)

                rr.log(f"metrics/tag_{det_id}/corner_rmse", rr.Scalars([min_err]))

                if (
                    positions is not None
                    and positions[gt_idx] is not None
                    and batch.poses[i] is not None
                ):
                    gt_pos = np.array(positions[gt_idx])
                    det_pos = np.array(batch.poses[i][:3])
                    t_err = np.linalg.norm(det_pos - gt_pos)
                    rr.log(f"metrics/tag_{det_id}/pose_translation_err", rr.Scalars([float(t_err)]))

        count += 1

    print(f"Finished visualizing {count} images.")


def log_charuco_frame(result: dict) -> None:
    """Log accepted and rejected ChAruco saddle points to Rerun.

    Green crosshairs show successfully refined saddles; red markers show
    rejected predictions with the structure tensor determinant as a label
    (low determinant = blurry or flat region at the predicted location).
    """
    saddle_pts = result.get("saddle_pts")
    if saddle_pts is not None and len(saddle_pts) > 0:
        n = len(saddle_pts)
        rr.log(
            "world/camera/charuco/saddles_accepted",
            rr.Points2D(
                saddle_pts,
                colors=np.tile([0, 220, 0], (n, 1)),
                radii=3.0,
                labels=["accepted"] * n,
            ),
        )
    else:
        rr.log("world/camera/charuco/saddles_accepted", rr.Clear(recursive=False))

    telemetry = result.get("telemetry")
    if telemetry is not None:
        rej_pts = telemetry.get("rejected_saddles")
        rej_dets = telemetry.get("rejected_determinants")
        if rej_pts is not None and len(rej_pts) > 0:
            r = len(rej_pts)
            labels = [f"det={d:.2e}" for d in rej_dets]
            rr.log(
                "world/camera/charuco/saddles_rejected",
                rr.Points2D(
                    rej_pts,
                    colors=np.tile([220, 0, 0], (r, 1)),
                    radii=3.0,
                    labels=labels,
                ),
            )
        else:
            rr.log("world/camera/charuco/saddles_rejected", rr.Clear(recursive=False))


if __name__ == "__main__":
    main()
