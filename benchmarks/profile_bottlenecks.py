import locus
import numpy as np
from benchmark_suite import BenchmarkSuite


def profile_locus():
    suite = BenchmarkSuite()

    scenarios = [
        ("Single Tag (1 target)", {"targets": 1, "noise": 0.0}),
        ("Medium Density (20 targets)", {"targets": 20, "noise": 0.0}),
        ("High Density (200 targets)", {"targets": 200, "noise": 0.0}),
    ]

    print(
        f"{'Scenario':<30} | {'Thresh':<8} | {'Seg':<8} | {'Quad':<8} | {'Dec':<8} | {'Cands':<6} | {'Dets':<6} | {'Total'}"
    )
    print("-" * 110)

    for name, cfg in scenarios:
        img, gt = suite.generate_image(cfg["targets"], noise_sigma=cfg["noise"])

        # Warmup
        _, _ = locus.detect_tags_with_stats(img)

        stats_list = []
        for _ in range(50):
            _, stats = locus.detect_tags_with_stats(img)
            stats_list.append(stats)

        avg_thresh = np.mean([s.threshold_ms for s in stats_list])
        avg_seg = np.mean([s.segmentation_ms for s in stats_list])
        avg_quad = np.mean([s.quad_extraction_ms for s in stats_list])
        avg_dec = np.mean([s.decoding_ms for s in stats_list])
        avg_cands = np.mean([s.num_candidates for s in stats_list])
        avg_dets = np.mean([s.num_detections for s in stats_list])
        avg_total = np.mean([s.total_ms for s in stats_list])

        print(
            f"{name:<30} | {avg_thresh:<8.2f} | {avg_seg:<8.2f} | {avg_quad:<8.2f} | {avg_dec:<8.2f} | {avg_cands:<6.1f} | {avg_dets:<6.1f} | {avg_total:.2f} ms"
        )


if __name__ == "__main__":
    profile_locus()
