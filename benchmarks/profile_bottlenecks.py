import locus
import numpy as np
from benchmark_suite import BenchmarkSuite


def profile_locus():
    suite = BenchmarkSuite()

    scenarios = [
        ("Clean (1 target)", {"targets": 1, "noise": 0.0}),
        ("Clean (20 targets)", {"targets": 20, "noise": 0.0}),
        ("Noisy (5 targets, sigma=10)", {"targets": 5, "noise": 10.0}),
    ]

    print(f"{'Scenario':<30} | {'Thresh':<8} | {'Seg':<8} | {'Quad':<8} | {'Dec':<8} | {'Total'}")
    print("-" * 80)

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
        avg_total = np.mean([s.total_ms for s in stats_list])

        print(
            f"{name:<30} | {avg_thresh:<8.2f} | {avg_seg:<8.2f} | {avg_quad:<8.2f} | {avg_dec:<8.2f} | {avg_total:.2f} ms"
        )


if __name__ == "__main__":
    profile_locus()
