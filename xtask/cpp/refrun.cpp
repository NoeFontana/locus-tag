// Reference-detector runner for `cargo xtask sota` (built by `cargo xtask sota setup`).
//
//   refrun <nano|opencv|opencv-subpix> <dict> <threads> <list.txt> <out.jsonl> [reps] [border_bits]
//
// Runs one *published, unpatched* reference detector over every image in
// `list.txt` and writes one JSON object per image:
//   {"image": ..., "ms": <best-of-reps detect() ms>, "ids": [...], "corners": [[[x,y]x4], ...]}
// Corners are in OpenCV's pixel-centre-at-integer convention.
//
// Protocol (mirrors aruco_nano's testperf.cpp so published numbers are comparable):
//   * image decode is outside the timer; one untimed warm-up call on the first image;
//   * OpenCV runs with errorCorrectionRate = 0 (as testperf.cpp) and CORNER_REFINE_NONE
//     unless `opencv-subpix` is requested;
//   * `border_bits` is passed through public parameters only. aruco_nano ignores
//     values != 1 (its bit grid is hard-coded to markerSize + 2), so datasets with
//     2-bit borders (Kalibr AprilGrid) are reported as unsupported for it rather than
//     patched — see xtask/README.md.
#include "aruco_nano.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace {
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: refrun <nano|opencv|opencv-subpix> <dict> <threads> <list> <out> [reps] [border_bits]\n";
        return 2;
    }
    const std::string mode = argv[1], dname = argv[2];
    const int threads = std::stoi(argv[3]);
    const int reps = argc > 6 ? std::stoi(argv[6]) : 2;
    const int border = argc > 7 ? std::stoi(argv[7]) : 1;
    const std::map<std::string, int> dicts = {
        {"ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
        {"APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
        {"APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
    };
    if (!dicts.count(dname) || (mode != "nano" && mode != "opencv" && mode != "opencv-subpix")) {
        std::cerr << "unknown dictionary or mode\n";
        return 2;
    }
    cv::setNumThreads(threads);
    const auto dict = cv::aruco::getPredefinedDictionary(dicts.at(dname));

    cv::aruco::DetectorParameters cv_params;
    cv_params.errorCorrectionRate = 0;
    cv_params.markerBorderBits = border;
    if (mode == "opencv-subpix") cv_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    const cv::aruco::ArucoDetector cv_det(dict, cv_params);

    aruco_nano::DetectorParameters nano_params;
    nano_params.dicts = {dict};
    nano_params.markerBorderBits = static_cast<float>(border);
    const aruco_nano::ArucoDetector nano_det(std::vector<cv::aruco::Dictionary>{dict}, nano_params);

    std::ifstream list(argv[4]);
    std::ofstream out(argv[5]);
    std::string path;
    bool warmed = false;
    while (std::getline(list, path)) {
        if (path.empty()) continue;
        const cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "cannot read " << path << "\n";
            return 1;
        }
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        auto run = [&]() {
            ids.clear();
            corners.clear();
            if (mode == "nano") nano_det.detectMarkers(img, corners, ids);
            else cv_det.detectMarkers(img, corners, ids);
        };
        if (!warmed) {
            run();
            warmed = true;
        }
        double best = 1e18;
        for (int r = 0; r < std::max(1, reps); ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            run();
            const auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        out << "{\"image\":\"" << json_escape(path) << "\",\"ms\":" << best << ",\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) out << (i ? "," : "") << ids[i];
        out << "],\"corners\":[";
        for (size_t i = 0; i < corners.size(); ++i) {
            out << (i ? "," : "") << "[";
            for (int c = 0; c < 4; ++c)
                out << (c ? "," : "") << "[" << corners[i][c].x << "," << corners[i][c].y << "]";
            out << "]";
        }
        out << "]}\n";
    }
    return 0;
}
