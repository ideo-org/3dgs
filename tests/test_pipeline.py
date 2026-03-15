#!/usr/bin/env python3
"""
End-to-end LiDAR pipeline test with synthetic data.

Validates the full path: synthetic bundle → converter → graphdeco scene load.
GPU-dependent training test is optional (skip with --no-gpu).

Usage:
  python tests/test_pipeline.py --no-gpu
  python tests/test_pipeline.py  # includes GPU training test
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR = os.path.dirname(SCRIPT_DIR)  # N:\2020\3dgs-lidar
GRAPHDECO_DIR = os.path.join(os.path.dirname(LIDAR_DIR), "3dgs")  # N:\2020\3dgs

BUNDLE_DIR = os.path.join("C:\\", "tmp", "e2e_bundle")
SCENE_DIR = os.path.join("C:\\", "tmp", "e2e_scene")
MODEL_DIR = os.path.join("C:\\", "tmp", "e2e_model")

NUM_FRAMES = 20
PASS_COUNT = 0
FAIL_COUNT = 0


def ok(msg: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"[PASS] {msg}")


def fail(msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"[FAIL] {msg}", file=sys.stderr)


def check(condition: bool, pass_msg: str, fail_msg: str):
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)
    return condition


# ---------------------------------------------------------------------------
# Stage 1: Generate synthetic bundle
# ---------------------------------------------------------------------------
def stage_generate():
    print("\n=== Stage 1: Generate synthetic bundle ===")

    if os.path.exists(BUNDLE_DIR):
        shutil.rmtree(BUNDLE_DIR)

    gen_script = os.path.join(SCRIPT_DIR, "generate_synthetic_bundle.py")
    result = subprocess.run(
        [
            sys.executable,
            gen_script,
            "--output",
            BUNDLE_DIR,
            "--num-frames",
            str(NUM_FRAMES),
        ],
        capture_output=True,
        text=True,
    )
    if not check(
        result.returncode == 0,
        f"Bundle generator exited 0",
        f"Bundle generator failed:\n{result.stderr}",
    ):
        return False

    # Verify session.json
    session_path = os.path.join(BUNDLE_DIR, "session.json")
    if not check(
        os.path.exists(session_path), "session.json exists", "session.json missing"
    ):
        return False

    with open(session_path) as f:
        session = json.load(f)
    check(
        "num_frames" in session,
        "session.json has num_frames",
        "session.json missing num_frames",
    )
    check(
        session.get("num_frames") == NUM_FRAMES,
        f"num_frames == {NUM_FRAMES}",
        f"num_frames mismatch: {session.get('num_frames')}",
    )

    # Verify frame files
    frames_dir = os.path.join(BUNDLE_DIR, "frames")
    frame_count = 0
    for i in range(1, NUM_FRAMES + 1):
        prefix = f"{i:06d}"
        rgb = os.path.join(frames_dir, f"{prefix}_rgb.jpg")
        depth = os.path.join(frames_dir, f"{prefix}_depth.bin")
        conf = os.path.join(frames_dir, f"{prefix}_conf.bin")
        meta = os.path.join(frames_dir, f"{prefix}_meta.json")
        if all(os.path.exists(p) for p in [rgb, depth, conf, meta]):
            frame_count += 1

    check(
        frame_count == NUM_FRAMES,
        f"Bundle generated: {NUM_FRAMES} frames",
        f"Only {frame_count}/{NUM_FRAMES} frame sets found",
    )

    # Verify depth binary size
    depth_path = os.path.join(frames_dir, "000001_depth.bin")
    if os.path.exists(depth_path):
        size = os.path.getsize(depth_path)
        check(
            size == 256 * 192 * 4,
            f"depth.bin size correct ({size} bytes)",
            f"depth.bin wrong size: {size} (expected {256 * 192 * 4})",
        )

    # Verify ground_truth.json
    gt_path = os.path.join(BUNDLE_DIR, "ground_truth.json")
    check(
        os.path.exists(gt_path), "ground_truth.json exists", "ground_truth.json missing"
    )

    return True


# ---------------------------------------------------------------------------
# Stage 2: Convert to COLMAP format
# ---------------------------------------------------------------------------
def stage_convert():
    print("\n=== Stage 2: Convert to COLMAP format ===")

    if os.path.exists(SCENE_DIR):
        shutil.rmtree(SCENE_DIR)

    converter = os.path.join(LIDAR_DIR, "lidar_converter.py")
    result = subprocess.run(
        [sys.executable, converter, "--input", BUNDLE_DIR, "--output", SCENE_DIR],
        capture_output=True,
        text=True,
    )
    if not check(
        result.returncode == 0,
        "Converter exited 0",
        f"Converter failed:\n{result.stderr}",
    ):
        print(result.stdout)
        return False

    # Verify directory structure
    images_dir = os.path.join(SCENE_DIR, "images")
    sparse_dir = os.path.join(SCENE_DIR, "sparse", "0")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    images_txt = os.path.join(sparse_dir, "images.txt")
    points_txt = os.path.join(sparse_dir, "points3D.txt")

    check(
        os.path.isdir(images_dir),
        "images/ directory exists",
        "images/ directory missing",
    )
    check(os.path.exists(cameras_txt), "cameras.txt exists", "cameras.txt missing")
    check(os.path.exists(images_txt), "images.txt exists", "images.txt missing")
    check(os.path.exists(points_txt), "points3D.txt exists", "points3D.txt missing")

    # Count images
    jpg_count = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    check(
        jpg_count == NUM_FRAMES,
        f"Converted: {NUM_FRAMES} images",
        f"Wrong image count: {jpg_count}",
    )

    # Verify images.txt format (2 lines per image: pose line + empty POINTS2D line)
    with open(images_txt) as f:
        raw_lines = [l.rstrip() for l in f.readlines()]
    # Count non-comment lines (includes empty POINTS2D lines)
    non_comment = [l for l in raw_lines if not l.startswith("#")]
    # Each image = 1 pose line + 1 empty line = 2 entries
    check(
        len(non_comment) == NUM_FRAMES * 2,
        f"images.txt has {NUM_FRAMES * 2} data lines",
        f"images.txt has {len(non_comment)} data lines (expected {NUM_FRAMES * 2})",
    )
    # For quaternion check, use only pose lines (non-empty non-comment lines)
    data_lines = [l for l in non_comment if l]

    # Verify quaternion normalization
    qnorm_errors = 0
    for line in data_lines:  # All pose lines (empty lines already filtered)
        parts = line.split()
        if len(parts) >= 5:
            try:
                qw, qx, qy, qz = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
                if abs(norm - 1.0) > 1e-4:
                    qnorm_errors += 1
            except (ValueError, IndexError):
                qnorm_errors += 1
    check(
        qnorm_errors == 0,
        "All quaternions normalized",
        f"{qnorm_errors} quaternions not normalized",
    )

    # Count points
    with open(points_txt) as f:
        pt_lines = [l for l in f.readlines() if l.strip() and not l.startswith("#")]
    point_count = len(pt_lines)
    check(
        point_count > 0,
        f"Converted: {NUM_FRAMES} images, {point_count} points",
        f"No points in points3D.txt",
    )

    return True


# ---------------------------------------------------------------------------
# Stage 3: Coordinate validation
# ---------------------------------------------------------------------------
def stage_coordinates():
    print("\n=== Stage 3: Coordinate validation ===")

    gt_path = os.path.join(BUNDLE_DIR, "ground_truth.json")
    images_txt = os.path.join(SCENE_DIR, "sparse", "0", "images.txt")

    if not os.path.exists(gt_path) or not os.path.exists(images_txt):
        fail("Missing ground_truth.json or images.txt for coordinate validation")
        return False

    with open(gt_path) as f:
        gt = json.load(f)
    gt_positions = gt.get("camera_positions_arkit", gt.get("camera_positions", []))

    # Parse images.txt: reconstruct camera world positions C = -R^T @ T
    parsed_positions = {}
    with open(images_txt) as f:
        lines = [l.rstrip() for l in f.readlines()]

    data_lines = [l for l in lines if not l.startswith("#")]  # includes empty POINTS2D lines
    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].split()
        if len(parts) < 9:
            continue
        try:
            img_id = int(parts[0])
            qw, qx, qy, qz = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9] if len(parts) > 9 else ""

            # Reconstruct rotation matrix from quaternion (COLMAP convention)
            R = np.array(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qw * qz),
                        2 * (qx * qz + qw * qy),
                    ],
                    [
                        2 * (qx * qy + qw * qz),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qw * qx),
                    ],
                    [
                        2 * (qx * qz - qw * qy),
                        2 * (qy * qz + qw * qx),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ]
            )
            T = np.array([tx, ty, tz])
            # Camera world position: C = -R^T @ T
            C = -R.T @ T
            parsed_positions[img_id] = C
        except (ValueError, IndexError):
            continue

    # Compare to ground truth
    matches = 0
    total = min(len(gt_positions), len(parsed_positions))
    for i, gt_pos in enumerate(gt_positions[:total]):
        img_id = i + 1
        if img_id not in parsed_positions:
            continue
        gt_arr = np.array(gt_pos)
        pred_arr = parsed_positions[img_id]
        dist = np.linalg.norm(gt_arr - pred_arr)
        if dist < 1e-3:  # 1mm tolerance
            matches += 1

    check(
        matches == total and total == NUM_FRAMES,
        f"Camera position validation: {matches}/{NUM_FRAMES} match",
        f"Camera position mismatch: {matches}/{total} match (expected {NUM_FRAMES})",
    )
    return matches == total


# ---------------------------------------------------------------------------
# Stage 4: graphdeco scene load
# ---------------------------------------------------------------------------
def stage_graphdeco(no_gpu: bool = False):
    print("\n=== Stage 4: graphdeco scene load ===")

    # Add graphdeco to path
    sys.path.insert(0, GRAPHDECO_DIR)

    try:
        from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
    except ImportError as e:
        err_str = str(e)
        if "torch" in err_str and no_gpu:
            print(f"[SKIP] graphdeco requires torch (not installed, --no-gpu mode): {e}")
            return True  # Not a failure in no-gpu mode
        fail(f"Cannot import graphdeco colmap_loader: {e}")
        return False

    images_txt = os.path.join(SCENE_DIR, "sparse", "0", "images.txt")
    cameras_txt = os.path.join(SCENE_DIR, "sparse", "0", "cameras.txt")

    try:
        extrinsics = read_extrinsics_text(images_txt)
        intrinsics = read_intrinsics_text(cameras_txt)
    except Exception as e:
        fail(f"graphdeco scene load failed: {e}")
        return False

    check(
        len(extrinsics) == NUM_FRAMES,
        f"graphdeco scene loads: {len(extrinsics)} images, {len(intrinsics)} camera(s)",
        f"graphdeco loaded {len(extrinsics)} images (expected {NUM_FRAMES})",
    )
    check(len(intrinsics) >= 1, "At least 1 camera loaded", f"No cameras loaded")
    return True


# ---------------------------------------------------------------------------
# Stage 5: GPU training (optional)
# ---------------------------------------------------------------------------
def stage_gpu_training():
    print("\n=== Stage 5: GPU training (100 iterations) ===")

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    train_py = os.path.join(GRAPHDECO_DIR, "train.py")
    result = subprocess.run(
        [
            sys.executable,
            train_py,
            "-s",
            SCENE_DIR,
            "-m",
            MODEL_DIR,
            "--iterations",
            "100",
            "--data_device",
            "cpu",
            "--save_iterations",
            "100",
            "--test_iterations",
            "-1",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=GRAPHDECO_DIR,
    )

    if not check(
        result.returncode == 0,
        "Training exited 0",
        f"Training failed:\n{result.stderr[-2000:]}",
    ):
        return False

    ply_path = os.path.join(
        MODEL_DIR, "point_cloud", "iteration_100", "point_cloud.ply"
    )
    check(
        os.path.exists(ply_path),
        f"Training: PLY output exists at {ply_path}",
        f"PLY not found at {ply_path}",
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="End-to-end LiDAR pipeline test")
    parser.add_argument(
        "--no-gpu", action="store_true", help="Skip GPU-dependent training test"
    )
    args = parser.parse_args()

    print(f"LiDAR Pipeline E2E Test")
    print(f"  Bundle dir: {BUNDLE_DIR}")
    print(f"  Scene dir:  {SCENE_DIR}")
    print(f"  Frames:     {NUM_FRAMES}")
    print(f"  GPU test:   {'SKIP' if args.no_gpu else 'YES'}")

    ok_gen = stage_generate()
    ok_conv = (
        stage_convert()
        if ok_gen
        else (fail("Skipping convert (generate failed)") or False)
    )
    ok_coord = (
        stage_coordinates()
        if ok_conv
        else (fail("Skipping coordinate validation") or False)
    )
    ok_gd = stage_graphdeco(no_gpu=args.no_gpu) if ok_conv else (fail("Skipping graphdeco load") or False)

    if not args.no_gpu:
        stage_gpu_training()

    print(f"\n{'=' * 50}")
    print(f"PASSED {PASS_COUNT}/{PASS_COUNT + FAIL_COUNT} tests")
    if FAIL_COUNT > 0:
        print(f"FAILED {FAIL_COUNT} tests")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
