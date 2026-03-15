#!/usr/bin/env python3
"""
Synthetic LiDAR capture bundle generator for pipeline testing.

Generates a mock capture bundle matching the brief.md format:
  <output>/
    frames/
      000001_rgb.jpg
      000001_depth.bin   # float32 LE, 256x192
      000001_conf.bin    # uint8, 256x192
      000001_meta.json
      ...
    global_cloud.ply
    session.json
    ground_truth.json

Usage:
  python generate_synthetic_bundle.py --output /tmp/test_bundle --num-frames 50
  python generate_synthetic_bundle.py --output /tmp/test_bundle --validate
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
ROOM_MIN = np.array([-1.5, 0.0, -1.5])
ROOM_MAX = np.array([1.5, 2.5, 1.5])
ROOM_CENTER = np.array([0.0, 1.25, 0.0])

DEPTH_W, DEPTH_H = 256, 192
IMAGE_W, IMAGE_H = 1920, 1440
FX = FY = 1597.78
CX, CY = 960.0, 720.0

# Depth intrinsics scaled to depth resolution
DEPTH_FX = FX * DEPTH_W / IMAGE_W
DEPTH_FY = FY * DEPTH_H / IMAGE_H
DEPTH_CX = CX * DEPTH_W / IMAGE_W
DEPTH_CY = CY * DEPTH_H / IMAGE_H


# ---------------------------------------------------------------------------
# Camera math
# ---------------------------------------------------------------------------


def make_c2w(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build ARKit camera-to-world matrix.

    ARKit convention:
      X = right, Y = up, Z = toward user (out of screen / backward from scene)
    """
    up = np.array([0.0, 1.0, 0.0])
    forward = pos - target  # Z axis: camera Z points AWAY from scene in ARKit
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm

    right = np.cross(up, forward)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-9:
        # Degenerate case: camera looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / r_norm

    up_corrected = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_corrected
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


def camera_positions(n: int) -> list:
    """Generate n camera positions orbiting the room center."""
    positions = []
    for i in range(n):
        angle = i * 2.0 * math.pi / n
        x = math.cos(angle) * 1.0
        z = math.sin(angle) * 1.0
        y = 1.25
        positions.append(np.array([x, y, z]))
    return positions


# ---------------------------------------------------------------------------
# Depth map generation (ray-box intersection)
# ---------------------------------------------------------------------------


def ray_box_intersect(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> float:
    """Return distance to nearest box face intersection, or inf if no hit."""
    tmin = -np.inf
    tmax = np.inf
    for i in range(3):
        if abs(ray_dir[i]) < 1e-9:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return np.inf
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
    if tmax < tmin:
        return np.inf
    # We want the exit distance (tmax) since camera is inside the box
    # But if tmin > 0, camera is outside — use tmin
    if tmin > 0:
        return tmin
    if tmax > 0:
        return tmax
    return np.inf


def generate_depth_map(c2w: np.ndarray) -> np.ndarray:
    """Generate a float32 depth map (DEPTH_H x DEPTH_W) for a camera pose."""
    depth = np.zeros((DEPTH_H, DEPTH_W), dtype=np.float32)

    cam_pos = c2w[:3, 3]
    R = c2w[:3, :3]  # columns: right, up, forward(Z toward user)

    # Precompute pixel grid
    us = np.arange(DEPTH_W, dtype=np.float32)
    vs = np.arange(DEPTH_H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)  # shape (H, W)

    # Camera-space ray directions (COLMAP convention: Y-down, Z-forward into scene)
    # But we're in ARKit convention: Z is toward user (backward).
    # Ray direction in camera space: (u-cx)/fx, (v-cy)/fy, -1 (into scene = -Z in ARKit)
    rx = (uu - DEPTH_CX) / DEPTH_FX
    ry = (vv - DEPTH_CY) / DEPTH_FY
    rz = -np.ones_like(rx)  # -Z = into scene in ARKit

    # Normalize
    ray_len = np.sqrt(rx**2 + ry**2 + rz**2)
    rx /= ray_len
    ry /= ray_len
    rz /= ray_len

    # Transform to world space: ray_world = R @ [rx, ry, rz]
    # R columns: right(X), up(Y), forward(Z toward user)
    ray_world_x = R[0, 0] * rx + R[0, 1] * ry + R[0, 2] * rz
    ray_world_y = R[1, 0] * rx + R[1, 1] * ry + R[1, 2] * rz
    ray_world_z = R[2, 0] * rx + R[2, 1] * ry + R[2, 2] * rz

    # Vectorized ray-box intersection
    # For each pixel, find t such that cam_pos + t * ray_world hits the room box
    # We want the exit intersection (camera is inside the room)
    eps = 1e-9
    t_vals = np.full((DEPTH_H, DEPTH_W), np.inf, dtype=np.float32)

    for axis, (rw, cp, bmin, bmax) in enumerate(
        [
            (ray_world_x, cam_pos[0], ROOM_MIN[0], ROOM_MAX[0]),
            (ray_world_y, cam_pos[1], ROOM_MIN[1], ROOM_MAX[1]),
            (ray_world_z, cam_pos[2], ROOM_MIN[2], ROOM_MAX[2]),
        ]
    ):
        mask = np.abs(rw) > eps
        t_pos = np.where(mask, (bmax - cp) / np.where(mask, rw, 1.0), np.inf)
        t_neg = np.where(mask, (bmin - cp) / np.where(mask, rw, 1.0), np.inf)
        t_face = np.where(t_pos > 0, t_pos, np.inf)
        t_face = np.minimum(t_face, np.where(t_neg > 0, t_neg, np.inf))
        t_vals = np.minimum(t_vals, t_face)

    # Clamp to valid depth range
    t_vals = np.clip(t_vals, 0.2, 5.0)
    t_vals = np.where(np.isinf(t_vals), 2.0, t_vals)  # fallback for degenerate rays

    return t_vals.astype(np.float32)


# ---------------------------------------------------------------------------
# RGB image generation
# ---------------------------------------------------------------------------


def generate_checkerboard(
    width: int = IMAGE_W, height: int = IMAGE_H, square_size: int = 32
) -> np.ndarray:
    """Generate a checkerboard pattern as uint8 RGB array."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y, x] = [255, 255, 255]
            else:
                img[y, x] = [0, 0, 0]
    return img


def generate_checkerboard_fast(
    width: int = IMAGE_W, height: int = IMAGE_H, square_size: int = 32
) -> np.ndarray:
    """Fast vectorized checkerboard generation."""
    ys = np.arange(height) // square_size
    xs = np.arange(width) // square_size
    pattern = ((ys[:, None] + xs[None, :]) % 2).astype(np.uint8) * 255
    img = np.stack([pattern, pattern, pattern], axis=2)
    return img


# ---------------------------------------------------------------------------
# PLY generation
# ---------------------------------------------------------------------------


def generate_global_cloud(n_points: int = 1000) -> list:
    """Generate points on room surfaces in ARKit world coordinates."""
    rng = np.random.default_rng(42)
    points = []

    # Distribute points across 6 faces
    per_face = n_points // 6

    # Floor (y=0): x in [-1.5,1.5], z in [-1.5,1.5]
    xs = rng.uniform(-1.5, 1.5, per_face)
    zs = rng.uniform(-1.5, 1.5, per_face)
    for x, z in zip(xs, zs):
        points.append((x, 0.0, z, 128, 128, 64))  # brownish floor

    # Ceiling (y=2.5)
    xs = rng.uniform(-1.5, 1.5, per_face)
    zs = rng.uniform(-1.5, 1.5, per_face)
    for x, z in zip(xs, zs):
        points.append((x, 2.5, z, 240, 240, 240))  # white ceiling

    # Wall x=-1.5
    ys = rng.uniform(0.0, 2.5, per_face)
    zs = rng.uniform(-1.5, 1.5, per_face)
    for y, z in zip(ys, zs):
        points.append((-1.5, y, z, 200, 100, 100))  # red wall

    # Wall x=+1.5
    ys = rng.uniform(0.0, 2.5, per_face)
    zs = rng.uniform(-1.5, 1.5, per_face)
    for y, z in zip(ys, zs):
        points.append((1.5, y, z, 100, 200, 100))  # green wall

    # Wall z=-1.5
    xs = rng.uniform(-1.5, 1.5, per_face)
    ys = rng.uniform(0.0, 2.5, per_face)
    for x, y in zip(xs, ys):
        points.append((x, y, -1.5, 100, 100, 200))  # blue wall

    # Wall z=+1.5
    xs = rng.uniform(-1.5, 1.5, per_face)
    ys = rng.uniform(0.0, 2.5, per_face)
    for x, y in zip(xs, ys):
        points.append((x, y, 1.5, 200, 200, 100))  # yellow wall

    return points


def write_ply(path: str, points: list):
    """Write ASCII PLY file with x,y,z,red,green,blue properties."""
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ---------------------------------------------------------------------------
# Bundle generation
# ---------------------------------------------------------------------------


def generate_bundle(output_dir: str, num_frames: int = 50):
    """Generate a synthetic capture bundle directory."""
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    positions = camera_positions(num_frames)
    checkerboard = generate_checkerboard_fast()
    rgb_img = Image.fromarray(checkerboard)

    camera_positions_list = []

    print(f"Generating {num_frames} frames...")
    for i, pos in enumerate(positions):
        frame_idx = i + 1
        prefix = f"{frame_idx:06d}"
        timestamp = i * (1.0 / 3.0)  # 3 fps

        c2w = make_c2w(pos, ROOM_CENTER)
        camera_positions_list.append(pos.tolist())

        # RGB JPEG
        rgb_path = os.path.join(frames_dir, f"{prefix}_rgb.jpg")
        rgb_img.save(rgb_path, "JPEG", quality=95)

        # Depth map
        depth = generate_depth_map(c2w)
        depth_path = os.path.join(frames_dir, f"{prefix}_depth.bin")
        with open(depth_path, "wb") as f:
            f.write(depth.astype("<f4").tobytes())

        # Confidence map (all high = 2)
        conf = np.full((DEPTH_H, DEPTH_W), 2, dtype=np.uint8)
        conf_path = os.path.join(frames_dir, f"{prefix}_conf.bin")
        with open(conf_path, "wb") as f:
            f.write(conf.tobytes())

        # Meta JSON
        meta = {
            "frame_index": frame_idx,
            "timestamp": round(timestamp, 6),
            "camera_to_world": c2w.tolist(),
            "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
            "image_width": IMAGE_W,
            "image_height": IMAGE_H,
            "depth_width": DEPTH_W,
            "depth_height": DEPTH_H,
        }
        meta_path = os.path.join(frames_dir, f"{prefix}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_frames} frames done")

    # session.json
    session = {
        "device_model": "iPhone 15 Pro",
        "ios_version": "17.4",
        "arkit_version": "6.0",
        "num_frames": num_frames,
        "frame_sampling_rate": 3,
        "capture_duration_sec": round(num_frames / 3.0, 1),
        "coordinate_system": "arkit_y_up_meters",
        "depth_source": "smoothedSceneDepth",
        "exposure_locked": True,
        "white_balance_locked": True,
        "focus_locked": True,
    }
    with open(os.path.join(output_dir, "session.json"), "w") as f:
        json.dump(session, f, indent=2)

    # global_cloud.ply
    print("Generating global point cloud...")
    cloud_points = generate_global_cloud(1000)
    write_ply(os.path.join(output_dir, "global_cloud.ply"), cloud_points)

    # ground_truth.json
    ground_truth = {
        "camera_positions_arkit": camera_positions_list,
        "room_bounds": {
            "min": ROOM_MIN.tolist(),
            "max": ROOM_MAX.tolist(),
        },
        "expected_colmap_flip": "c2w_colmap = c2w_arkit @ diag(1,-1,-1,1)",
    }
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Bundle generated at: {output_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Point cloud: {len(cloud_points)} points")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_bundle(output_dir: str) -> bool:
    """Validate a generated bundle directory. Returns True if valid."""
    errors = []

    # session.json
    session_path = os.path.join(output_dir, "session.json")
    if not os.path.exists(session_path):
        errors.append("MISSING: session.json")
    else:
        with open(session_path) as f:
            session = json.load(f)
        required_fields = [
            "device_model",
            "num_frames",
            "coordinate_system",
            "depth_source",
            "exposure_locked",
        ]
        for field in required_fields:
            if field not in session:
                errors.append(f"session.json missing field: {field}")
        num_frames = session.get("num_frames", 0)

    # frames/
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.isdir(frames_dir):
        errors.append("MISSING: frames/ directory")
    else:
        for i in range(1, num_frames + 1):
            prefix = f"{i:06d}"

            rgb_path = os.path.join(frames_dir, f"{prefix}_rgb.jpg")
            if not os.path.exists(rgb_path):
                errors.append(f"MISSING: frames/{prefix}_rgb.jpg")

            depth_path = os.path.join(frames_dir, f"{prefix}_depth.bin")
            if not os.path.exists(depth_path):
                errors.append(f"MISSING: frames/{prefix}_depth.bin")
            else:
                size = os.path.getsize(depth_path)
                expected = DEPTH_W * DEPTH_H * 4
                if size != expected:
                    errors.append(
                        f"WRONG SIZE: {prefix}_depth.bin is {size} bytes, expected {expected}"
                    )

            conf_path = os.path.join(frames_dir, f"{prefix}_conf.bin")
            if not os.path.exists(conf_path):
                errors.append(f"MISSING: frames/{prefix}_conf.bin")
            else:
                size = os.path.getsize(conf_path)
                expected = DEPTH_W * DEPTH_H
                if size != expected:
                    errors.append(
                        f"WRONG SIZE: {prefix}_conf.bin is {size} bytes, expected {expected}"
                    )

            meta_path = os.path.join(frames_dir, f"{prefix}_meta.json")
            if not os.path.exists(meta_path):
                errors.append(f"MISSING: frames/{prefix}_meta.json")
            else:
                with open(meta_path) as f:
                    meta = json.load(f)
                for field in [
                    "camera_to_world",
                    "intrinsics",
                    "image_width",
                    "image_height",
                    "depth_width",
                    "depth_height",
                ]:
                    if field not in meta:
                        errors.append(f"{prefix}_meta.json missing field: {field}")
                if "camera_to_world" in meta:
                    c2w = np.array(meta["camera_to_world"])
                    if c2w.shape != (4, 4):
                        errors.append(f"{prefix}_meta.json: camera_to_world is not 4x4")
                    else:
                        det = np.linalg.det(c2w[:3, :3])
                        if abs(det - 1.0) > 0.01:
                            errors.append(
                                f"{prefix}_meta.json: rotation det={det:.4f}, expected ~1.0"
                            )
                if "intrinsics" in meta:
                    for k in ["fx", "fy", "cx", "cy"]:
                        if k not in meta["intrinsics"]:
                            errors.append(f"{prefix}_meta.json: intrinsics missing {k}")

    # global_cloud.ply
    ply_path = os.path.join(output_dir, "global_cloud.ply")
    if not os.path.exists(ply_path):
        errors.append("MISSING: global_cloud.ply")
    else:
        with open(ply_path) as f:
            header = f.readline().strip()
        if header != "ply":
            errors.append("global_cloud.ply: invalid PLY header")

    # ground_truth.json
    gt_path = os.path.join(output_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        errors.append("MISSING: ground_truth.json")
    else:
        with open(gt_path) as f:
            gt = json.load(f)
        if "camera_positions_arkit" not in gt:
            errors.append("ground_truth.json missing camera_positions_arkit")
        else:
            if len(gt["camera_positions_arkit"]) != num_frames:
                errors.append(
                    f"ground_truth.json: expected {num_frames} positions, got {len(gt['camera_positions_arkit'])}"
                )

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print(f"VALIDATION PASSED: {num_frames} frames, all checks OK")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic LiDAR capture bundle for pipeline testing"
    )
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames to generate (default: 50, max: 100)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing bundle at --output and exit",
    )
    args = parser.parse_args()

    if args.num_frames > 100:
        print("ERROR: --num-frames cannot exceed 100", file=sys.stderr)
        sys.exit(1)

    if args.validate:
        ok = validate_bundle(args.output)
        sys.exit(0 if ok else 1)
    else:
        generate_bundle(args.output, args.num_frames)
        print("\nRunning validation...")
        ok = validate_bundle(args.output)
        if not ok:
            print(
                "WARNING: Bundle generated but validation found issues", file=sys.stderr
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
