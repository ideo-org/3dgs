#!/usr/bin/env python3
"""
lidar_converter.py — Convert iOS LiDAR capture bundle to COLMAP text format.

Converts a capture bundle (from Portal iOS app) into a scene directory
that graphdeco-inria's train.py can load via readColmapSceneInfo().

Usage:
  python lidar_converter.py --input /path/to/capture_bundle --output /path/to/scene
  python lidar_converter.py --input /path/to/bundle --output /path/to/scene --skip-cloud
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_meta(frames_dir: Path, frame_id: int) -> dict:
    """Load per-frame meta.json."""
    path = frames_dir / f"{frame_id:06d}_meta.json"
    with open(path) as f:
        return json.load(f)


def load_depth(frames_dir: Path, frame_id: int, width: int, height: int) -> np.ndarray:
    """Load depth map as float32 array (height, width)."""
    path = frames_dir / f"{frame_id:06d}_depth.bin"
    data = np.frombuffer(path.read_bytes(), dtype="<f4")
    return data.reshape(height, width)


def load_confidence(
    frames_dir: Path, frame_id: int, width: int, height: int
) -> np.ndarray:
    """Load confidence map as uint8 array (height, width)."""
    path = frames_dir / f"{frame_id:06d}_conf.bin"
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return data.reshape(height, width)


def load_rgb(frames_dir: Path, frame_id: int) -> np.ndarray:
    """Load RGB image as uint8 array (height, width, 3)."""
    path = frames_dir / f"{frame_id:06d}_rgb.jpg"
    img = Image.open(path).convert("RGB")
    return np.array(img)


def enumerate_frame_ids(frames_dir: Path) -> list:
    """Return sorted list of frame IDs from frames directory."""
    ids = []
    for p in frames_dir.iterdir():
        if p.name.endswith("_meta.json"):
            try:
                ids.append(int(p.name.split("_")[0]))
            except ValueError:
                pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def arkit_c2w_to_colmap(c2w_arkit: np.ndarray):
    """
    Convert ARKit camera-to-world to COLMAP world-to-camera.

    ARKit camera: X=right, Y=up, Z=toward-user (out of screen)
    COLMAP camera: X=right, Y=down, Z=into-scene

    The flip matrix rotates 180° around X-axis in camera space.
    """
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_colmap = c2w_arkit @ flip
    w2c = np.linalg.inv(c2w_colmap)
    return w2c, c2w_colmap


def rotation_to_quat_colmap(R: np.ndarray):
    """
    Convert 3x3 rotation matrix to COLMAP quaternion (qw, qx, qy, qz).

    scipy returns [qx, qy, qz, qw] (scalar-LAST).
    COLMAP expects (qw, qx, qy, qz) (scalar-FIRST).
    """
    quat_xyzw = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    return qw, qx, qy, qz


# ---------------------------------------------------------------------------
# COLMAP text file writers
# ---------------------------------------------------------------------------


def write_cameras_txt(
    path: Path, fx: float, fy: float, cx: float, cy: float, width: int, height: int
):
    """Write cameras.txt with single PINHOLE camera."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(path: Path, frames_data: list):
    """
    Write images.txt with TWO lines per image.

    Format:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      [empty POINTS2D line]

    CRITICAL: IMAGE_ID starts at 1. Second line MUST exist (even if empty).
    """
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            f"# Number of images: {len(frames_data)}, mean observations per image: 0\n"
        )
        for frame in frames_data:
            f.write(
                f"{frame['image_id']} "
                f"{frame['qw']:.9f} {frame['qx']:.9f} {frame['qy']:.9f} {frame['qz']:.9f} "
                f"{frame['tx']:.9f} {frame['ty']:.9f} {frame['tz']:.9f} "
                f"1 {frame['name']}\n"
            )
            f.write("\n")  # Empty POINTS2D line — REQUIRED by graphdeco parser


def write_points3d_txt(path: Path, points: np.ndarray, colors: np.ndarray):
    """
    Write points3D.txt.

    Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    POINT3D_ID starts at 1. ERROR = 0.0. TRACK is empty.
    """
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for i, (pt, col) in enumerate(zip(points, colors)):
            r, g, b = int(col[0]), int(col[1]), int(col[2])
            f.write(f"{i + 1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b} 0.0\n")


# ---------------------------------------------------------------------------
# Point cloud building
# ---------------------------------------------------------------------------


def build_point_cloud_from_depth(
    frames_dir: Path, frames_data: list, sample_every: int = 5
) -> tuple:
    """
    Build world-space point cloud from per-frame depth maps.

    Uses COLMAP camera convention for unprojection (Y-down, Z-forward).
    Returns (points, colors) as numpy arrays.
    """
    all_points = []
    all_colors = []

    sampled = frames_data[::sample_every]
    print(
        f"  Building point cloud from {len(sampled)} frames (every {sample_every}th)..."
    )

    for frame in sampled:
        frame_id = frame["frame_id"]
        meta = frame["meta"]
        depth_w = meta["depth_width"]
        depth_h = meta["depth_height"]
        image_w = meta["image_width"]
        image_h = meta["image_height"]
        fx = meta["intrinsics"]["fx"]
        fy = meta["intrinsics"]["fy"]
        cx = meta["intrinsics"]["cx"]
        cy = meta["intrinsics"]["cy"]

        # Scale intrinsics to depth resolution
        scale_x = depth_w / image_w
        scale_y = depth_h / image_h
        fx_d = fx * scale_x
        fy_d = fy * scale_y
        cx_d = cx * scale_x
        cy_d = cy * scale_y

        try:
            depth = load_depth(frames_dir, frame_id, depth_w, depth_h)
            conf = load_confidence(frames_dir, frame_id, depth_w, depth_h)
        except FileNotFoundError:
            continue

        # Filter valid pixels
        valid = (conf >= 1) & (depth > 0.2) & (depth < 5.0) & np.isfinite(depth)
        if not valid.any():
            continue

        # Pixel grid
        u, v = np.meshgrid(
            np.arange(depth_w, dtype=np.float32), np.arange(depth_h, dtype=np.float32)
        )

        # Unproject in COLMAP camera convention (Y-down, Z-forward)
        X_cam = (u - cx_d) * depth / fx_d
        Y_cam = (v - cy_d) * depth / fy_d  # Y-down (v increases downward)
        Z_cam = depth  # Z-forward (depth along Z)

        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)[valid]  # (N, 3)

        # Transform to world using c2w_colmap
        c2w_colmap = frame["c2w_colmap"]
        R_c2w = c2w_colmap[:3, :3]
        t_c2w = c2w_colmap[:3, 3]
        pts_world = (R_c2w @ pts_cam.T).T + t_c2w  # (N, 3)

        # Sample RGB color from image
        try:
            rgb = load_rgb(frames_dir, frame_id)
        except Exception:
            colors_frame = np.full((len(pts_world), 3), 128, dtype=np.uint8)
            all_points.append(pts_world)
            all_colors.append(colors_frame)
            continue

        u_rgb = (u[valid] / scale_x).astype(int).clip(0, image_w - 1)
        v_rgb = (v[valid] / scale_y).astype(int).clip(0, image_h - 1)
        colors_frame = rgb[v_rgb, u_rgb]  # (N, 3) uint8

        all_points.append(pts_world)
        all_colors.append(colors_frame)

    if not all_points:
        print("  WARNING: No valid depth points found, using empty cloud")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    print(f"  Raw point count: {len(points)}")

    # Downsample with open3d if available
    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        cloud = cloud.voxel_down_sample(voxel_size=0.01)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = (np.asarray(cloud.colors) * 255).astype(np.uint8)
        print(f"  After downsampling: {len(points)} points")
    except ImportError:
        print("  open3d not available, skipping downsampling")

    return points, colors


def load_global_cloud(ply_path: Path) -> tuple:
    """
    Load global_cloud.ply and return (points, colors).

    The PLY is in ARKit world space. Since the flip only affects camera
    convention (not world coordinates), points can be used directly.
    """
    try:
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(cloud.points, dtype=np.float32)
        if cloud.has_colors():
            colors = (np.asarray(cloud.colors) * 255).astype(np.uint8)
        else:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
        print(f"  Loaded global_cloud.ply: {len(points)} points")
        return points, colors
    except ImportError:
        pass

    # Fallback: parse ASCII PLY manually
    with open(ply_path) as f:
        lines = f.readlines()

    # Find end_header
    header_end = 0
    n_vertices = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i + 1
            break

    points = []
    colors = []
    for line in lines[header_end : header_end + n_vertices]:
        parts = line.strip().split()
        if len(parts) >= 6:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

    print(f"  Loaded global_cloud.ply (ASCII): {len(points)} points")
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert iOS LiDAR capture bundle to COLMAP text format"
    )
    parser.add_argument(
        "--input", required=True, help="Path to capture bundle directory"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output scene directory"
    )
    parser.add_argument(
        "--skip-cloud",
        action="store_true",
        help="Use global_cloud.ply if present, skip depth processing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    frames_dir = input_path / "frames"

    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not frames_dir.exists():
        print(f"ERROR: frames/ directory not found in {input_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load session.json
    session_path = input_path / "session.json"
    if session_path.exists():
        with open(session_path) as f:
            session = json.load(f)
        print(
            f"Session: {session.get('num_frames', '?')} frames, "
            f"device: {session.get('device_model', '?')}"
        )
    else:
        print("WARNING: session.json not found, continuing without it")

    # 2. Enumerate frames
    frame_ids = enumerate_frame_ids(frames_dir)
    if not frame_ids:
        print("ERROR: No frames found in frames/ directory", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(frame_ids)} frames")

    # 3. Load all meta.json files
    print("Loading frame metadata...")
    frames_data = []
    for frame_id in frame_ids:
        try:
            meta = load_meta(frames_dir, frame_id)
        except FileNotFoundError:
            print(f"  WARNING: Missing meta for frame {frame_id}, skipping")
            continue

        c2w_arkit = np.array(meta["camera_to_world"], dtype=np.float64)
        w2c, c2w_colmap = arkit_c2w_to_colmap(c2w_arkit)
        qw, qx, qy, qz = rotation_to_quat_colmap(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]

        frames_data.append(
            {
                "frame_id": frame_id,
                "meta": meta,
                "c2w_colmap": c2w_colmap,
                "w2c": w2c,
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "image_id": len(frames_data) + 1,  # 1-indexed
                "name": f"{frame_id:06d}_rgb.jpg",
            }
        )

    print(f"Loaded {len(frames_data)} valid frames")

    # 4. Compute median intrinsics
    fx_vals = [f["meta"]["intrinsics"]["fx"] for f in frames_data]
    fy_vals = [f["meta"]["intrinsics"]["fy"] for f in frames_data]
    cx_vals = [f["meta"]["intrinsics"]["cx"] for f in frames_data]
    cy_vals = [f["meta"]["intrinsics"]["cy"] for f in frames_data]
    fx = float(np.median(fx_vals))
    fy = float(np.median(fy_vals))
    cx = float(np.median(cx_vals))
    cy = float(np.median(cy_vals))
    image_w = frames_data[0]["meta"]["image_width"]
    image_h = frames_data[0]["meta"]["image_height"]
    print(f"Median intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # 5. Create output directory structure
    images_dir = output_path / "images"
    sparse_dir = output_path / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 6. Copy RGB images to images/
    print("Copying images...")
    for frame in frames_data:
        src = frames_dir / f"{frame['frame_id']:06d}_rgb.jpg"
        dst = images_dir / frame["name"]
        if src.exists():
            shutil.copy2(src, dst)

    # 7. Write cameras.txt
    write_cameras_txt(sparse_dir / "cameras.txt", fx, fy, cx, cy, image_w, image_h)
    print(f"Written cameras.txt (PINHOLE {image_w}x{image_h})")

    # 8. Write images.txt
    write_images_txt(sparse_dir / "images.txt", frames_data)
    print(f"Written images.txt ({len(frames_data)} images, 2 lines each)")

    # 9. Build/load point cloud
    global_cloud_path = input_path / "global_cloud.ply"
    if global_cloud_path.exists() and args.skip_cloud:
        print("Loading global_cloud.ply...")
        points, colors = load_global_cloud(global_cloud_path)
    elif global_cloud_path.exists():
        print("Loading global_cloud.ply (use --skip-cloud to skip depth processing)...")
        points, colors = load_global_cloud(global_cloud_path)
    else:
        print("Building point cloud from depth maps...")
        points, colors = build_point_cloud_from_depth(frames_dir, frames_data)

    # 10. Write points3D.txt
    write_points3d_txt(sparse_dir / "points3D.txt", points, colors)
    print(f"Written points3D.txt ({len(points)} points)")

    # 11. Summary
    print(f"\n=== Conversion complete ===")
    print(f"Scene directory: {output_path}")
    print(f"  Images: {len(frames_data)}")
    print(f"  Points: {len(points)}")
    print(f"  cameras.txt: {sparse_dir / 'cameras.txt'}")
    print(f"  images.txt: {sparse_dir / 'images.txt'}")
    print(f"  points3D.txt: {sparse_dir / 'points3D.txt'}")


if __name__ == "__main__":
    main()
