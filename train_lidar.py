#!/usr/bin/env python3
"""
LiDAR 3DGS two-pass training pipeline.

Orchestrates: convert → train 7K → export preview SPZ →
              continue to 30K → export final SPZ → webhook notifications.

Usage:
    python train_lidar.py --input /path/to/bundle --output /path/to/model
    python train_lidar.py --input /path/to/bundle --output /path/to/model --dry-run
"""

import subprocess
import sys
import os
import json
import time
import argparse
import shutil
from pathlib import Path
import urllib.request


def run_stage(cmd, env=None, stage_name=""):
    """Run a subprocess stage, print output, raise on failure."""
    print(f"\n[{stage_name}] Running: {' '.join(str(c) for c in cmd)}")
    start = time.time()
    result = subprocess.run(cmd, env=env, check=True)
    elapsed = time.time() - start
    print(f"[{stage_name}] Done in {elapsed:.1f}s")
    return elapsed


def post_callback(url, payload):
    """POST JSON payload to callback URL. Silently ignore errors."""
    if not url:
        return
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[callback] Posted to {url}")
    except Exception as e:
        print(f"[callback] Warning: failed to post to {url}: {e}")


def find_converter(args):
    """Find lidar_converter.py path."""
    if args.converter_path:
        p = Path(args.converter_path)
        if not p.exists():
            raise FileNotFoundError(f"Converter not found at {p}")
        return p
    # Look relative to this script
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "lidar_converter.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("lidar_converter.py not found. Use --converter-path.")


def find_train_script(args):
    """Find graphdeco train.py path."""
    if args.train_script:
        p = Path(args.train_script)
        if not p.exists():
            raise FileNotFoundError(f"Train script not found at {p}")
        return p
    # Look in common locations relative to this script
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir / "train.py",
        script_dir.parent / "train.py",
        script_dir.parent / "3dgs" / "train.py",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("train.py not found. Use --train-script.")


def export_spz(ply_path, spz_path):
    """Export PLY to SPZ using nianticlabs/spz package."""
    try:
        import spz as spz_mod

        spz_mod.compress(str(ply_path), str(spz_path))
        size_kb = spz_path.stat().st_size // 1024
        print(f"[spz] Exported {ply_path} -> {spz_path} ({size_kb}KB)")
        return True
    except ImportError:
        print("[spz] WARNING: spz package not installed. Copying PLY as fallback.")
        shutil.copy2(ply_path, spz_path.with_suffix(".ply"))
        return False
    except Exception as e:
        print(f"[spz] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LiDAR 3DGS two-pass training pipeline"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to capture bundle directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to model output directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Final training iterations",
    )
    parser.add_argument(
        "--preview-iterations",
        type=int,
        default=7000,
        help="Preview training iterations",
    )
    parser.add_argument(
        "--preview-callback",
        type=str,
        default=None,
        help="URL to POST when preview SPZ is ready",
    )
    parser.add_argument(
        "--final-callback",
        type=str,
        default=None,
        help="URL to POST when final SPZ is ready",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scene loads without training",
    )
    parser.add_argument(
        "--converter-path",
        type=str,
        default=None,
        help="Path to lidar_converter.py (auto-detected if not set)",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=None,
        help="Path to graphdeco train.py (auto-detected if not set)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_dir = output_path / "scene"
    model_dir = output_path / "model"

    # ── Stage 1: Convert bundle to COLMAP format ────────────────────
    converter = find_converter(args)
    run_stage(
        [
            sys.executable,
            str(converter),
            "--input",
            str(input_path),
            "--output",
            str(scene_dir),
        ],
        stage_name="CONVERT",
    )

    # ── Dry-run: validate scene structure, skip training ────────────
    if args.dry_run:
        required = [
            scene_dir / "images",
            scene_dir / "sparse" / "0" / "cameras.txt",
            scene_dir / "sparse" / "0" / "images.txt",
            scene_dir / "sparse" / "0" / "points3D.txt",
        ]
        for p in required:
            if not p.exists():
                print(f"ERROR: Missing {p}", file=sys.stderr)
                sys.exit(1)

        images_count = len(list((scene_dir / "images").glob("*.jpg")))
        if images_count == 0:
            images_count = len(list((scene_dir / "images").glob("*.png")))

        points_file = scene_dir / "sparse" / "0" / "points3D.txt"
        with open(points_file) as f:
            points_count = sum(
                1 for line in f if line.strip() and not line.startswith("#")
            )

        print(
            f"Scene loaded successfully: {images_count} images, {points_count} points"
        )
        print("Dry-run complete — scene structure is valid.")
        return

    # ── Stage 2: Train to preview iterations (7K) ──────────────────
    train_script = find_train_script(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    preview_iter = args.preview_iterations
    run_stage(
        [
            sys.executable,
            str(train_script),
            "-s",
            str(scene_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(preview_iter),
            "--data_device",
            "cpu",
            "--save_iterations",
            str(preview_iter),
            "--checkpoint_iterations",
            str(preview_iter),
            "--test_iterations",
            "-1",
        ],
        env=env,
        stage_name=f"TRAIN_{preview_iter}",
    )

    # ── Stage 3: Export preview SPZ ─────────────────────────────────
    preview_ply = (
        model_dir / "point_cloud" / f"iteration_{preview_iter}" / "point_cloud.ply"
    )
    preview_spz = output_path / "preview.spz"
    if preview_ply.exists():
        export_spz(preview_ply, preview_spz)
        post_callback(
            args.preview_callback,
            {
                "pass": "preview",
                "status": "done",
                "previewUrl": str(preview_spz),
                "iterations": preview_iter,
            },
        )
    else:
        print(f"[WARN] Preview PLY not found at {preview_ply}")

    # ── Stage 4: Continue training to final iterations (30K) ───────
    # Checkpoint saved by graphdeco: {model_path}/chkpnt{iteration}.pth
    chkpnt_path = model_dir / f"chkpnt{preview_iter}.pth"

    final_iter = args.iterations
    train_cmd = [
        sys.executable,
        str(train_script),
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(final_iter),
        "--data_device",
        "cpu",
        "--save_iterations",
        str(final_iter),
        "--test_iterations",
        "-1",
    ]
    if chkpnt_path.exists():
        train_cmd += ["--start_checkpoint", str(chkpnt_path)]
    else:
        print(f"[WARN] Checkpoint not found at {chkpnt_path}, training from scratch")

    run_stage(train_cmd, env=env, stage_name=f"TRAIN_{final_iter}")

    # ── Stage 5: Export final SPZ ──────────────────────────────────
    final_ply = (
        model_dir / "point_cloud" / f"iteration_{final_iter}" / "point_cloud.ply"
    )
    final_spz = output_path / "final.spz"
    if final_ply.exists():
        export_spz(final_ply, final_spz)
        post_callback(
            args.final_callback,
            {
                "pass": "final",
                "status": "done",
                "splatUrl": str(final_spz),
                "iterations": final_iter,
            },
        )
    else:
        print(f"[WARN] Final PLY not found at {final_ply}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"=== Training complete ===")
    print(f"Preview SPZ: {preview_spz}")
    print(f"Final SPZ:   {final_spz}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
