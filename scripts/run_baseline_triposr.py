import argparse
from pathlib import Path

from baseline_common import brickize_mesh, build_case_dir, init_summary, run_command, slugify, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TripoSR and optionally brickize the output mesh.")
    parser.add_argument("--repo_root", required=True, help="Path to the TripoSR repo root.")
    parser.add_argument("--input_image", required=True, help="Input image path.")
    parser.add_argument("--case_id", default=None, help="Optional case id. Defaults to the input image stem.")
    parser.add_argument(
        "--output_root",
        default=r"comparison_output/baselines",
        help="Root directory for standardized outputs.",
    )
    parser.add_argument("--python_exe", default="python", help="Python executable used to launch the repo.")
    parser.add_argument("--resolution", type=int, default=16, help="Voxel resolution for optional brickization.")
    parser.add_argument("--skip_brickize", action="store_true", help="Skip coarse voxelization and brick mapping.")
    parser.add_argument("--model_save_format", default="glb", choices=["glb", "obj"], help="Saved mesh format.")
    parser.add_argument("--device", default=None, help="Optional device flag passed to TripoSR.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_image = Path(args.input_image).resolve()
    case_id = args.case_id or slugify(input_image.stem)
    case_dir = build_case_dir(Path(args.output_root), "triposr", case_id)
    raw_dir = case_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_exe,
        "run.py",
        str(input_image),
        "--output-dir",
        str(raw_dir),
        "--model-save-format",
        args.model_save_format,
    ]
    if args.device:
        command.extend(["--device", args.device])

    summary = init_summary("triposr", case_id, "image", str(input_image), repo_root, case_dir)
    summary["command"] = command
    return_code = run_command(
        command=command,
        cwd=repo_root,
        stdout_path=case_dir / "stdout.log",
        stderr_path=case_dir / "stderr.log",
    )
    summary["return_code"] = return_code

    mesh_suffix = f"mesh.{args.model_save_format}"
    mesh_candidates = [
        raw_dir / "0" / mesh_suffix,
        raw_dir / mesh_suffix,
        raw_dir / input_image.stem / mesh_suffix,
    ]
    mesh_path = next((path for path in mesh_candidates if path.exists()), None)
    summary["mesh_path"] = str(mesh_path) if mesh_path else None

    if mesh_path is not None and not args.skip_brickize:
        summary["brickized"] = brickize_mesh(mesh_path=mesh_path, output_dir=case_dir, resolution=args.resolution)

    summary["success"] = return_code == 0 and mesh_path is not None
    write_json(case_dir / "summary.json", summary)
    print(f"Saved TripoSR case to: {case_dir}")


if __name__ == "__main__":
    main()
