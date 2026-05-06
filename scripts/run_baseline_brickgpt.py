import argparse
from pathlib import Path

from baseline_common import build_case_dir, copy_first_existing, init_summary, run_command, slugify, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BrickGPT and collect native brick outputs.")
    parser.add_argument("--repo_root", required=True, help="Path to the BrickGPT repo root.")
    parser.add_argument("--prompt", required=True, help="Text prompt to send to BrickGPT.")
    parser.add_argument("--case_id", default=None, help="Optional case id. Defaults to a slugified prompt.")
    parser.add_argument(
        "--output_root",
        default=r"comparison_output/baselines",
        help="Root directory for standardized outputs.",
    )
    parser.add_argument("--uv_exe", default="uv", help="Executable used to run BrickGPT.")
    parser.add_argument("--seed", default="42", help="Seed passed through the interactive prompt.")
    parser.add_argument("--render_name", default="output.png", help="Image filename entered into BrickGPT.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    case_id = args.case_id or slugify(args.prompt)[:80]
    case_dir = build_case_dir(Path(args.output_root), "brickgpt", case_id)
    raw_dir = case_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary = init_summary("brickgpt", case_id, "text", args.prompt, repo_root, case_dir)
    summary["command"] = [args.uv_exe, "run", "infer"]

    stdin_text = f"{args.prompt}\n{args.render_name}\n{args.seed}\n\n"
    return_code = run_command(
        command=summary["command"],
        cwd=repo_root,
        stdout_path=case_dir / "stdout.log",
        stderr_path=case_dir / "stderr.log",
        stdin_text=stdin_text,
    )
    summary["return_code"] = return_code

    ldr_path = copy_first_existing(
        [repo_root / "output.ldr", repo_root / "result.ldr", repo_root / "outputs" / "output.ldr"],
        raw_dir / "output.ldr",
    )
    txt_path = copy_first_existing(
        [repo_root / "output.txt", repo_root / "result.txt", repo_root / "outputs" / "output.txt"],
        raw_dir / "output.txt",
    )
    png_candidates = [
        repo_root / args.render_name,
        repo_root / "output.png",
        repo_root / "outputs" / args.render_name,
        repo_root / "outputs" / "output.png",
    ]
    png_path = copy_first_existing(png_candidates, raw_dir / "output.png")

    summary["native_outputs"] = {
        "ldr_path": str(ldr_path) if ldr_path else None,
        "text_path": str(txt_path) if txt_path else None,
        "render_path": str(png_path) if png_path else None,
    }
    summary["success"] = return_code == 0 and ldr_path is not None
    write_json(case_dir / "summary.json", summary)

    # Keep the repo root clean after collection if the outputs were generated there.
    for candidate in [repo_root / "output.ldr", repo_root / "output.txt", repo_root / "output.png"]:
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass

    print(f"Saved BrickGPT case to: {case_dir}")


if __name__ == "__main__":
    main()
