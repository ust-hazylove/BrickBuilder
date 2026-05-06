import argparse
import os
import sys
import time
from pathlib import Path

import torch


def load_triposg(triposg_root: Path):
    sys.path.insert(0, str(triposg_root))
    sys.path.insert(0, str(triposg_root / "scripts"))

    from briarmbg import BriaRMBG
    from scripts.inference_triposg import run_triposg
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    device = "cuda"
    dtype = torch.float16
    rmbg_dir = triposg_root / "pretrained_weights" / "RMBG-1.4"
    triposg_weights_dir = triposg_root / "pretrained_weights" / "TripoSG"

    rmbg_net = BriaRMBG.from_pretrained(str(rmbg_dir)).to(device)
    rmbg_net.eval()

    pipe = TripoSGPipeline.from_pretrained(str(triposg_weights_dir)).to(device, dtype)
    return run_triposg, pipe, rmbg_net


def iter_cases(image_compare_root: Path):
    for case_dir in sorted(p for p in image_compare_root.iterdir() if p.is_dir()):
        image_path = case_dir / "input.png"
        if image_path.exists():
            yield case_dir, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-compare-root",
        type=Path,
        default=Path(r"qualitative_pack_100cases_20260417\image_compare"),
    )
    parser.add_argument("--triposg-root", type=Path, default=Path(r"third_party/TripoSG"))
    parser.add_argument("--output-name", type=str, default="triposg.glb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--faces", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    cases = list(iter_cases(args.image_compare_root))
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"Found {len(cases)} cases with input.png")
    run_triposg, pipe, rmbg_net = load_triposg(args.triposg_root)

    started = time.time()
    completed = 0
    skipped = 0
    failed = []

    for index, (case_dir, image_path) in enumerate(cases, start=1):
        output_path = case_dir / args.output_name
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(cases)}] skip existing: {output_path}")
            skipped += 1
            continue

        case_start = time.time()
        print(f"[{index}/{len(cases)}] running {case_dir.name}")
        try:
            mesh = run_triposg(
                pipe=pipe,
                image_input=str(image_path),
                rmbg_net=rmbg_net,
                seed=args.seed,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                faces=args.faces,
            )
            mesh.export(str(output_path))
            completed += 1
            elapsed = time.time() - case_start
            print(f"[{index}/{len(cases)}] saved {output_path} ({elapsed:.1f}s)")
        except Exception as exc:
            failed.append((case_dir.name, repr(exc)))
            print(f"[{index}/{len(cases)}] FAILED {case_dir.name}: {exc!r}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - started
    print(f"Done in {total_elapsed / 60:.1f} min; completed={completed}, skipped={skipped}, failed={len(failed)}")
    for case_name, error in failed:
        print(f"FAILED\t{case_name}\t{error}")


if __name__ == "__main__":
    main()
