import argparse
from pathlib import Path
import math

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Composite transparent brick renders onto pure white with a soft ground shadow.")
    parser.add_argument("--input_root", required=True, help="Root directory to scan for rendered RGBA PNG files.")
    parser.add_argument("--input_suffix", default="_reference_rgba.png", help="Suffix of RGBA source renders.")
    parser.add_argument("--output_suffix", default="_white_render_hd.png", help="Suffix of final white-background renders.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--include_stems",
        default="",
        help="Comma-separated filename stems to process, e.g. instantmesh,triposr. Empty means process all.",
    )
    parser.add_argument(
        "--exclude_dirnames",
        default="",
        help="Comma-separated directory names to skip during recursive scan.",
    )
    return parser.parse_args()


def make_shadow(alpha: Image.Image, size):
    bbox = alpha.getbbox()
    if bbox is None:
        return Image.new("RGBA", size, (0, 0, 0, 0))

    model_w = max(1, bbox[2] - bbox[0])
    model_h = max(1, bbox[3] - bbox[1])
    shadow_w = max(20, int(model_w * 0.36))
    shadow_h = max(12, int(model_h * 0.08))
    shadow_mask = Image.new("L", (shadow_w, shadow_h), 0)
    pixels = []
    cx = (shadow_w - 1) * 0.5
    cy = (shadow_h - 1) * 0.5
    sx = max(1.0, shadow_w * 0.22)
    sy = max(1.0, shadow_h * 0.30)
    for y in range(shadow_h):
        for x in range(shadow_w):
            dx = (x - cx) / sx
            dy = (y - cy) / sy
            value = int(255.0 * math.exp(-(dx * dx + dy * dy)))
            pixels.append(value)
    shadow_mask.putdata(pixels)

    shadow_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    shadow_left = bbox[0] + (model_w - shadow_w) // 2
    shadow_top = min(size[1] - shadow_h, bbox[3] - int(shadow_h * 0.15))
    shadow_color = (0, 0, 0, 28)
    shadow_patch = Image.new("RGBA", (shadow_w, shadow_h), shadow_color)
    shadow_layer.paste(shadow_patch, (shadow_left, shadow_top), shadow_mask)
    return shadow_layer


def process_file(path: Path, input_suffix: str, output_suffix: str, overwrite: bool):
    stem = path.name[: -len(input_suffix)] if path.name.endswith(input_suffix) else path.stem
    out_path = path.with_name(f"{stem}{output_suffix}")
    if out_path.exists() and not overwrite:
        print(f"SKIP {out_path}")
        return

    img = Image.open(path).convert("RGBA")
    alpha = img.getchannel("A")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(make_shadow(alpha, img.size))
    bg.alpha_composite(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(out_path)
    print(f"WRITE {out_path}")


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    paths = sorted(input_root.rglob(f"*{args.input_suffix}"))
    include_stems = {stem.strip() for stem in args.include_stems.split(",") if stem.strip()}
    exclude_dirnames = {name.strip() for name in args.exclude_dirnames.split(",") if name.strip()}
    for path in paths:
        if exclude_dirnames and any(part in exclude_dirnames for part in path.parts):
            continue
        stem = path.name[: -len(args.input_suffix)] if path.name.endswith(args.input_suffix) else path.stem
        if include_stems and stem not in include_stems:
            continue
        process_file(path, args.input_suffix, args.output_suffix, args.overwrite)


if __name__ == "__main__":
    main()
