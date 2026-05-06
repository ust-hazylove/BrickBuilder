from pathlib import Path


SAMPLES = [
    Path(r"qualitative_pack\image_compare\chest_of_drawers_0010\instantmesh.mpd"),
    Path(r"qualitative_pack\image_compare\desks_0036\instantmesh.mpd"),
    Path(r"qualitative_pack\image_compare\softtoys_0030\instantmesh.mpd"),
    Path(r"qualitative_pack\image_compare\chest_of_drawers_0010\triposr.mpd"),
    Path(r"qualitative_pack\image_compare\desks_0036\triposr.mpd"),
    Path(r"qualitative_pack\image_compare\softtoys_0030\triposr.mpd"),
    Path(r"qualitative_pack\image_compare\chest_of_drawers_0056\stable3dgen.mpd"),
    Path(r"qualitative_pack\text_compare\dining_chair\cube_v05.mpd"),
    Path(r"qualitative_pack\text_compare\ceramic_mug\cube_v05.mpd"),
    Path(r"qualitative_pack\text_compare\watering_can\cube_v05.mpd"),
    Path(r"qualitative_pack\text_compare\dining_chair\trellis_text_large.mpd"),
    Path(r"qualitative_pack\text_compare\ceramic_mug\trellis_text_large.mpd"),
    Path(r"qualitative_pack\text_compare\watering_can\trellis_text_large.mpd"),
]


TRANSFORMS = {
    "identity": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "rotate_x_180": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "rotate_y_180": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "rotate_z_180": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "mirror_x": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "mirror_y": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "mirror_z": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "swap_xy": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "swap_xz": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
    "swap_yz": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "swap_yz_mirror_x": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "swap_yz_mirror_y": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "swap_yz_mirror_z": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}


def matmul3(a, b):
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def matvec3(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def parse_type1(line):
    tokens = line.strip().split()
    if len(tokens) < 15 or tokens[0] != "1":
        return None
    return tokens


def bbox_center(lines):
    xs = []
    ys = []
    zs = []
    for line in lines:
        tokens = parse_type1(line)
        if tokens is None:
            continue
        xs.append(float(tokens[2]))
        ys.append(float(tokens[3]))
        zs.append(float(tokens[4]))
    return (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )


def apply_linear_transform(lines, transform, label):
    center = bbox_center(lines)
    out = [f"0 ORIENTATION_TRIAL {label}\n"]
    for line in lines:
        tokens = parse_type1(line)
        if tokens is None:
            out.append(line if line.endswith("\n") else line + "\n")
            continue

        pos = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        rel = [pos[i] - center[i] for i in range(3)]
        new_rel = matvec3(transform, rel)
        new_pos = [new_rel[i] + center[i] for i in range(3)]

        basis = [
            [float(tokens[5]), float(tokens[6]), float(tokens[7])],
            [float(tokens[8]), float(tokens[9]), float(tokens[10])],
            [float(tokens[11]), float(tokens[12]), float(tokens[13])],
        ]
        new_basis = matmul3(transform, basis)
        part = " ".join(tokens[14:])

        out.append(
            "1 {color} {x:.6f} {y:.6f} {z:.6f} "
            "{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} "
            "{g:.6f} {h:.6f} {i:.6f} {part}\n".format(
                color=tokens[1],
                x=new_pos[0],
                y=new_pos[1],
                z=new_pos[2],
                a=new_basis[0][0],
                b=new_basis[0][1],
                c=new_basis[0][2],
                d=new_basis[1][0],
                e=new_basis[1][1],
                f=new_basis[1][2],
                g=new_basis[2][0],
                h=new_basis[2][1],
                i=new_basis[2][2],
                part=part,
            )
        )
    return out


def strip_existing_trial_headers(lines):
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("0 ROTATE_180 axis=x pivot="):
            continue
        if stripped.startswith("0 ORIENTATION_TRIAL "):
            continue
        out.append(line)
    return out


def recover_original(lines):
    # Current non-ours MPDs were already fixed with rotate_x once.
    # rotate_x_180 is self-inverse, so applying it again reconstructs the original.
    lines = strip_existing_trial_headers(lines)
    return apply_linear_transform(lines, TRANSFORMS["rotate_x_180"], "recover_original")[1:]


def reference_image_for(sample):
    if "image_compare" in str(sample):
        return sample.with_name("ours_full_white_render_hd.png")
    return sample.with_name("ours_full_white_render_hd.png")


def main():
    out_root = Path(r"qualitative_pack\orientation_trials")
    out_root.mkdir(parents=True, exist_ok=True)

    for sample in SAMPLES:
        trial_dir = out_root / f"{sample.parent.name}__{sample.stem}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        lines = sample.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        base_lines = recover_original(lines)

        for name, transform in TRANSFORMS.items():
            out_path = trial_dir / f"{sample.stem}__{name}.mpd"
            out_path.write_text("".join(apply_linear_transform(base_lines, transform, name)), encoding="utf-8")

        ref_src = reference_image_for(sample)
        if ref_src.exists():
            ref_dst = trial_dir / "reference_ours_full_white_render_hd.png"
            ref_dst.write_bytes(ref_src.read_bytes())

        summary = trial_dir / "trial_list.txt"
        summary.write_text("\n".join(TRANSFORMS.keys()) + "\n", encoding="utf-8")
        print(f"READY {trial_dir}")


if __name__ == "__main__":
    main()
