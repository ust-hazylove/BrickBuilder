from pathlib import Path


METHOD_TO_TARGET = {
    "instantmesh": "swap_yz_mirror_z",
    "triposr": "swap_yz",
    "stable3dgen": "rotate_x_180",
    "cube_v05": "rotate_x_180",
    "trellis_text_large": "swap_yz_mirror_y",
}


TRANSFORMS = {
    "identity": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "rotate_x_180": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "swap_yz": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
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
    if not xs:
        return None
    return (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )


def apply_linear_transform(lines, transform, label):
    center = bbox_center(lines)
    if center is None:
        return lines

    out = [f"0 ORIENTATION_APPLIED {label}\n"]
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


def strip_existing_headers(lines):
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("0 ROTATE_180 axis=x pivot="):
            continue
        if stripped.startswith("0 ORIENTATION_APPLIED "):
            continue
        out.append(line)
    return out


def recover_original(lines):
    # The current qualitative pack was previously batch-fixed with rotate_x_180.
    # Since this transform is self-inverse, applying it once reconstructs the
    # original orientation before the method-specific rule is applied.
    clean = strip_existing_headers(lines)
    recovered = apply_linear_transform(clean, TRANSFORMS["rotate_x_180"], "recover_original")
    return recovered[1:]


def fix_file(path):
    target_name = METHOD_TO_TARGET[path.stem]
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    original = recover_original(lines)
    fixed = apply_linear_transform(original, TRANSFORMS[target_name], target_name)
    path.write_text("".join(fixed), encoding="utf-8")
    print(f"FIXED {path} -> {target_name}")


def main():
    root = Path(r"qualitative_pack")
    fixed = 0
    skipped = 0

    for path in sorted(root.rglob("*.mpd")):
        if path.stem not in METHOD_TO_TARGET:
            skipped += 1
            continue
        fix_file(path)
        fixed += 1

    print(f"FIXED_COUNT={fixed}")
    print(f"SKIPPED_COUNT={skipped}")


if __name__ == "__main__":
    main()
