import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector


LDR_PATH = Path(r"third_party/LDraw")
IMPORT_LDRAW_ZIP = Path(r"third_party/LDraw/importldraw1.2.2.zip")


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render *_floating_red.mpd diagnostic files.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--resolution", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def set_enum_if_available(owner, attr_name, preferred, fallback=None):
    prop = owner.bl_rna.properties.get(attr_name)
    if prop is None:
        return
    valid = [item.identifier for item in prop.enum_items]
    if preferred in valid:
        setattr(owner, attr_name, preferred)
    elif fallback in valid:
        setattr(owner, attr_name, fallback)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
    ):
        for datablock in list(datablocks):
            if getattr(datablock, "users", 0) == 0:
                datablocks.remove(datablock)


def configure_scene(resolution: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.03
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 5
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 2

    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.use_freestyle = False

    set_enum_if_available(scene.view_settings, "view_transform", "Standard", fallback="Filmic")
    set_enum_if_available(scene.view_settings, "look", "None", fallback="None")
    scene.view_settings.exposure = -1.15
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"

    try:
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA"):
            try:
                cycles_prefs.compute_device_type = device_type
                break
            except Exception:
                continue
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            device.use = device.type in {"OPTIX", "CUDA"}
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"

    world = bpy.data.worlds.new("WhiteWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.06


def prepare_mpd_for_import(mpd_path: Path) -> Path:
    lines = mpd_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cleaned = []
    changed = False
    for line in lines:
        if line.strip().startswith("0 ORIENTATION_APPLIED"):
            changed = True
            continue
        cleaned.append(line)
    if not changed:
        return mpd_path
    temp_dir = Path(tempfile.gettempdir()) / "img2build_ldraw_clean"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / mpd_path.name
    temp_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return temp_path


def import_mpd(mpd_path: Path):
    if IMPORT_LDRAW_ZIP.exists():
        zip_path = str(IMPORT_LDRAW_ZIP)
        if zip_path not in sys.path:
            sys.path.insert(0, zip_path)
    bpy.ops.preferences.addon_enable(module="io_scene_importldraw")
    bpy.ops.import_scene.importldraw(
        filepath=str(prepare_mpd_for_import(mpd_path)),
        ldrawPath=str(LDR_PATH),
        realScale=1.0,
        look="normal",
        addEnvironment=False,
        positionCamera=False,
        cameraBorderPercentage=5.0,
        colourScheme="ldraw",
        resPrims="Standard",
        smoothParts=True,
        bevelEdges=False,
        bevelWidth=0.2,
        addGaps=False,
        gapWidthMM=0.2,
        curvedWalls=False,
        importCameras=False,
        linkParts=True,
        useUnofficialParts=True,
        useLogoStuds=False,
        instanceStuds=False,
        positionOnGround=False,
        numberNodes=False,
        flatten=True,
        minifigHierarchy=False,
        resolveNormals="guess",
    )


def collect_mesh_objects():
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def compute_bounds(objects):
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v


def create_camera(objects):
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.type = "ORTHO"
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    min_v, max_v = compute_bounds(objects)
    center = (min_v + max_v) / 2.0
    span = max_v - min_v
    radius = max(span.x, span.y, span.z, 0.08)
    offset = Vector((1.2, -1.5, 1.05)).normalized()
    cam_obj.location = center + offset * (radius * 3.3)
    cam_obj.rotation_euler = (center - cam_obj.location).to_track_quat("-Z", "Y").to_euler()

    inv = cam_obj.matrix_world.inverted()
    local = []
    for obj in objects:
        for corner in obj.bound_box:
            local.append(inv @ (obj.matrix_world @ Vector(corner)))
    span_x = max(v.x for v in local) - min(v.x for v in local)
    span_y = max(v.y for v in local) - min(v.y for v in local)
    cam_data.ortho_scale = max(span_x, span_y) * 1.18
    cam_data.clip_start = 0.001
    cam_data.clip_end = radius * 20.0
    return center, radius


def add_lights(center: Vector, radius: float):
    lights = [
        ("KeyArea", (1.7, -2.4, 2.7), 58.0, 2.8, (1.0, 0.94, 0.88)),
        ("FillArea", (-2.2, -1.4, 1.7), 22.0, 3.4, (0.82, 0.91, 1.0)),
        ("TopArea", (0.0, 0.1, 3.4), 20.0, 3.0, (1.0, 1.0, 1.0)),
    ]
    for name, rel, energy, size_mul, color in lights:
        data = bpy.data.lights.new(name=name, type="AREA")
        data.energy = energy * max(radius, 0.08) * 4.0
        data.use_shadow = True
        data.shape = "RECTANGLE"
        data.size = max(radius * size_mul, 0.08)
        data.size_y = max(radius * size_mul * 0.75, 0.08)
        data.color = color
        obj = bpy.data.objects.new(name, data)
        obj.location = center + Vector(rel) * max(radius, 0.08)
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(obj)


def render_one(mpd_path: Path, output_path: Path, resolution: int):
    reset_scene()
    configure_scene(resolution)
    import_mpd(mpd_path)
    objects = collect_mesh_objects()
    if not objects:
        raise RuntimeError(f"No mesh objects imported from {mpd_path}")
    center, radius = create_camera(objects)
    add_lights(center, radius)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    root = Path(args.input_root).resolve()
    mpd_files = sorted(root.rglob("*_floating_red.mpd"))
    if args.limit > 0:
        mpd_files = mpd_files[: args.limit]

    rendered = 0
    skipped = 0
    for mpd_path in mpd_files:
        output_path = mpd_path.with_name(f"{mpd_path.stem}_render.png")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"SKIP {output_path}")
            continue
        print(f"RENDER {mpd_path} -> {output_path}")
        try:
            render_one(mpd_path, output_path, args.resolution)
            rendered += 1
        except Exception as exc:
            print(f"ERROR {mpd_path}: {exc}")
    print(f"RENDERED={rendered}")
    print(f"SKIPPED={skipped}")


if __name__ == "__main__":
    main()
