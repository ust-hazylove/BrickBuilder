import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector


CLAY_BRICK = (0.84, 0.80, 0.56, 1.0)
LDR_PATH = Path(r"third_party/LDraw")
IMPORT_LDRAW_ZIP = Path(r"third_party/LDraw/importldraw1.2.2.zip")


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render MPD files through Blender's Import LDraw addon.")
    parser.add_argument("--input_root", required=True, help="Root directory to scan for .mpd files.")
    parser.add_argument("--resolution", type=int, default=1536, help="Square render resolution.")
    parser.add_argument("--output_suffix", default="_reference_rgba.png", help="Suffix appended to the MPD stem.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing renders.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of MPDs to render.")
    parser.add_argument("--rotate_x_180", action="store_true", help="Rotate imported model 180 degrees around X before rendering.")
    parser.add_argument("--rotate_y_180", action="store_true", help="Rotate imported model 180 degrees around Y before rendering.")
    parser.add_argument("--rotate_z_180", action="store_true", help="Rotate imported model 180 degrees around Z before rendering.")
    parser.add_argument(
        "--include_stems",
        default="",
        help="Comma-separated MPD stems to render, e.g. instantmesh,triposr. Empty means render all.",
    )
    parser.add_argument(
        "--exclude_dirnames",
        default="",
        help="Comma-separated directory names to skip during recursive scan.",
    )
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

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for camera in list(bpy.data.cameras):
        bpy.data.cameras.remove(camera)
    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world)


def configure_scene(scene, resolution):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.04
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 4

    scene.render.film_transparent = True
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
    scene.view_settings.exposure = -1.7
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
    bg.inputs[1].default_value = 0.015


def import_mpd(mpd_path: Path):
    if IMPORT_LDRAW_ZIP.exists():
        zip_path = str(IMPORT_LDRAW_ZIP)
        if zip_path not in sys.path:
            sys.path.insert(0, zip_path)
    import_path = prepare_mpd_for_import(mpd_path)
    bpy.ops.preferences.addon_enable(module="io_scene_importldraw")
    result = bpy.ops.import_scene.importldraw(
        filepath=str(import_path),
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
    return result


def prepare_mpd_for_import(mpd_path: Path) -> Path:
    lines = mpd_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cleaned = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("0 ORIENTATION_APPLIED"):
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


def get_or_create_clay_material():
    material = bpy.data.materials.get("ClayOverride")
    if material is not None:
        return material

    material = bpy.data.materials.new(name="ClayOverride")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    ao_node = nodes.new(type="ShaderNodeAmbientOcclusion")
    ramp_node = nodes.new(type="ShaderNodeValToRGB")
    mix_node = nodes.new(type="ShaderNodeMixRGB")

    bsdf_node.inputs["Base Color"].default_value = CLAY_BRICK
    bsdf_node.inputs["Roughness"].default_value = 0.92
    if "Specular" in bsdf_node.inputs:
        bsdf_node.inputs["Specular"].default_value = 0.18
    ao_node.inputs["Distance"].default_value = 0.02
    ramp_node.color_ramp.elements[0].position = 0.0
    ramp_node.color_ramp.elements[0].color = (0.67, 0.63, 0.44, 1.0)
    ramp_node.color_ramp.elements[1].position = 1.0
    ramp_node.color_ramp.elements[1].color = CLAY_BRICK
    mix_node.blend_type = "MULTIPLY"
    mix_node.inputs["Fac"].default_value = 0.32
    mix_node.inputs[1].default_value = CLAY_BRICK

    ao_node.location = (-700, -40)
    ramp_node.location = (-480, -40)
    mix_node.location = (-230, 0)
    bsdf_node.location = (40, 0)
    out_node.location = (240, 0)

    links.new(ao_node.outputs["AO"], ramp_node.inputs["Fac"])
    links.new(ramp_node.outputs["Color"], mix_node.inputs[2])
    links.new(mix_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])
    return material


def collect_mesh_objects():
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def override_materials(objects):
    clay = get_or_create_clay_material()
    for obj in objects:
        if not obj.data.materials:
            obj.data.materials.append(clay)
            continue
        for idx in range(len(obj.data.materials)):
            obj.data.materials[idx] = clay


def rotate_objects(
    objects,
    rotate_x_180: bool = False,
    rotate_y_180: bool = False,
    rotate_z_180: bool = False,
):
    min_v, max_v = compute_bounds(objects)
    center = (min_v + max_v) / 2.0

    pivot = bpy.data.objects.new("ModelPivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = center

    bpy.context.view_layer.update()
    for obj in objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
        obj.matrix_world = world_matrix

    if rotate_x_180:
        pivot.rotation_euler[0] += math.pi
    if rotate_y_180:
        pivot.rotation_euler[1] += math.pi
    if rotate_z_180:
        pivot.rotation_euler[2] += math.pi
    bpy.context.view_layer.update()


def compute_bounds(objects):
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v


def add_lights(center, radius):
    collection = bpy.context.scene.collection
    lights = [
        ("KeyArea", (1.65, -2.15, 2.35), 62.0, False, 2.4),
        ("FillArea", (-1.7, -1.1, 1.65), 20.0, False, 2.8),
        ("TopArea", (0.0, -0.25, 3.0), 10.0, False, 3.0),
    ]
    for name, rel, energy, use_shadow, size_mul in lights:
        data = bpy.data.lights.new(name=name, type="AREA")
        data.energy = energy * max(radius, 0.08) * 4.0
        data.use_shadow = use_shadow
        data.shape = "RECTANGLE"
        data.size = max(radius * size_mul, 0.08)
        data.size_y = max(radius * size_mul * 0.8, 0.08)
        obj = bpy.data.objects.new(name, data)
        obj.location = center + Vector(rel) * max(radius, 0.08)
        obj.rotation_euler = ((center - obj.location).to_track_quat("-Z", "Y").to_euler())
        collection.objects.link(obj)


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
    cam_obj.location = center + offset * (radius * 3.2)
    cam_obj.rotation_euler = (center - cam_obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    inv = cam_obj.matrix_world.inverted()
    local = []
    for obj in objects:
        for corner in obj.bound_box:
            local.append(inv @ (obj.matrix_world @ Vector(corner)))
    span_x = max(v.x for v in local) - min(v.x for v in local)
    span_y = max(v.y for v in local) - min(v.y for v in local)

    cam_data.ortho_scale = max(span_x, span_y) * 1.12
    cam_data.clip_start = 0.001
    cam_data.clip_end = radius * 20.0
    return center, radius


def cleanup_non_mesh_objects():
    for obj in list(bpy.data.objects):
        if obj.type not in {"MESH", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def render_mpd(
    mpd_path: Path,
    output_path: Path,
    resolution: int,
    rotate_x_180: bool = False,
    rotate_y_180: bool = False,
    rotate_z_180: bool = False,
):
    reset_scene()
    configure_scene(bpy.context.scene, resolution)
    import_mpd(mpd_path)
    configure_scene(bpy.context.scene, resolution)
    cleanup_non_mesh_objects()
    objects = collect_mesh_objects()
    if not objects:
        raise RuntimeError(f"No mesh objects were imported from {mpd_path}")
    if rotate_x_180 or rotate_y_180 or rotate_z_180:
        rotate_objects(
            objects,
            rotate_x_180=rotate_x_180,
            rotate_y_180=rotate_y_180,
            rotate_z_180=rotate_z_180,
        )
    override_materials(objects)
    center, radius = create_camera(objects)
    add_lights(center, radius)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    mpd_files = sorted(input_root.rglob("*.mpd"))
    include_stems = {stem.strip() for stem in args.include_stems.split(",") if stem.strip()}
    exclude_dirnames = {name.strip() for name in args.exclude_dirnames.split(",") if name.strip()}
    if exclude_dirnames:
        mpd_files = [mpd_path for mpd_path in mpd_files if not any(part in exclude_dirnames for part in mpd_path.parts)]
    if include_stems:
        mpd_files = [mpd_path for mpd_path in mpd_files if mpd_path.stem in include_stems]
    if args.limit > 0:
        mpd_files = mpd_files[: args.limit]

    rendered = 0
    skipped = 0
    for mpd_path in mpd_files:
        output_path = mpd_path.with_name(f"{mpd_path.stem}{args.output_suffix}")
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"SKIP {output_path}")
            continue
        print(f"RENDER {mpd_path} -> {output_path}")
        try:
            render_mpd(
                mpd_path,
                output_path,
                args.resolution,
                rotate_x_180=args.rotate_x_180,
                rotate_y_180=args.rotate_y_180,
                rotate_z_180=args.rotate_z_180,
            )
            rendered += 1
        except Exception as exc:
            print(f"ERROR {mpd_path}: {exc}")

    print(f"RENDERED={rendered}")
    print(f"SKIPPED={skipped}")


if __name__ == "__main__":
    main()
