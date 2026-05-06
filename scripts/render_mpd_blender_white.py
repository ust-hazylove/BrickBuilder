import argparse
import bmesh
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


PART_GRID = {
    "3005.dat": (1, 1),  # 1x1
    "3004.dat": (2, 1),  # 1x2
    "3622.dat": (3, 1),  # 1x3
    "3010.dat": (4, 1),  # 1x4
    "3009.dat": (6, 1),  # 1x6
    "3008.dat": (8, 1),  # 1x8
    "3003.dat": (2, 2),  # 2x2
    "3002.dat": (3, 2),  # 2x3
    "3001.dat": (4, 2),  # 2x4
    "2456.dat": (6, 2),  # 2x6
    "3007.dat": (8, 2),  # 2x8
}


STUD = 1.0
BRICK_HEIGHT = 24.0 / 20.0
CLAY_BRICK = (0.82, 0.77, 0.42, 1.0)
GROUND_WHITE = (1.0, 1.0, 1.0, 1.0)
GAP_XY = 0.12
GAP_Z = 0.06


def set_enum_if_available(owner, attr_name, preferred, fallback=None):
    prop = owner.bl_rna.properties.get(attr_name)
    if prop is None:
        return
    valid = [item.identifier for item in prop.enum_items]
    if preferred in valid:
        setattr(owner, attr_name, preferred)
    elif fallback in valid:
        setattr(owner, attr_name, fallback)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render exporter-style MPD files to white-background PNGs in Blender.")
    parser.add_argument("--input_root", required=True, help="Root directory to scan for .mpd files.")
    parser.add_argument("--resolution", type=int, default=2048, help="Square render resolution.")
    parser.add_argument(
        "--output_suffix",
        default="_white_render_hd.png",
        help="Suffix appended to the MPD stem to build the output PNG filename.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing renders.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of MPDs to render.")
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world)


def configure_scene(scene, resolution):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
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
    scene.view_settings.exposure = 0.12
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
    bg.inputs[1].default_value = 0.9

    scene.use_nodes = False


def add_studio_lights(collection):
    lights = [
        ("KeyArea", "AREA", (5.4, -6.2, 8.2), (math.radians(58.0), 0.0, math.radians(36.0)), 2200.0, True, 15.0, 12.0),
        ("FillArea", "AREA", (-5.8, -4.6, 6.4), (math.radians(70.0), 0.0, math.radians(-28.0)), 1600.0, False, 18.0, 14.0),
        ("TopArea", "AREA", (0.0, -0.4, 11.0), (0.0, 0.0, 0.0), 900.0, False, 16.0, 16.0),
    ]
    for name, light_type, location, rotation, energy, use_shadow, size_x, size_y in lights:
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_data.use_shadow = use_shadow
        light_data.shadow_soft_size = 7.0
        if light_type == "AREA":
            light_data.shape = "RECTANGLE"
            light_data.size = size_x
            light_data.size_y = size_y
        light_obj = bpy.data.objects.new(name, light_data)
        light_obj.location = location
        light_obj.rotation_euler = rotation
        collection.objects.link(light_obj)


def get_material(cache):
    if "brick" in cache:
        return cache["brick"]

    mat = bpy.data.materials.new(name="BrickMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    ao_node = nodes.new(type="ShaderNodeAmbientOcclusion")
    mix_node = nodes.new(type="ShaderNodeMixRGB")
    gamma_node = nodes.new(type="ShaderNodeGamma")

    ao_node.location = (-700, -70)
    mix_node.location = (-430, 10)
    gamma_node.location = (-220, 10)
    bsdf_node.location = (10, 20)
    out_node.location = (250, 20)

    mix_node.blend_type = "MULTIPLY"
    mix_node.inputs["Fac"].default_value = 0.04
    mix_node.inputs[1].default_value = CLAY_BRICK
    ao_node.inputs["Distance"].default_value = 0.14
    gamma_node.inputs["Gamma"].default_value = 1.0

    bsdf_node.inputs["Roughness"].default_value = 0.58
    if "Specular IOR Level" in bsdf_node.inputs:
        bsdf_node.inputs["Specular IOR Level"].default_value = 0.1
    elif "Specular" in bsdf_node.inputs:
        bsdf_node.inputs["Specular"].default_value = 0.1
    if "Sheen Tint" in bsdf_node.inputs:
        bsdf_node.inputs["Sheen Tint"].default_value = 0.0

    links.new(ao_node.outputs["Color"], mix_node.inputs[2])
    links.new(mix_node.outputs["Color"], gamma_node.inputs["Color"])
    links.new(gamma_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])
    cache["brick"] = mat
    return mat


def get_ground_material(cache):
    if "ground" in cache:
        return cache["ground"]

    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.inputs["Base Color"].default_value = GROUND_WHITE
    bsdf_node.inputs["Roughness"].default_value = 1.0
    bsdf_node.inputs["Specular"].default_value = 0.0
    bsdf_node.inputs["Alpha"].default_value = 1.0
    links.new(bsdf_node.outputs["BSDF"], out_node.inputs["Surface"])
    cache["ground"] = mat
    return mat


def is_supported_axis_aligned_matrix(matrix_tokens):
    if len(matrix_tokens) != 9:
        return False

    values = [float(token) for token in matrix_tokens]
    rows = [values[0:3], values[3:6], values[6:9]]
    cols = list(zip(*rows))

    for value in values:
        if abs(value) > 1e-6 and abs(abs(value) - 1.0) > 1e-6:
            return False

    for row in rows:
        if sum(1 for value in row if abs(value) > 1e-6) != 1:
            return False
    for col in cols:
        if sum(1 for value in col if abs(value) > 1e-6) != 1:
            return False
    return True


def ldraw_matrix_to_blender(matrix_tokens):
    values = [float(token) for token in matrix_tokens]
    ldraw = Matrix(
        (
            (values[0], values[1], values[2]),
            (values[3], values[4], values[5]),
            (values[6], values[7], values[8]),
        )
    )
    perm = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    return perm @ ldraw @ perm


def create_part_mesh(part_name, cache):
    part_name = part_name.lower()
    if part_name in cache:
        return cache[part_name]

    dims = PART_GRID.get(part_name)
    if dims is None:
        return None

    dx, dy = dims
    sx = max(dx * STUD - GAP_XY, 0.2)
    sy = max(dy * STUD - GAP_XY, 0.2)
    sz = max(BRICK_HEIGHT - GAP_Z, 0.2)

    x = sx / 2.0
    y = sy / 2.0
    z = sz / 2.0
    mesh = bpy.data.meshes.new(name=f"Mesh_{part_name}")
    bm = bmesh.new()
    cube = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=cube["verts"], vec=(x, y, z))

    stud_radius = 0.24
    stud_height = 0.18
    top_z = z + stud_height / 2.0 - 0.01
    x0 = -sx / 2.0 + 0.5
    y0 = -sy / 2.0 + 0.5
    for ix in range(dx):
        for iy in range(dy):
            cx = x0 + ix * STUD
            cy = y0 + iy * STUD
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=20,
                radius1=stud_radius,
                radius2=stud_radius,
                depth=stud_height,
                matrix=Matrix.Translation((cx, cy, top_z)),
            )

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    cache[part_name] = mesh
    return mesh


def iter_part_lines(path: Path):
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("1 "):
            continue
        tokens = line.split()
        if len(tokens) < 15:
            continue
        yield tokens


def build_objects_for_mpd(path: Path):
    mesh_cache = {}
    mat_cache = {}
    objects = []
    collection = bpy.context.scene.collection

    for idx, tokens in enumerate(iter_part_lines(path)):
        x_ldr = float(tokens[2])
        y_ldr = float(tokens[3])
        z_ldr = float(tokens[4])
        matrix_tokens = tokens[5:14]
        part_name = tokens[14].lower()

        # We support axis-aligned exporter MPDs, including the rotate_x-fixed
        # files whose transform matrices are diagonal with +/-1 entries.
        if not is_supported_axis_aligned_matrix(matrix_tokens):
            continue

        base_mesh = create_part_mesh(part_name, mesh_cache)
        if base_mesh is None:
            continue

        obj = bpy.data.objects.new(f"{part_name}_{idx}", base_mesh.copy())
        location = Vector(
            (
            x_ldr / 20.0,
            z_ldr / 20.0,
            y_ldr / 20.0,
            )
        )
        obj.matrix_world = Matrix.Translation(location) @ ldraw_matrix_to_blender(matrix_tokens).to_4x4()
        obj.data.materials.append(get_material(mat_cache))
        collection.objects.link(obj)
        objects.append(obj)

    return objects


def create_camera_for_objects(objects):
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.type = "ORTHO"
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    if not objects:
        cam_obj.location = (18.0, -18.0, 18.0)
        cam_obj.rotation_euler = (math.radians(54.0), 0.0, math.radians(42.0))
        cam_data.ortho_scale = 20.0
        return cam_obj

    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))

    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) / 2.0
    span = max_v - min_v
    radius = max(span.x, span.y, span.z, 1.0)

    offset = Vector((1.15, -1.45, 1.0)).normalized()
    cam_obj.location = center + offset * (radius * 3.0)
    forward = center - cam_obj.location
    cam_obj.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

    inv = cam_obj.matrix_world.inverted()
    local = [inv @ v for v in corners]
    span_x = max(v.x for v in local) - min(v.x for v in local)
    span_y = max(v.y for v in local) - min(v.y for v in local)

    cam_data.ortho_scale = max(span_x, span_y) * 1.18
    cam_data.clip_start = 0.1
    cam_data.clip_end = radius * 20.0
    return cam_obj


def create_ground_plane(objects):
    if not objects:
        return None

    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))

    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) / 2.0
    span = max_v - min_v
    plane_size = max(span.x, span.y, 1.0) * 8.0

    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(center.x, center.y, min_v.z - 0.03))
    plane = bpy.context.active_object
    plane.name = "GroundPlane"
    plane.data.materials.clear()
    plane.data.materials.append(get_ground_material({}))
    if hasattr(plane, "is_shadow_catcher"):
        plane.is_shadow_catcher = True
    return plane


def render_mpd(mpd_path: Path, output_path: Path, resolution: int):
    reset_scene()
    configure_scene(bpy.context.scene, resolution)
    add_studio_lights(bpy.context.scene.collection)
    objects = build_objects_for_mpd(mpd_path)
    create_camera_for_objects(objects)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    mpd_files = sorted(input_root.rglob("*.mpd"))
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
            render_mpd(mpd_path, output_path, args.resolution)
            rendered += 1
        except Exception as exc:
            print(f"ERROR {mpd_path}: {exc}")

    print(f"RENDERED={rendered}")
    print(f"SKIPPED={skipped}")


if __name__ == "__main__":
    main()
