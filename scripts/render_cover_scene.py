import argparse
import math
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Euler, Vector


LDR_PATH = Path(r"third_party/LDraw")
IMPORT_LDRAW_ZIP = Path(r"third_party/LDraw/importldraw1.2.2.zip")
DATA_ROOT = Path(r"qualitative_pack_100cases_20260417")


MODEL_SPECS = [
    {
        "id": "text_desk_fan",
        "path": DATA_ROOT / "text_compare" / "desk_fan" / "ours_full.mpd",
        "color": (0.93, 0.66, 0.58, 1.0),
        "target_h": 0.90,
        "loc": (-2.10, -0.50, 1.82),
        "yaw": 22.0,
    },
    {
        "id": "text_watering_can",
        "path": DATA_ROOT / "text_compare" / "watering_can" / "ours_full.mpd",
        "color": (0.58, 0.80, 0.78, 1.0),
        "target_h": 0.88,
        "loc": (2.40, -0.58, 1.94),
        "yaw": -28.0,
    },
    {
        "id": "text_round_stool",
        "path": DATA_ROOT / "text_compare" / "round_stool" / "ours_full.mpd",
        "color": (0.73, 0.67, 0.86, 1.0),
        "target_h": 0.95,
        "loc": (0.98, 0.05, 0.98),
        "yaw": -32.0,
    },
    {
        "id": "text_dining_chair",
        "path": DATA_ROOT / "text_compare" / "dining_chair" / "ours_full.mpd",
        "color": (0.64, 0.81, 0.68, 1.0),
        "target_h": 1.14,
        "loc": (-1.10, 0.08, 0.98),
        "yaw": 26.0,
    },
    {
        "id": "text_standing_mirror",
        "path": DATA_ROOT / "text_compare" / "standing_mirror" / "ours_full.mpd",
        "color": (0.90, 0.72, 0.56, 1.0),
        "target_h": 1.18,
        "loc": (2.55, 0.56, 0.84),
        "yaw": -18.0,
    },
    {
        "id": "text_succulent_planter",
        "path": DATA_ROOT / "text_compare" / "succulent_planter" / "ours_full.mpd",
        "color": (0.79, 0.67, 0.86, 1.0),
        "target_h": 0.76,
        "loc": (-2.82, 1.18, 0.53),
        "yaw": 34.0,
    },
    {
        "id": "image_softtoy",
        "path": DATA_ROOT / "image_compare" / "softtoys_0011" / "ours_full.mpd",
        "color": (0.72, 0.69, 0.86, 1.0),
        "target_h": 0.78,
        "loc": (1.84, 1.34, 0.42),
        "yaw": -24.0,
    },
    {
        "id": "image_chair",
        "path": DATA_ROOT / "image_compare" / "chair_0005" / "ours_full.mpd",
        "color": (0.60, 0.76, 0.86, 1.0),
        "target_h": 0.96,
        "loc": (-0.14, 1.28, 0.49),
        "yaw": 146.0,
    },
    {
        "id": "image_desk",
        "path": DATA_ROOT / "image_compare" / "desks_0036" / "ours_full.mpd",
        "color": (0.90, 0.70, 0.58, 1.0),
        "target_h": 0.74,
        "loc": (0.92, -0.82, 2.16),
        "yaw": -22.0,
    },
    {
        "id": "image_coffee_maker",
        "path": DATA_ROOT / "image_compare" / "coffee_and_tea_makers_0038" / "ours_full.mpd",
        "color": (0.60, 0.75, 0.88, 1.0),
        "target_h": 1.05,
        "loc": (-0.24, -0.92, 2.08),
        "yaw": 10.0,
    },
]


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render a curated paper-cover scene from MPD models.")
    parser.add_argument(
        "--output",
        default=str(DATA_ROOT / "paper_cover_scene_v1.png"),
        help="Output image path.",
    )
    parser.add_argument("--resolution_x", type=int, default=3200)
    parser.add_argument("--resolution_y", type=int, default=1800)
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
    for datablock_collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
        bpy.data.images,
    ):
        for datablock in list(datablock_collection):
            if getattr(datablock, "users", 0) == 0:
                datablock_collection.remove(datablock)


def configure_scene(scene, resolution_x, resolution_y):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 8

    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
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

    world = bpy.data.worlds.new("CoverWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        nodes.remove(node)

    out = nodes.new(type="ShaderNodeOutputWorld")
    bg = nodes.new(type="ShaderNodeBackground")
    grad = nodes.new(type="ShaderNodeTexGradient")
    coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    ramp = nodes.new(type="ShaderNodeValToRGB")

    coord.location = (-900, 0)
    mapping.location = (-700, 0)
    grad.location = (-500, 0)
    ramp.location = (-280, 0)
    bg.location = (-60, 0)
    out.location = (150, 0)

    mapping.inputs["Rotation"].default_value[1] = math.radians(90.0)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.88, 0.91, 0.96, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.94, 0.90, 0.86, 1.0)
    bg.inputs[1].default_value = 0.12

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
    links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


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
    import_path = prepare_mpd_for_import(mpd_path)
    bpy.ops.preferences.addon_enable(module="io_scene_importldraw")
    bpy.ops.import_scene.importldraw(
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


def collect_new_mesh_objects(existing_names):
    return [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name not in existing_names
    ]


def compute_bounds(objects):
    bpy.context.view_layer.update()
    corners = []
    for obj in objects:
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return min_v, max_v


def create_model_material(name, rgba):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.9
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.14
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.14

    out.location = (320, 0)
    bsdf.location = (80, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return material


def assign_material(objects, material):
    for obj in objects:
        if not obj.data.materials:
            obj.data.materials.append(material)
        else:
            for idx in range(len(obj.data.materials)):
                obj.data.materials[idx] = material


def create_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(objects, collection):
    for obj in objects:
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)


def normalize_and_place(objects, target_h, location, yaw_deg):
    min_v, max_v = compute_bounds(objects)
    span = max_v - min_v
    height = max(span.z, 1e-6)
    scale = target_h / height

    pivot = bpy.data.objects.new(f"Pivot_{objects[0].name}", None)
    bpy.context.scene.collection.objects.link(pivot)
    center = (min_v + max_v) / 2.0
    pivot.location = center
    bpy.context.view_layer.update()

    for obj in objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
        obj.matrix_world = world_matrix

    pivot.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    min_v2, max_v2 = compute_bounds(objects)
    center2 = (min_v2 + max_v2) / 2.0
    pivot.location += Vector((location[0] - center2.x, location[1] - center2.y, location[2] - min_v2.z))
    pivot.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
    bpy.context.view_layer.update()
    return pivot


def create_box(name, location, scale, color, bevel=0.06):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.modifier_add(type="BEVEL")
    mod = obj.modifiers[-1]
    mod.width = bevel
    mod.segments = 4
    mod.limit_method = "NONE"
    material = create_model_material(f"{name}Mat", color)
    assign_material([obj], material)
    return obj


def build_environment():
    env_collection = create_collection("Environment")
    pieces = [
        create_box("Floor", (0.0, 0.0, -0.08), (4.8, 3.4, 0.08), (0.88, 0.89, 0.92, 1.0), bevel=0.02),
        create_box("BackWall", (0.0, 3.35, 2.3), (4.8, 0.08, 2.45), (0.90, 0.91, 0.95, 1.0), bevel=0.02),
        create_box("LeftWall", (-4.75, 0.0, 2.3), (0.08, 3.4, 2.45), (0.86, 0.89, 0.94, 1.0), bevel=0.02),
        create_box("MainPlinth", (-0.45, 0.92, 0.30), (1.72, 0.92, 0.30), (0.85, 0.89, 0.94, 1.0)),
        create_box("LeftPlinth", (-2.65, 1.55, 0.18), (0.92, 0.82, 0.18), (0.88, 0.86, 0.92, 1.0)),
        create_box("RightPlinth", (2.38, 1.46, 0.22), (1.02, 0.90, 0.22), (0.85, 0.92, 0.92, 1.0)),
        create_box("MidShelf", (-1.60, -0.52, 1.16), (1.28, 0.42, 0.06), (0.93, 0.90, 0.88, 1.0)),
        create_box("RightShelf", (2.08, -0.64, 1.28), (1.18, 0.42, 0.06), (0.93, 0.90, 0.88, 1.0)),
        create_box("CenterShelf", (0.30, -0.94, 1.84), (1.18, 0.38, 0.06), (0.93, 0.90, 0.88, 1.0)),
    ]
    link_to_collection(pieces, env_collection)
    return env_collection


def add_lights():
    collection = create_collection("Lights")
    lights = [
        ("KeyWarm", "AREA", (-2.9, -3.6, 4.4), (math.radians(62), 0.0, math.radians(-28)), 520, (1.0, 0.86, 0.77), 5.2),
        ("FillCool", "AREA", (3.6, -2.3, 3.4), (math.radians(58), 0.0, math.radians(35)), 240, (0.74, 0.87, 1.0), 5.8),
        ("RimRose", "AREA", (0.0, 2.9, 3.9), (math.radians(-70), 0.0, math.radians(180)), 160, (1.0, 0.78, 0.84), 4.6),
        ("SpotHero", "SPOT", (0.0, -1.5, 4.9), (math.radians(72), 0.0, 0.0), 360, (1.0, 0.98, 0.92), 0.0),
    ]
    for name, light_type, loc, rot, energy, color, area_size in lights:
        data = bpy.data.lights.new(name=name, type=light_type)
        data.energy = energy
        data.color = color
        data.use_shadow = True
        if light_type == "AREA":
            data.shape = "RECTANGLE"
            data.size = area_size
            data.size_y = area_size * 0.72
        else:
            data.spot_size = math.radians(42.0)
            data.spot_blend = 0.55
        obj = bpy.data.objects.new(name, data)
        obj.location = Vector(loc)
        obj.rotation_euler = Euler(rot, "XYZ")
        collection.objects.link(obj)


def create_camera():
    cam_data = bpy.data.cameras.new("CoverCamera")
    cam_obj = bpy.data.objects.new("CoverCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_obj.location = Vector((0.2, -6.2, 2.45))
    cam_obj.rotation_euler = Euler((math.radians(75.0), 0.0, 0.0), "XYZ")
    cam_data.lens = 44
    cam_data.sensor_width = 36
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 6.0
    cam_data.dof.aperture_fstop = 5.0


def import_and_place_models():
    model_collection = create_collection("Models")
    for spec in MODEL_SPECS:
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing MPD: {spec['path']}")
        existing = {obj.name for obj in bpy.data.objects}
        import_mpd(spec["path"])
        imported = collect_new_mesh_objects(existing)
        if not imported:
            raise RuntimeError(f"No mesh objects imported from {spec['path']}")
        link_to_collection(imported, model_collection)
        material = create_model_material(f"{spec['id']}_Mat", spec["color"])
        assign_material(imported, material)
        normalize_and_place(imported, spec["target_h"], spec["loc"], spec["yaw"])


def render(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    reset_scene()
    configure_scene(bpy.context.scene, args.resolution_x, args.resolution_y)
    build_environment()
    add_lights()
    import_and_place_models()
    create_camera()
    render(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
