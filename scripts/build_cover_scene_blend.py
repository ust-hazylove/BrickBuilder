import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector


OUT_BLEND = Path(r"qualitative_pack_100cases_20260417\paper_cover_scene_template.blend")


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
        bpy.data.curves,
    ):
        for datablock in list(datablock_collection):
            if getattr(datablock, "users", 0) == 0:
                datablock_collection.remove(datablock)


def configure_scene(scene):
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
    scene.render.resolution_x = 3200
    scene.render.resolution_y = 1800
    scene.render.resolution_percentage = 100

    set_enum_if_available(scene.view_settings, "view_transform", "Standard", fallback="Filmic")
    set_enum_if_available(scene.view_settings, "look", "None", fallback="None")
    scene.view_settings.exposure = -0.3
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"

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
    ramp.color_ramp.elements[0].color = (0.89, 0.92, 0.97, 1.0)
    ramp.color_ramp.elements[1].color = (0.97, 0.92, 0.88, 1.0)
    bg.inputs[1].default_value = 0.18

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], grad.inputs["Vector"])
    links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def make_material(name, rgba, roughness=0.9):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.14
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.14
    out.location = (220, 0)
    bsdf.location = (0, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_material(obj, mat):
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        for idx in range(len(obj.data.materials)):
            obj.data.materials[idx] = mat


def create_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def move_to_collection(obj, col):
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    col.objects.link(obj)


def create_box(name, location, scale, color, bevel=0.04, collection=None):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.modifier_add(type="BEVEL")
    mod = obj.modifiers[-1]
    mod.width = bevel
    mod.segments = 4
    mod.limit_method = "NONE"
    assign_material(obj, make_material(f"{name}_Mat", color, roughness=0.92))
    if collection is not None:
        move_to_collection(obj, collection)
    return obj


def build_environment():
    env = create_collection("Environment")
    colors = {
        "floor": (0.87, 0.89, 0.93, 1.0),
        "wall_back": (0.90, 0.91, 0.95, 1.0),
        "wall_side": (0.86, 0.89, 0.94, 1.0),
        "plinth_main": (0.84, 0.88, 0.93, 1.0),
        "plinth_left": (0.87, 0.85, 0.92, 1.0),
        "plinth_right": (0.84, 0.91, 0.91, 1.0),
        "shelf": (0.94, 0.91, 0.88, 1.0),
    }

    create_box("Floor", (0.0, 0.0, -0.08), (4.8, 3.4, 0.08), colors["floor"], bevel=0.02, collection=env)
    create_box("BackWall", (0.0, 3.35, 2.3), (4.8, 0.08, 2.45), colors["wall_back"], bevel=0.02, collection=env)
    create_box("LeftWall", (-4.75, 0.0, 2.3), (0.08, 3.4, 2.45), colors["wall_side"], bevel=0.02, collection=env)
    create_box("MainPlinth", (-0.45, 0.92, 0.30), (1.72, 0.92, 0.30), colors["plinth_main"], bevel=0.05, collection=env)
    create_box("LeftPlinth", (-2.65, 1.55, 0.18), (0.92, 0.82, 0.18), colors["plinth_left"], bevel=0.05, collection=env)
    create_box("RightPlinth", (2.38, 1.46, 0.22), (1.02, 0.90, 0.22), colors["plinth_right"], bevel=0.05, collection=env)
    create_box("MidShelf", (-1.60, -0.52, 1.16), (1.28, 0.42, 0.06), colors["shelf"], bevel=0.03, collection=env)
    create_box("RightShelf", (2.08, -0.64, 1.28), (1.18, 0.42, 0.06), colors["shelf"], bevel=0.03, collection=env)
    create_box("CenterShelf", (0.30, -0.94, 1.84), (1.18, 0.38, 0.06), colors["shelf"], bevel=0.03, collection=env)


def add_area_light(collection, name, location, rotation_deg, energy, color, size, size_y=None):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.color = color
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size if size_y is None else size_y
    obj = bpy.data.objects.new(name, data)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ")
    collection.objects.link(obj)
    return obj


def add_spot_light(collection, name, location, rotation_deg, energy, color, spot_deg=42):
    data = bpy.data.lights.new(name=name, type="SPOT")
    data.energy = energy
    data.color = color
    data.spot_size = math.radians(spot_deg)
    data.spot_blend = 0.55
    obj = bpy.data.objects.new(name, data)
    obj.location = Vector(location)
    obj.rotation_euler = Euler(tuple(math.radians(v) for v in rotation_deg), "XYZ")
    collection.objects.link(obj)
    return obj


def build_lights():
    lights = create_collection("Lights")
    add_area_light(lights, "KeyWarm", (-2.9, -3.6, 4.4), (62, 0, -28), 700, (1.0, 0.86, 0.78), 5.2, 3.7)
    add_area_light(lights, "FillCool", (3.6, -2.3, 3.4), (58, 0, 35), 300, (0.76, 0.88, 1.0), 5.8, 4.1)
    add_area_light(lights, "RimRose", (0.0, 2.9, 3.9), (-70, 0, 180), 220, (1.0, 0.80, 0.86), 4.6, 3.2)
    add_spot_light(lights, "HeroSpot", (0.0, -1.5, 4.9), (72, 0, 0), 420, (1.0, 0.98, 0.93), 40)


def build_camera():
    cams = create_collection("CameraRig")
    cam_data = bpy.data.cameras.new("CoverCamera")
    cam = bpy.data.objects.new("CoverCamera", cam_data)
    cam.location = Vector((0.2, -6.2, 2.45))
    cam.rotation_euler = Euler((math.radians(75.0), 0.0, 0.0), "XYZ")
    cam_data.lens = 44
    cam_data.sensor_width = 36
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 6.0
    cam_data.dof.aperture_fstop = 5.0
    cams.objects.link(cam)
    bpy.context.scene.camera = cam

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.9, 0.8))
    focus = bpy.context.active_object
    focus.name = "FocusTarget"
    move_to_collection(focus, cams)
    cam_data.dof.focus_object = focus


def add_helper_planes():
    helpers = create_collection("ModelHelpers")
    helper_mat = make_material("HelperMat", (0.98, 0.72, 0.72, 1.0), roughness=0.95)
    coords = [
        ("DropZone_A", (-2.1, -0.5, 1.82), (0.55, 0.55, 0.03)),
        ("DropZone_B", (2.4, -0.58, 1.94), (0.55, 0.55, 0.03)),
        ("DropZone_C", (0.98, 0.05, 0.98), (0.48, 0.48, 0.03)),
        ("DropZone_D", (-1.10, 0.08, 0.98), (0.52, 0.52, 0.03)),
        ("DropZone_E", (2.55, 0.56, 0.84), (0.45, 0.45, 0.03)),
        ("DropZone_F", (-2.82, 1.18, 0.53), (0.40, 0.40, 0.03)),
        ("DropZone_G", (1.84, 1.34, 0.42), (0.40, 0.40, 0.03)),
        ("DropZone_H", (-0.14, 1.28, 0.49), (0.44, 0.44, 0.03)),
        ("DropZone_I", (0.92, -0.82, 2.16), (0.48, 0.48, 0.03)),
        ("DropZone_J", (-0.24, -0.92, 2.08), (0.46, 0.46, 0.03)),
    ]
    for name, loc, scale in coords:
        bpy.ops.mesh.primitive_cube_add(location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = scale
        assign_material(obj, helper_mat)
        move_to_collection(obj, helpers)
        obj.display_type = "WIRE"

    helpers.hide_render = True
    helpers.hide_viewport = False


def add_notes():
    notes = create_collection("Notes")
    bpy.ops.object.text_add(location=(-4.15, -2.65, 3.7))
    txt = bpy.context.active_object
    txt.name = "SceneNote"
    txt.data.body = (
        "Paper Cover Template\\n"
        "1. Drag your MPD models into the scene\\n"
        "2. Put them near DropZone_* helpers\\n"
        "3. Hide ModelHelpers before final render\\n"
        "4. Ask Codex to render after your layout tweak"
    )
    txt.data.size = 0.22
    txt.rotation_euler = Euler((math.radians(90), 0.0, 0.0), "XYZ")
    move_to_collection(txt, notes)
    notes.hide_render = True


def main():
    reset_scene()
    configure_scene(bpy.context.scene)
    build_environment()
    build_lights()
    build_camera()
    add_helper_planes()
    add_notes()
    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
    print(OUT_BLEND)


if __name__ == "__main__":
    main()
