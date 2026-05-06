from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


ROOT = Path(r"qualitative_pack_100cases_20260417")
OUT = ROOT / "paper_cover_composite_v1.png"
W, H = 3200, 1800


ITEMS = [
    {
        "src": ROOT / "text_compare" / "desk_fan" / "ours_full_reference_rgba.png",
        "box": (140, 120, 580, 720),
        "color": (242, 177, 164),
    },
    {
        "src": ROOT / "image_compare" / "desks_0036" / "ours_full_reference_rgba.png",
        "box": (720, 220, 1180, 760),
        "color": (185, 220, 231),
    },
    {
        "src": ROOT / "text_compare" / "watering_can" / "ours_full_reference_rgba.png",
        "box": (2470, 170, 3020, 760),
        "color": (164, 225, 214),
    },
    {
        "src": ROOT / "image_compare" / "coffee_and_tea_makers_0038" / "ours_full_reference_rgba.png",
        "box": (1840, 180, 2260, 700),
        "color": (168, 203, 241),
    },
    {
        "src": ROOT / "text_compare" / "dining_chair" / "ours_full_reference_rgba.png",
        "box": (650, 740, 1060, 1380),
        "color": (182, 221, 191),
    },
    {
        "src": ROOT / "image_compare" / "chair_0005" / "ours_full_reference_rgba.png",
        "box": (1240, 760, 1650, 1400),
        "color": (184, 203, 240),
    },
    {
        "src": ROOT / "text_compare" / "round_stool" / "ours_full_reference_rgba.png",
        "box": (1940, 770, 2390, 1380),
        "color": (202, 184, 236),
    },
    {
        "src": ROOT / "text_compare" / "standing_mirror" / "ours_full_reference_rgba.png",
        "box": (2680, 760, 3050, 1410),
        "color": (239, 197, 154),
    },
    {
        "src": ROOT / "text_compare" / "succulent_planter" / "ours_full_reference_rgba.png",
        "box": (150, 1120, 520, 1610),
        "color": (201, 181, 235),
    },
    {
        "src": ROOT / "image_compare" / "softtoys_0011" / "ours_full_reference_rgba.png",
        "box": (2400, 1190, 2780, 1660),
        "color": (194, 187, 235),
    },
]


def lerp(a, b, t):
    return int(a + (b - a) * t)


def build_background():
    img = Image.new("RGB", (W, H), (235, 239, 245))
    px = img.load()
    for y in range(H):
        ty = y / max(H - 1, 1)
        top = (233, 239, 247)
        bottom = (248, 240, 234)
        row = tuple(lerp(top[i], bottom[i], ty) for i in range(3))
        for x in range(W):
            tx = x / max(W - 1, 1)
            cool = (224, 236, 250)
            warm = (248, 233, 226)
            mix = tuple(lerp(cool[i], warm[i], tx) for i in range(3))
            px[x, y] = tuple((row[i] + mix[i]) // 2 for i in range(3))
    return img


def draw_ambient_glows(base):
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    d.ellipse((-200, -50, 1550, 1350), fill=(255, 191, 181, 72))
    d.ellipse((1550, -150, 3500, 1250), fill=(165, 216, 255, 64))
    d.ellipse((780, 900, 2400, 2200), fill=(255, 228, 194, 46))
    glow = glow.filter(ImageFilter.GaussianBlur(140))
    return Image.alpha_composite(base.convert("RGBA"), glow)


def rounded_rect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def add_pedestals(canvas):
    ped = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ped)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    blocks = [
        ((240, 1240, 1280, 1600), 45, (231, 236, 243, 255)),
        ((1040, 1160, 1960, 1520), 45, (236, 240, 245, 255)),
        ((1960, 1230, 2940, 1600), 45, (231, 238, 239, 255)),
        ((520, 720, 1180, 1010), 34, (246, 243, 240, 255)),
        ((1780, 700, 2460, 1000), 34, (244, 242, 239, 255)),
        ((1220, 420, 1960, 680), 34, (246, 244, 241, 255)),
    ]
    for box, radius, fill in blocks:
        shadow_box = (box[0] + 26, box[1] + 46, box[2] + 58, box[3] + 70)
        sd.rounded_rectangle(shadow_box, radius=radius + 20, fill=(110, 130, 155, 44))
        rounded_rect(d, box, radius, fill)
    shadow = shadow.filter(ImageFilter.GaussianBlur(42))
    ped = Image.alpha_composite(shadow, ped)
    ped = Image.alpha_composite(ped, ped)
    ped = Image.alpha_composite(ped, Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    ped = Image.alpha_composite(ped, ped)
    real = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    real = Image.alpha_composite(real, shadow)
    real = Image.alpha_composite(real, ped)
    real = Image.alpha_composite(real, ped)
    final = Image.alpha_composite(real, Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    d2 = ImageDraw.Draw(final)
    for box, radius, fill in blocks:
        rounded_rect(d2, box, radius, fill)
    return Image.alpha_composite(canvas, final)


def extract_tinted_object(path: Path, color):
    src = Image.open(path).convert("RGB")
    # The source uses a black background, so luminance is also our mask.
    gray = src.convert("L")
    mask = gray.point(lambda p: 0 if p < 8 else min(255, int((p - 8) * 1.25)))
    bbox = mask.getbbox()
    if bbox is None:
        raise RuntimeError(f"Could not isolate foreground from {path}")
    src = src.crop(bbox)
    mask = mask.crop(bbox)

    # Colorize while preserving the original shading.
    rgb = src.convert("RGB")
    tinted = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    rp, gp, bp = color
    out_px = []
    for (r, g, b), a in zip(list(rgb.getdata()), list(mask.getdata())):
        lum = max(r, g, b) / 255.0
        if a == 0:
            out_px.append((0, 0, 0, 0))
            continue
        shade = 0.32 + 0.68 * lum
        rr = int(min(255, rp * shade))
        gg = int(min(255, gp * shade))
        bb = int(min(255, bp * shade))
        out_px.append((rr, gg, bb, a))
    tinted.putdata(out_px)
    tinted = ImageEnhance.Contrast(tinted).enhance(1.08)
    return tinted


def add_object(canvas, spec):
    obj = extract_tinted_object(spec["src"], spec["color"])
    x1, y1, x2, y2 = spec["box"]
    tw, th = x2 - x1, y2 - y1
    obj.thumbnail((tw, th))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # Contact shadow from the cutout itself.
    shadow = Image.new("RGBA", obj.size, (0, 0, 0, 0))
    alpha = obj.getchannel("A")
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    shadow = ImageEnhance.Brightness(shadow).enhance(0.42)
    sx = x1 + (tw - obj.size[0]) // 2 + 26
    sy = y2 - obj.size[1] + 34
    layer.alpha_composite(shadow, (sx, sy))

    # Soft ground ellipse.
    ellipse = Image.new("RGBA", (obj.size[0] + 120, 120), (0, 0, 0, 0))
    ed = ImageDraw.Draw(ellipse)
    ed.ellipse((24, 26, ellipse.size[0] - 24, ellipse.size[1] - 18), fill=(116, 126, 146, 70))
    ellipse = ellipse.filter(ImageFilter.GaussianBlur(18))
    ex = x1 + (tw - ellipse.size[0]) // 2
    ey = y2 - 24
    layer.alpha_composite(ellipse, (ex, ey))

    px = x1 + (tw - obj.size[0]) // 2
    py = y2 - obj.size[1]
    layer.alpha_composite(obj, (px, py))
    return Image.alpha_composite(canvas, layer)


def add_finish(canvas):
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    vd.rectangle((0, 0, W, H), fill=(255, 255, 255, 0))
    for i in range(18):
        alpha = int(8 + i * 2.2)
        vd.rounded_rectangle((i * 18, i * 16, W - i * 18, H - i * 12), radius=110, outline=(180, 170, 170, alpha))
    vignette = vignette.filter(ImageFilter.GaussianBlur(42))
    canvas = Image.alpha_composite(canvas, vignette)
    canvas = ImageEnhance.Color(canvas).enhance(1.14)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
    return canvas


def main():
    canvas = build_background().convert("RGBA")
    canvas = draw_ambient_glows(canvas)
    canvas = add_pedestals(canvas)
    for spec in ITEMS:
        canvas = add_object(canvas, spec)
    canvas = add_finish(canvas)
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
