#!/usr/bin/env python3
# Region-by-region SDXL inpainting wrinkle softening with grids (adds CHIN).
# Deps: pip install torch diffusers pillow numpy

import os, math, itertools, torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline

# ---------- user settings ----------
INPUT_PATH  = "test.png"
OUT_DIR     = "region_out"
SIZE        = 1024

# try these strength/seed combos
STRENGTHS = [0.20, 0.54, 0.88]
SEEDS     = [42, 542, 1242]

# which regions to test (each item is a list to allow sequenced passes)
REGION_SETS = [
    ["forehead"],
    ["undereye"],
    ["smilelines"],
    ["chin"],               # NEW
    ["forehead", "undereye"],
    ["forehead", "undereye", "smilelines", "chin"],  # all
]

STEPS    = 34
GUIDANCE = 7.0

PROMPT = (
    "photorealistic portrait, natural skin texture, subtle reduction of facial wrinkles, "
    "even skin tone, preserve identity and lighting"
)
NEG_PROMPT = "over-smoothed, plastic, waxy, blur, cartoon, color shift, changed identity"
# -----------------------------------

# ---------------- helpers ----------------
def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    l, t = (w - s)//2, (h - s)//2
    return img.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)

def blur(mask: Image.Image, r: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius=r))

# ---- region masks (coarse but effective for centered portrait) ----
def mask_forehead(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    y0, y1 = int(size*0.16), int(size*0.42)
    ImageDraw.Draw(m).rectangle([0, y0, size, y1], fill=255)
    return blur(m, 26)

def mask_undereye(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    cx, cy = size//2, int(size*0.42)
    # left bag
    d.ellipse([cx-int(size*0.42), cy, cx-int(size*0.08), cy+int(size*0.22)], fill=255)
    # right bag
    d.ellipse([cx+int(size*0.08), cy, cx+int(size*0.42), cy+int(size*0.22)], fill=255)
    return blur(m, 20)

def mask_smilelines(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    cx, cy = size//2, int(size*0.62)
    w, h = int(size*0.18), int(size*0.24)
    d.ellipse([cx-int(size*0.28)-w//2, cy-h//2, cx-int(size*0.28)+w//2, cy+h//2], fill=255)
    d.ellipse([cx+int(size*0.28)-w//2, cy-h//2, cx+int(size*0.28)+w//2, cy+h//2], fill=255)
    return blur(m, 22)

def mask_chin(size: int) -> Image.Image:
    # Soft oval around chin area below lower lip
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    cx, cy = size//2, int(size*0.78)
    w, h = int(size*0.42), int(size*0.22)
    d.ellipse([cx-w//2, cy-h//2, cx+w//2, cy+h//2], fill=255)
    # trim slightly upward so it doesn’t climb into lips
    trim = Image.new("L", (size, size), 0)
    ImageDraw.Draw(trim).rectangle([0, int(size*0.70), size, size], fill=255)
    m = Image.composite(m, Image.new("L", (size, size), 0), trim)
    return blur(m, 24)

MASK_FUNCS = {
    "forehead":   mask_forehead,
    "undereye":   mask_undereye,
    "smilelines": mask_smilelines,
    "chin":       mask_chin,
}

# per-region gentle strengths (multiplier)
REGION_STRENGTH_CAP = {
    "forehead":   0.26,
    "undereye":   0.24,
    "smilelines": 0.24,
    "chin":       0.22,
}

def inpaint(pipe, img: Image.Image, mask: Image.Image, strength: float, seed: int) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=img,
        mask_image=mask,     # white = edit
        strength=strength,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=gen,
    ).images[0]
    return out

def run_region_set(pipe, base_img: Image.Image, regions: list, strength: float, seed: int) -> Image.Image:
    img = base_img
    s = strength
    for i, r in enumerate(regions):
        cap = REGION_STRENGTH_CAP.get(r, strength)
        step_strength = min(s, cap)
        img = inpaint(pipe, img, MASK_FUNCS[r](SIZE), step_strength, seed + i)
    return img

def build_grid(images, cols):
    if not images: return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols*w, rows*h), (0,0,0))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c*w, r*h))
    return grid
# ---------------- end helpers ----------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = Image.open(INPUT_PATH).convert("RGB")
    base = center_square_resize(base, SIZE)

    # SDXL inpaint pipeline
    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    device = "cuda" if use_fp16 else "cpu"
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype, use_safetensors=True, safety_checker=None
    ).to(device)

    # iterate region sets
    for regions in REGION_SETS:
        tag = "+".join(regions)
        region_dir = os.path.join(OUT_DIR, tag)
        os.makedirs(region_dir, exist_ok=True)

        imgs = []
        labels = []
        for strength, seed in itertools.product(STRENGTHS, SEEDS):
            result = run_region_set(pipe, base, regions, strength, seed)
            name = f"{tag}_s{strength:.2f}_seed{seed}.png"
            result.save(os.path.join(region_dir, name))
            imgs.append(result); labels.append(name)
            print("saved", os.path.join(tag, name))

        # grid per region-set
        grid = build_grid(imgs, cols=len(SEEDS))
        if grid:
            grid.save(os.path.join(region_dir, f"{tag}_grid.png"))
            print("grid saved:", os.path.join(tag, f"{tag}_grid.png"))
            print("order left→right, top→bottom:")
            for i, lab in enumerate(labels):
                print(f"{i+1:02d}: {lab}")

if __name__ == "__main__":
    main()
