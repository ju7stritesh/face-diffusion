#!/usr/bin/env python3
# Try multiple (strength, seed) combos for SDXL inpainting wrinkle softening
# Deps: pip install torch diffusers pillow numpy

import os, math, itertools, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline

# ------------ user settings ------------
INPUT_PATH  = "test.png"        # your face image
OUT_DIR     = "batch_out"       # results folder
SIZE        = 1024              # SDXL works well at 1024
STEPS       = 34
GUIDANCE    = 7.0

PROMPT = (
    "photorealistic portrait, natural skin texture, subtle reduction of forehead lines "
    "and crow's feet, slightly smoother under-eye area, even skin tone, preserve identity"
)
NEG_PROMPT = "over-smoothed, plastic, waxy, blur, cartoon, changed identity, color shift"

# Try these combos (edit freely)
STRENGTHS = [0.20, 0.24, 0.28]
SEEDS     = [42, 142, 242]

# Set to True to run 3 gentle passes (forehead → under-eye → smile lines) per combo.
# Set to False to run a single pass (upper-face band).
THREE_PASSES = True
# ---------------------------------------

def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    l, t = (w - s)//2, (h - s)//2
    return img.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)

def blur(mask: Image.Image, r: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius=r))

def mask_upper_band(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    y0, y1 = int(size*0.18), int(size*0.62)
    ImageDraw.Draw(m).rectangle([0, y0, size, y1], fill=255)
    return blur(m, 24)

def mask_forehead(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    y0, y1 = int(size*0.16), int(size*0.42)
    ImageDraw.Draw(m).rectangle([0, y0, size, y1], fill=255)
    return blur(m, 26)

def mask_undereye(size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    cx, cy = size//2, int(size*0.42)
    # left
    d.ellipse([cx-int(size*0.42), cy, cx-int(size*0.08), cy+int(size*0.22)], fill=255)
    # right
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

def inpaint(pipe, img: Image.Image, mask: Image.Image, strength: float, seed: int) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=img,
        mask_image=mask,     # white = edit
        strength=strength,   # 0.18–0.30 is subtle
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=gen,
    ).images[0]
    return out

def run_combo(pipe, base_img: Image.Image, strength: float, seed: int) -> Image.Image:
    img = base_img
    if THREE_PASSES:
        img = inpaint(pipe, img, mask_forehead(SIZE), strength=min(strength, 0.26), seed=seed)
        img = inpaint(pipe, img, mask_undereye(SIZE), strength=min(strength, 0.24), seed=seed+1)
        img = inpaint(pipe, img, mask_smilelines(SIZE), strength=min(strength, 0.22), seed=seed+2)
    else:
        img = inpaint(pipe, img, mask_upper_band(SIZE), strength=strength, seed=seed)
    return img

def build_grid(images, cols):
    if not images: return None
    w, h = images[0].size
    rows = math.ceil(len(images)/cols)
    grid = Image.new("RGB", (cols*w, rows*h), (0,0,0))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c*w, r*h))
    return grid

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load & prep
    base = Image.open(INPUT_PATH).convert("RGB")
    base = center_square_resize(base, SIZE)

    # Pipe (SDXL inpaint)
    use_fp16 = torch.cuda.is_available()
    dtype = torch.float16 if use_fp16 else torch.float32
    device = "cuda" if use_fp16 else "cpu"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype, use_safetensors=True, safety_checker=None
    ).to(device)

    # Try all combos
    results = []
    labels  = []
    for strength, seed in itertools.product(STRENGTHS, SEEDS):
        img = run_combo(pipe, base, strength, seed)
        name = f"s{strength:.2f}_seed{seed}.png"
        img.save(os.path.join(OUT_DIR, name))
        results.append(img)
        labels.append(name)
        print("saved", name)

    # Build a labeled grid preview
    cols = len(SEEDS)
    grid = build_grid(results, cols=cols)
    if grid:
        # add labels as a header bar (simple)
        grid_path = os.path.join(OUT_DIR, "grid.png")
        grid.save(grid_path)
        print("grid saved:", grid_path)
        print("order left→right, top→bottom:")
        for i, lab in enumerate(labels):
            print(f"{i+1:02d}: {lab}")

if __name__ == "__main__":
    main()
