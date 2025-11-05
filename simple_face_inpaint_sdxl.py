#!/usr/bin/env python3
# Minimal SDXL inpaint demo for subtle face improvements (forehead/under-eye smoothing)
# Deps: pip install torch diffusers pillow numpy

import torch
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from diffusers import StableDiffusionXLInpaintPipeline

# ------------ user settings ------------
INPUT_PATH  = "test.png"          # your face photo
OUTPUT_PATH = "simple_out.png"
SEED        = 942
STEPS       = 30
GUIDANCE    = 7.0
STRENGTH    = 0.28   # 0.18–0.30 is subtle; raise slowly if too weak

PROMPT = (
    "photorealistic portrait, natural skin texture, subtle reduction of forehead lines "
    "and crow's feet, slightly smoother under-eye area, even skin tone, preserve identity"
)
NEG_PROMPT = "over-smoothed, waxy, plastic skin, blur, cartoon, changed identity, color shift"

SIZE = 1024  # SDXL inpaint works well at 1024x1024
# ---------------------------------------

def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)

def make_simple_upper_face_mask(size: int) -> Image.Image:
    """
    White = edit area. Soft band covering forehead + under-eye.
    No landmarks—just a conservative region in the top ~62% of the face.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    y0 = int(size * 0.18)   # top of band
    y1 = int(size * 0.62)   # bottom of band
    draw.rectangle([0, y0, size, y1], fill=255)
    # feather edges to avoid seams
    mask = mask.filter(ImageFilter.GaussianBlur(radius=24))
    return mask

def main():
    # load & prep image
    img = Image.open(INPUT_PATH).convert("RGB")
    img = center_square_resize(img, SIZE)

    # simple soft mask over upper face
    mask = make_simple_upper_face_mask(SIZE)

    # build SDXL inpaint pipe
    use_fp16 = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    device = "cuda" if use_fp16 else "cpu"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch_dtype,
        use_safetensors=True,
        safety_checker=None,
    ).to(device)

    gen = torch.Generator(device=device).manual_seed(SEED)

    # run
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=img,
        mask_image=mask,        # white = edit
        strength=STRENGTH,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=gen,
    )
    result = out.images[0]
    result.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
