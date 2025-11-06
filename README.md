# face-diffusion — Facial Wrinkle Reduction using SDXL Inpainting

> Realistic skin refinements using Stable Diffusion XL Inpainting — preserving identity while reducing wrinkle visibility.

This project explores the use of **diffusion-based masked editing** to simulate subtle cosmetic enhancements (e.g. Botox-like wrinkle reduction) without modifying identity.

No retargeting.  
No generative replacement.  
Only **local inpainting** over precise facial regions.

---

## Example: original vs enhanced outputs

| Original Input | Batch Strength/Seed Variants | Chin Region Comparison |
|---|---|---|
| ![original](https://raw.githubusercontent.com/ju7stritesh/face-diffusion/main/original.jpg) | ![grid](https://raw.githubusercontent.com/ju7stritesh/face-diffusion/main/batch_out/grid.png) | ![chin_grid](https://raw.githubusercontent.com/ju7stritesh/face-diffusion/main/region_out/chin/chin_grid.png) |

---

## What this repo contains

| File | Purpose |
|------|---------|
| `simple_face_inpaint_sdxl.py` | single-shot whole-face wrinkle softening using SDXL Inpaint |
| `batch_wrinkle_search.py` | strength × seed sweeps to generate comparison grids |
| `region_wrinkle_search_sdxl.py` | forehead / under-eye / smile-line / chin selective masking in multi-pass form |

---

## Why SDXL?

### Prior attempts (SD1.5 / SD2.1 inpaint) → problems:

| issue | cause |
|-------|------|
| blue-tinted skin | SD2.1 latent color instability |
| skin "plastic" look | high-strength diffusion with no identity anchor |
| identity drift | non-local generation |

**SDXL fixes this**:

- higher fidelity skin reflectance
- more robust fine texture retention
- larger training corpus

---

## Key architecture reference papers

> this project stands on the shoulders of these architectures

| Component | Paper |
|----------|-------|
| Diffusion Models | Ho et al., *DDPM* (2020) |
| Latent Diffusion | Rombach et al., *LDM* (CVPR 2022) |
| Inpainting Diffusion | Lugmayr et al., *RePaint* (CVPR 2022) |
| SDXL Architecture | Podell et al., *SDXL* (2023) |

---

## Conceptual Pipeline Flow

            ┌────────────────────┐
┌────────────────────┐

input image ───▶│ region mask builder│
└───────┬────────────┘
│ mask (white = editable)
▼
┌──────────────┐
│ SDXL Inpaint │ ← prompt + strength + seed
└───────┬──────┘
│
▼
partial enhancement (per region)
│
▼
combine regions → final composite

---

## strength and seed explained

| parameter | impact |
|----------|--------|
| `strength` | how far from original pixels SDXL is allowed to overwrite. 0.18–0.32 recommended. |
| `seed` | controls micro-level sampling noise → different subtle texture outcomes |

> strength = “how aggressive”
> seed = “which variant of the same idea”

---

## installing environment

onda create -n face-diffusion python=3.10
conda activate face-diffusion

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

---

## How to run basic inpaint

python simple_face_inpaint_sdxl.py

## How to run batch

python batch_wrinkle_search.py

## How to run region selectors (forehead / under-eye / smile / chin)

python region_wrinkle_search_sdxl.py

---

## Future Work

| Direction | Value |
|----------|-------|
| IP-Adapter identity anchors | language + CLIP identity locking |
| per-pixel landmark-aware masks | dynamic masks for non-frontal heads |
| LLM-based aesthetic scoring | automated best-variant selection |
| gradio web UI | non-technical users |
| depth-lattice constraint | bone-structure preservation |

---

## License

For research & non-commercial experimental use.

---

## Conclusion

This repository demonstrates a realistic, practical path to **face enhancement without face replacement**.

Instead of regenerating a new face — we modify *only the wrinkle zones* inside stable diffusion’s native inpainting interface.

The result is a new class of “cosmetic simulation” AI: subtle, structured, region-aware skin refinement.

