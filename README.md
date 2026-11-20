# Running modern image models on a 4 GB GPU and 7.4 GB of RAM

Engineering notes, measurements and small tools from getting current diffusion models (Z-Image, FLUX.2 Klein 4B/9B, FLUX.3 Klein, SDXL) to **generate and edit images in roughly 8-30 seconds on a laptop that officially cannot run them**.

**Machine:** Ryzen 7 4800H, RTX 3050 laptop (4 GB VRAM, Ampere), 7.4 GB RAM, Windows 11. All numbers are measured on this one machine unless a line says otherwise.

> **The finding that shaped everything:** the bottleneck is **RAM and disk, not the GPU**. The working set is 9-10 GB against 7.4 GB of RAM, so each job re-reads weights from disk. Good optimisations here *avoid loading or copying* things; faster kernels barely matter.

## Results

| What | Time (512x768 unless noted) | How |
|---|---|---|
| Z-Image, repeated prompt | **7.6 s** | W4A8 weights, partial residency, prompt cache |
| Z-Image, new prompt | **~16 s** | text encoder on CPU via llama.cpp (~8.5 s of the total) |
| Klein 9B, generate | 21-26 s | W4A8 (and a mixed-precision file, 1.5-4 s faster), pruned reader |
| Klein 9B, edit with reference | ~21 s at 0.6 MP | whole-image VAE decode (was 32 s) |
| Klein 3, generate | 20 s (was 30 s) | W6A8 file, 4.35 GB vs official 5.83 GB INT8 |
| Photo to anime, 9B, 0.7 MP | **22.7 s, steady** | was ~50 s with a crash every second job |
| SDXL, 4-step distilled | 17-19 s | fp8 UNet, tiled VAE |

A 60-prompt comparison on identical prompts: Z-Image **17.5 s** mean, Klein 9B **28.4 s**; Klein 9B followed instructions better (counts, lighting), Z-Image was slightly crisper.

## What is in here

| Doc | Topic |
|---|---|
| [01 Where the time goes](docs/01-where-the-time-goes.md) | the memory arithmetic and how I found the real bottleneck |
| [02 Quantisation](docs/02-quantisation.md) | a streaming W4A8 converter, mixed precision, baking LoRAs in, what stays BF16 |
| [03 The text encoder](docs/03-text-encoder.md) | pruning unread layers, a CPU GGUF reader, a bit-identical prompt cache |
| [04 Memory and offloading](docs/04-memory-and-offloading.md) | dynamic VRAM internals, VAE reserve, flags, the CUDA-cache crash and its fix |
| [05 Editing quality](docs/05-editing-quality.md) | a VAE bug that ruined chained edits, a better photo-to-anime method, colour lock |
| [06 Method and lessons](docs/06-method-and-lessons.md) | how I compared things, dead ends, pitfalls |

| Code | Purpose |
|---|---|
| [`nodes/clean_cuda`](nodes/clean_cuda) | ComfyUI nodes that release the CUDA cache at chosen points without unloading models |
| [`nodes/prompt_embed_cache`](nodes/prompt_embed_cache) | `CLIPTextEncode` with a persistent, bit-identical disk cache |
| [`nodes/klein_pruned_reader`](nodes/klein_pruned_reader) | loads a layer-pruned Qwen3-8B reader for FLUX.2 Klein 9B |
| [`tools/lab_colour_lock.py`](tools/lab_colour_lock.py) | coarse Lab-space colour transfer to stop palette drift after an edit |

## Three results worth highlighting

1. **A reader that was 29% dead weight.** Klein 9B only reads layers 9/18/27 of its Qwen3-8B text encoder. Layers 27-35, the final norm and the LM head can go (a pixel difference of ~0.02), but only if the final norm is also disabled; otherwise the last tap comes out normalised and wrong.
2. **A quality bug that was also a speed bug.** Tiled VAE decoding made chained edits burn out. A whole-image decode fixed the drift and cut 32 s to 21 s.
3. **A crash every second job**, traced to the CUDA allocator colliding with dynamic VRAM. `/free` fixed it for +4 s, `--disable-dynamic-vram` for 62 s; a pass-through node that clears the cache at the right point fixed it for free.

## Using the nodes

Copy a folder into `ComfyUI/custom_nodes/` and restart ComfyUI. They were written against a recent ComfyUI; `klein_pruned_reader` patches internal classes and may need updating when those change. The prompt cache directory is set with the `EMBED_CACHE_DIR` environment variable.

## Honesty notes

- Numbers are one laptop. Quality was judged by eye on side-by-side sheets; I did not compute image-quality metrics, and published LPIPS figures I cite are third-party and self-reported.
- Things I researched but did not verify or adopt are labelled as such in the docs (for example truncating the FLUX.3 reader and the KV-cache edit variant).
- The quantisation converter is not published: it derives from a community script whose licence I have not confirmed. Doc 02 describes what it does.
- The model weights, the web front end I built around ComfyUI, and any generated images are not included.
