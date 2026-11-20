# 4. Memory and offloading

A diffusion transformer bigger than VRAM has to *stream*: some weights stay on the GPU, the rest are copied in during every sampling step. How that is organised decides the speed more than the model does.

## 4.1 How ComfyUI's dynamic VRAM works (from reading the source)

Recent ComfyUI builds use a dynamic-VRAM patcher (`ModelPatcherDynamic`, backed by the `comfy-aimdo` library):

- `load()` does **not** copy weights. Each Linear layer gets a slice of a per-model **virtual address range** (32 MB pages) and the weights stay in the memory-mapped file.
- Residency follows a priority rule: a newer model's range has higher priority; within a range, lower addresses win. Faulting in a higher-priority weight evicts lower-priority pages. When something is evicted, a *watermark* is lowered and every weight above it fails to fault in until the next `prioritize()` call.
- In effect it already does "the first N weights stay resident, the rest stream every step", where N is whatever fits under the VRAM headroom limit. On this card that is small: about 1.8-1.9 GB for a 4B model, around 3 GB for Klein INT8.
- Per step, a prefetch queue copies block *i+1* on a second CUDA stream while block *i* computes (two async streams by default).

This matters because it means the standard approach is already close to what a hand-written offloader would do, so my wins came from the settings around it and from removing leaks, not from replacing it.

I also looked at community alternatives (static-placement nodes, model-specific block-swap nodes) and chose not to install any: they run arbitrary Python inside the ComfyUI process, are tied to other model classes, or are being archived.

## 4.2 Partial residency and the VAE reserve

For Z-Image I used the Nunchaku pipeline with **partial residency**: part of the quantised transformer stays on the GPU, the rest streams. The important knob is how much VRAM to *reserve* for the VAE decode at the end:

| Reserve | VAE decode time |
|---|---|
| 700 MiB | 3.4 s |
| 1100 MiB | **0.6 s** |

Reserving too little forces the VAE to work in small pieces or spill. A 3 s difference on a 7 s job is large.

## 4.3 Launch flags, measured

I compared flag sets on repeated jobs (quiet machine, median):

| Flags | Median |
|---|---|
| defaults | 22.5 s |
| `--cache-lru 8 --fast --disable-pinned-memory` | **20.8 s** |
| adding `--high-ram` | slower |
| adding `--fast-disk` | slower |

`--cache-lru 8` caches the results of up to eight nodes, and pinned memory takes RAM that is scarce here, so disabling it plausibly helped (I did not isolate each flag's contribution). These results are specific to a 7.4 GB machine and would likely reverse on a machine with more RAM.

## 4.4 The "every second job crashes" bug

Symptom: jobs alternated between succeeding and failing with `CUDA out of memory (cuMemSetAccess)`, and each failure also broke the allocator until ComfyUI was restarted.

Diagnosis: after a 1-megapixel VAE decode, the CUDA caching allocator was holding leftover blocks that collided with the dynamic-VRAM allocator on the next job's model mapping.

Options I measured:

| Fix | Result |
|---|---|
| `POST /free` between jobs | works, but reloads the model: +4 s per job |
| `--disable-dynamic-vram` | works, but **62 s** per job |
| **A pass-through node that empties the CUDA cache at the right point** | works, models stay loaded |

[`nodes/clean_cuda`](../nodes/clean_cuda) provides three small nodes:

- `CleanCudaCache`: placed right after the VAE decode; collects garbage and empties the CUDA cache without unloading anything.
- `CleanCudaCacheCond`: the same, after the text encoder has run and before the sampler maps the diffusion model.
- `UnloadAllCond`: if the prompt came from the cache (so the reader did *not* run), the previous job's diffusion-model mapping is still resident, which is the crashing case; this node unloads every model so the sampler maps a fresh one. If the reader did run, it does nothing extra.

After the fix, Real-to-Anime jobs went from **~50 s with a crash every second job** to a **steady 22.7 s with 0 crashes in 8 consecutive jobs**. That is a small sample; I report it as what I observed, not as a guarantee.

## 4.5 Things that waste RAM and are easy to miss

- An **idle ComfyUI keeps ~11 GB of committed memory** and starves a conversion running next to it. Stop it during conversions.
- Windows memory compression takes ~2.4 GB; the free pool is under 1 GB. Browser tabs matter.
- After a CUDA out-of-memory error the allocator may stay broken; restart rather than retry.
- The second drive matters: models are on a different drive from the OS because the system drive has only a few GB free.

## 4.6 Open questions

For Klein 3 INT8 I measured about **2.9 s per step** and could not tell whether the cause is pinned-memory starvation or compute. I wrote a diagnostic plan (a per-step log of watermark, pinned MB, available RAM and free VRAM, plus a 256x256 control run), but have not run it. Treat that part as unresolved.
