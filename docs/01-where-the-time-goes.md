# 1. Where the time goes

Before optimising anything I wanted to know what the bottleneck actually was, because the obvious guess (a slow GPU) turned out to be wrong.

## The machine

| | |
|---|---|
| CPU | Ryzen 7 4800H (8 cores) |
| GPU | RTX 3050 laptop, **4 GB VRAM**, Ampere (no FP4 / MXFP8 hardware) |
| RAM | **7.4 GB**, of which Windows memory compression already holds ~2.4 GB; under 1 GB is usually free |
| Disk | A small system drive (a few GB free), so models live on a second drive |

## The arithmetic that explains everything

Take the setup I ended up using for FLUX.2 Klein 9B:

| Component | Size on disk |
|---|---|
| Diffusion transformer, 4-bit (W4A8) | 5.24 GB |
| Text encoder (Qwen3-8B), pruned + 4-bit | 4.18 GB |
| VAE, activations, CUDA context | ~1 GB |
| **Total working set** | **~9-10 GB** |

That is 2.3x the VRAM and more than the total RAM. So **every generation re-reads a large part of ~9 GB of weights from disk**, and most of the wall-clock time is data movement, not arithmetic. Two observations support this:

- The text encoder ran in **3.3 s** on its own, but **8.5 s** inside the full pipeline. The only difference was that, in the pipeline, the OS had evicted its 4 GB file from RAM to make room for everything else.
- Closing a browser with a few tabs visibly changed generation times. Background apps compete for the same ~1 GB of free RAM.

On a machine like this, a good optimisation is one that **avoids loading, copying or keeping something**, not one that makes a kernel faster.

## What this implies

1. Make the weights smaller (quantisation), so there is less to move. → [02](02-quantisation.md)
2. Do not load weights that the model never uses (pruning the text encoder). → [03](03-text-encoder.md)
3. Do not load the same thing twice (prompt caching, keeping the diffusion model resident). → [03](03-text-encoder.md)
4. Control what stays on the GPU and what streams. → [04](04-memory-and-offloading.md)
5. Avoid pipeline choices that cost time *and* quality. → [05](05-editing-quality.md)

## Where the time goes in one job (Klein 9B, 0.7 MP, measured)

Roughly 8 s of fixed overhead (model residency, text encoding, VAE, post-processing) plus about 5 s per sampling step. With 4 steps that is the 21-26 s I report. The fixed part is where the memory work pays off: a cached prompt skips the text encoder, and a resident model skips the reload.

## A note on measurement

I timed jobs on a quiet machine (browsers and chat apps closed), repeated them, and looked at medians. A single timing on this hardware is nearly meaningless: one OS memory-pressure event adds 10 s. Where I give a number, it is a median or a clearly labelled single observation.
