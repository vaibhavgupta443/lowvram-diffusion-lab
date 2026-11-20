# 2. Quantisation

Goal: make 5-9B-parameter diffusion transformers small enough to stream through 4 GB of VRAM quickly, without a visible loss in quality. I converted checkpoints myself, layer by layer, using the formats supported by ComfyUI's `comfy-kitchen` kernels.

## What the hardware and software allow

The card is Ampere. That rules out FP4 and MXFP8 (Blackwell-only) and leaves:

| Format | Bytes / weight | Notes |
|---|---|---|
| BF16 | 2.0 | the reference |
| FP8 (e4m3) | 1.0 | **no speed or size benefit over INT8 on Ampere** (no FP8 tensor cores); rejected |
| `int8_tensorwise` (+ ConvRot) | ~1.0 | the format used in official INT8 releases |
| `w6a8_int8` | ~0.81 | 6-bit weights, 8-bit activations |
| `asym_w4a8_int8` | ~0.56 | 4-bit weights, group size 16, per-channel scale, optional codebook |

ConvRot applies a rotation before quantising so that outliers are spread out; it needs the input dimension K to be divisible by the rotation group (256). Klein 9B (K = 4096 / 12288) and Klein 3 (3072 / 9216) satisfy that, so every large linear layer qualifies; for any other model it has to be checked per layer. Layers that do not qualify are written as plain INT8 (about 1% error).

## The streaming converter

The stock converters load the whole checkpoint into RAM. A 9B model in BF16 is ~18 GB, which does not fit in 7.4 GB. My converter changes three things:

1. **The output file is written in place with its header computed up front.** A safetensors header lists every tensor's name, dtype, shape and byte offset, so those can all be calculated from the input before writing any weight data. The converter then handles one layer at a time, so peak RAM is about one layer, and no temporary copy of the output is needed.
2. **A plain `seek`/`read` safetensors reader.** Memory-mapping segfaulted on this machine for large files; ordinary reads did not.
3. **A resumable run** (`--resume-from N`), because a 40-minute conversion on a machine that is also being used will get interrupted.

It also reads **GGUF sources**: it can dequantise a community Q8_0 GGUF of a text encoder one tensor at a time (in row chunks) and write it out as a ComfyUI-format file, renaming llama.cpp tensor names to the ones ComfyUI expects.

I have not published the converter itself. It is derived from a community script whose licence I have not confirmed, so I describe it here instead.

## Which layers stay in BF16

Quantising everything is not the best trade-off. I keep these in BF16:

- the input and output embedders (`img_in`, `txt_in`, `final_layer.*`, `time_in.*`, the final adaptive-norm modulation): tiny, and quality-critical because they are the entry and exit points of the network;
- tensors that ComfyUI reads during model detection.

For Z-Image there is a hard constraint: **`x_embedder.weight` must stay BF16 or the W4A8 kernel segfaults.** I found that by crashing it.

## Mixed precision

The converter accepts a regular expression (`--int8-layers`); matching layers are written as INT8 instead of W4A8. Research on Flux-family transformers consistently finds that sensitivity grows with depth and that attention output projections and MLP down-projections are the most sensitive, while wide early up-projections are the most tolerant. The ConvRot paper reports, for FLUX.1, that keeping about 20% of linear layers at 8 bits recovers nearly all the quality lost by uniform 4-bit.

My result on Klein 9B: a **mixed file with 22% of the weights at INT8** was **1.5-4 s faster** than the all-4-bit file on image creation, with no visible difference on 10 hard prompts and 6 edits. I did not profile why it is faster; one plausible reason is the cost of unpacking 4-bit weights on every step, but that is a guess.

## Baking LoRAs into the weights

A LoRA applied at runtime on top of a 4-bit model has to be re-quantised, and quality degrades. Instead the converter can **merge a LoRA into the full-precision weights before quantising**:

```
W' = W + strength * (alpha / rank) * (up @ down)
```

It accepts both common naming schemes (`lora_unet_*` with `lora_up/down`, and `diffusion_model.*` with `lora_A/B`), and asserts that the delta's shape equals the weight's shape. Baking costs about 40 minutes (the machine is busy converting), but the result runs at normal speed with the LoRA's effect intact.

## How I judged quality (and what I can claim)

- **Mine:** side-by-side contact sheets of the same prompts on the original and the converted file, judged by eye, on a set of hard prompts (counting objects, text on signs, lighting instructions) and a set of edits.
- **Not mine:** published LPIPS/SSIM numbers. For Klein-family models, uploaders report LPIPS vs BF16 of roughly 0.03 for official INT8 and 0.07-0.11 for uniform W4A8 with GPTQ-style calibration. These are self-reported, not independent. My converter is data-free (no calibration), so I expect it to be at the worse end for uniform 4-bit, which is the reason I use a mixed file.
- A community 4-bit file I tried produced visibly bad faces and was rejected. A 1-step distilled variant was also rejected for quality.

Everything here is "good enough that I could not see a difference on my test set", not a measured guarantee.

## Klein 3: W6A8

For the 5.6B Klein 3 model, the official file is INT8 (5.83 GB). A **6-bit (W6A8) conversion at 4.35 GB** looked identical by eye and generated in **20 s instead of 30 s** (reference-image edits ~30 s). That is the best size/quality/speed point I found for that model, and it fits the pattern: when memory is the bottleneck, bytes saved convert almost directly into seconds saved.
