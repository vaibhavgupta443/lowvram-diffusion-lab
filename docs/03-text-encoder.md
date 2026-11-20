# 3. The text encoder ("reader")

Modern image models use a large language model as their text encoder. For Klein 9B that is Qwen3-8B, which is **as big as the diffusion model itself**. On a 7.4 GB machine the reader is a first-class problem, and it has more slack than the diffusion model does.

## 3.1 Prune what the model never reads

FLUX.2 Klein does not use the final output of its Qwen3-8B reader. It takes **hidden states from three layers: 9, 18 and 27**, concatenated into 3 x 4096 = 12288 channels. I verified this in ComfyUI's source (`text_encoders/flux.py`, `llama.py`):

- Tap *i* is the **input** to layer *i* (collected at the start of iteration *i*), un-normalised.
- Tap 27 is therefore the output of layer 26.
- Layers 27-35 (nine of 36), the final norm and the LM head never influence the result.

So I removed them. The reader drops from 36 layers to 27, which is **29% fewer parameters** and 25% less compute and weight streaming. Measured effect on output: a pixel difference of about **0.02** between images made with the pruned and the full reader.

**The trap.** ComfyUI applies a final RMSNorm after the layer loop and, if the last tap index equals the layer count, appends the post-loop activation. With only 27 layers and the norm still on, tap 27 would come out *normalised*, which is wrong. The norm has to be disabled (`final_norm=False`) so the post-loop tap is the raw activation, bit-identical to the original tap.

**Making ComfyUI load it.** The stock loader hard-codes 36 layers and, with `strict=False`, would leave the missing layers uninitialised and still run them. [`nodes/klein_pruned_reader`](../nodes/klein_pruned_reader) wraps `load_clip` and the model constructor: if any text-encoder filename contains `pruned27`, it builds the model with `num_hidden_layers=27, final_norm=False, lm_head=False`. Other files are untouched.

The same analysis for **FLUX.3 Klein** (Qwen3-VL-4B text tower, taps at layers 4, 8, ..., 32) says layers 32-35 and the final norm are wasted (11% of the reader). I wrote the exact truncation recipe, including the same norm trap, but I have not shipped or verified it, so I mark it as untested.

## 3.2 Run the reader on the CPU

Z-Image's reader (Qwen3-4B) uses `hidden_states[-2]`, the input to the last-but-one layer. I run it through **llama.cpp on the CPU** from a Q8_0 GGUF, which keeps the GPU's 4 GB for the diffusion model:

- llama.cpp's layer-input extraction API (`llama_set_embeddings_layer_inp`) returns exactly that tensor, so no final-norm output or per-token logits are computed;
- the prompt is wrapped in the same chat template the reference pipeline uses, with no padding, so the embeddings are meant to match what the reference implementation produces;
- it runs as a small local server that keeps the GGUF memory-mapped, so it adds almost no RAM beyond the page cache.

Measured: **brand-new prompt about 16 s end to end, of which the reader is ~8.5 s**. Standalone the reader takes 3.3 s; inside the pipeline it takes 8.5 s because the 4 GB file is evicted from RAM by everything else. That discrepancy is what convinced me the bottleneck is RAM and not compute (see [01](01-where-the-time-goes.md)). A smaller quantisation (Q5/Q4) is the next thing to try.

## 3.3 Cache the encoded prompt

The same prompt is often sent more than once (retries, changing only the seed or resolution, chained edits). Encoding it again costs a reader load and 10-15 s on this machine, and also evicts the diffusion model from memory.

[`nodes/prompt_embed_cache`](../nodes/prompt_embed_cache) is a drop-in `CLIPTextEncode` with a persistent disk cache:

- Key: `sha256(key + "|" + normalised_text)`. Trailing whitespace is stripped, because my front end appends spaces to prompts to defeat ComfyUI's in-memory cache (sending the *identical* prompt twice in a row triggered an out-of-memory error, which is its own bug).
- The stored object is exactly what `CLIPTextEncode` returned, so **a hit is bit-identical to a miss**. Nothing about quality changes.
- On a hit the reader is never loaded. A repeated job went from **31 s to 16 s**.
- Writes are atomic (write to `.tmp`, then rename) and the oldest files are pruned above ~2 GB.

One safety note: the cache uses `torch.load(..., weights_only=False)` on files the node itself wrote. Do not point it at a directory you did not create.

## Summary

| Technique | Effect | Risk to quality |
|---|---|---|
| Prune unread layers | -29% reader parameters | ~0.02 pixel difference |
| CPU GGUF reader | frees GPU memory | none intended (same template and layer as the reference) |
| Prompt cache | skips the reader on a repeat (31 s → 16 s) | none (bit-identical) |
