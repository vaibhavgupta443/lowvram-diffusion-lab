# 5. Editing and image-to-image quality

Speed is only worth having if the result is still right. Most of my time on edit models went into *fidelity*: edits that preserve identity, colours and counts, and chains of edits that do not degrade.

## 5.1 Bug: chained edits burned out

**Symptom.** Feeding an edit's output back in as the next edit's input, five times, produced growing contrast, saturation, noise and blocky colour patches (worse on the 9B model than on 4B).

**Diagnosis.** The edit graph decoded with `VAEDecodeTiled` (tile size 512). Tiled decoding is an approximation at tile borders; re-encoding and re-decoding the approximated image compounds the error on every pass.

**Fix.** Use a whole-image `VAEDecode` by default and keep tiling as an opt-in. ComfyUI falls back to tiled decoding on its own if the full decode runs out of memory, so nothing is lost.

**Result.** Five chained edits stayed clean, and each edit got faster: **21 s instead of 32 s** per 0.6 MP edit. A quality fix that was also a speed fix.

## 5.2 Real photo to anime: from prompt tuning to a different method

**The problem.** Asking an edit model to "redraw this photo as anime" drifted: eye colour changed, saturation shifted, ages drifted, and objects were invented. Most of the drift happened on difficult photos (several people, small accessories).

**What failed.**
- Negative phrasing such as "do not add glasses" tends to *add* glasses; naming an item at all puts it in the model's head. Black Forest Labs' own guidance says the same: no negative prompts, state what to keep.
- Listing wording like "large expressive eyes" opened closed eyes.
- Other ways of injecting the reference image (offsets, alternative reference methods) did not help.

**What worked.** A different graph rather than a better prompt:

1. Encode the original photo to a latent.
2. Run a **partial denoise** from that latent (not from pure noise), so composition and colours are inherited from the photo. The final settings were denoise 0.8 over 5 scheduled steps (4 effective).
3. Also pass **the same photo as a reference latent**, so the model sees identity while redrawing the style. A smaller (0.4 MP) copy of the reference cut ~20% of the time with no visible loss.
4. A short positive-only style prompt: the style, then "keep the same people, exactly as many, the same faces and apparent age, same eye and hair colour".
5. Run at 1.0 MP for this mode (fine details like a pen in a pocket vanish at lower resolution).

Without the reference latent the partial-denoise version was unusable.

## 5.3 Colour lock (post-process)

Even with the method above, the palette can drift. [`tools/lab_colour_lock.py`](../tools/lab_colour_lock.py) is a small, dependency-light post-process:

- convert both images to **CIE Lab** (separates lightness from colour);
- blur both to a coarse grid (about 20 cells along the long edge);
- move the generated image's **chroma** (a, b) toward the original's, cell by cell; move its **lightness** only a little (tone 0.35-0.6) so line-work and shading stay as drawn.

Because only low-frequency colour is corrected, outlines and detail are untouched. On top of that I built a face-aware step that estimates each person's skin colour in the photo and in the drawing (faces found with a small ONNX face detector) and corrects each person separately, so it works across skin tones. It must stay gentle: an over-strong correction turned a fair-skinned subject brown, and I eventually switched the skin step off by default and kept only the coarse colour lock. That part depends on private paths, so it is described here, not published.

## 5.4 Step count

For the distilled Klein 3 model, the official range is 4-16 steps. In my edits **4 steps gave bad edits, so I used 8**. BFL's reference code shows the distilled model is a *flow map*: it takes the **next** sigma as an extra input, so any step count is legal input, but quality at a given count depends on which jump sizes were trained, which is unpublished. Another detail that is easy to miss: the official workflow overrides the sampling shift to ln(16) (`ModelSamplingFlux` with max/base shift 2.77259); without that node the schedule differs from the reference. I flagged this as a thing to verify in my own workflow rather than something I have confirmed.

## 5.5 Edit prompting rules that held up

From the vendor's editing guide and my own tests:

- Instruction style, not a full caption: "replace only her top with the sweater from image 2; keep her face, hair and the grey wall unchanged".
- Name the target uniquely (position, colour, neighbours) and state old value → new value.
- Give each reference image a role by position ("image 1 is the base, image 2 is the outfit").
- Concrete wording works; vague wording ("make it better", "soft lighting") rewrites identity.
- Do 2-3 small edits rather than one large one.
- Keep working resolution around 1 MP on a 4 GB card.

## 5.6 Reference-image KV caching (researched, not adopted)

Klein 9B has a KV-cache variant: the reference tokens' keys and values are computed once at step 0 and reused for later steps, so the reference no longer costs more as steps increase (vendor-reported speedups of roughly 1.2x-2.7x depending on resolution and number of references). I read the implementation: ComfyUI's node depends on the reference tokens getting timestep-0 modulation, which suggests the KV model is fine-tuned for it, and community reports mention weaker identity consistency. It is not part of the pipeline described here.
