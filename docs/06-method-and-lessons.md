# 6. Method, dead ends and lessons

## How I compared things

- **Same prompts, side by side, one change at a time.** Contact sheets with the old and new result next to each other, judged by eye. I made a 60-prompt suite and ran both Z-Image and Klein 9B on all 60: Z-Image averaged **17.5 s**, Klein 9B **28.4 s**. Klein 9B was clearly better at following instructions (object counts, stage-lighting instructions); Z-Image was a little crisper and faster. That is why Z-Image stays the plain-generation default and Klein 9B is the editing and instruction-following model.
- **Timings on a quiet machine, repeated, median.** Apps in the background change results by whole seconds. I labelled single observations as such.
- **Quality-first rules.** I set a rule up front: no speed optimisation that visibly costs quality. So every speed idea had to be *exact* (same output) or pass a side-by-side check. The prompt cache is exactly output-neutral; the reader pruning changes pixels by about 0.02; everything else was checked by eye.
- **A hard-prompt set, not an easy one.** Counting, text on signs, specific lighting, multi-person scenes, and edits that must keep identity. Easy prompts hide quantisation damage.

## What did not work (and I stopped)

| Idea | Why it was dropped |
|---|---|
| A community 4-bit (INT4) Klein file | Visibly bad faces |
| A 1-step distilled model | Bad quality |
| 4-step sampling for Klein 3 edits | Bad edits; 8 steps needed |
| A lower-bit text-encoder quantisation (Q3_K_M) | Crashed ComfyUI |
| Alternative reference-injection methods for edits | Did not control leakage; concrete wording did |
| Prompt-only tuning for photo-to-anime | Drifted on hard photos; a different graph worked (see [05](05-editing-quality.md)) |
| Strong face-skin correction | Shifted skin tone the wrong way; kept gentle, then turned off |
| Tiled VAE decode for edits | Compounding error on chained edits |
| A bigger edit model that scored better | Too large for 7.4 GB of RAM |

## Lessons I would give someone starting this

1. **Find the bottleneck first.** I assumed GPU speed. It was RAM and disk. The 3.3 s versus 8.5 s text-encoder timing was the single most useful measurement.
2. **Read the code path before pruning.** "These layers are unused" is only true if you check which activations are tapped and whether a final norm is applied afterwards. The bug I almost shipped was a normalised tap.
3. **Prefer changes that are provably output-neutral.** A disk cache of encoded prompts changes nothing about the image. Start there.
4. **Leaks look like speed problems.** A job that is "slow every second time" can be an allocator problem; fixing it gave a larger gain than any kernel change.
5. **Quality bugs hide inside speed settings.** The tiled VAE decode was chosen for memory and silently degraded chained edits.
6. **Keep a log of what you measured and under what conditions.** I could write these notes only because I recorded flags, file sizes and timings as I went.
7. **Be honest about confidence.** In my notes every claim is tagged: measured by me, read from source, reported by a third party, or my guess. This write-up keeps that distinction.

## Pitfalls (cost me hours)

- A shell here-document turned a Windows path `\a` into a BEL control character inside a generated Python file. Write generated files with an editor tool and forward slashes.
- A long-running log file opened with `tail -F` locks it for the process that is appending to it.
- Downloading "the latest version" from a model site can silently give you an FP8 variant. Download by explicit file ID, and verify sizes and hashes.
- Download CDNs throttle per connection; parallel range requests restore throughput.
- Always back up a working script before changing it (`*.bak_before_x`), so that a bad change is one copy away from undone.

## What I would do next

- Run the per-step diagnostic for Klein 3 (pinned memory vs compute).
- Try a smaller text-encoder quantisation (Q5/Q4) to stop the 4 GB encoder file being evicted.
- Try calibrated (GPTQ-style) 4-bit conversion for the layers that are most sensitive, and measure LPIPS instead of only judging by eye.
- Test the truncated FLUX.3 reader and confirm it is bit-identical.
