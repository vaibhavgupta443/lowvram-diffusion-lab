"""CLIPTextEncodeCached: CLIPTextEncode with a persistent disk cache of the encoded prompt.
On a cache hit the text encoder (the reader) is never loaded or run, which saves ~10-15 s per job on a RAM-starved machine and keeps the
diffusion model resident. The cache key is sha256(key + normalised text); trailing spaces are ignored (the Studio adds them to defeat ComfyUI's
in-memory cache). The stored object is exactly what CLIPTextEncode returned, so results are identical. Oldest files are pruned above ~2 GB."""
import hashlib
import os

import nodes
import torch

CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "comfy_embed_cache"))  # set EMBED_CACHE_DIR to use a big disk
MAX_BYTES = 2 * 1024 ** 3


def _prune():
    try:
        fs = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith(".pt")]
        tot = sum(os.path.getsize(f) for f in fs)
        for f in sorted(fs, key=os.path.getmtime):
            if tot <= MAX_BYTES:
                break
            tot -= os.path.getsize(f)
            os.remove(f)
    except Exception:
        pass


class CLIPTextEncodeCached:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                             "key": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, key):
        os.makedirs(CACHE_DIR, exist_ok=True)
        h = hashlib.sha256((key + "|" + text.rstrip()).encode("utf-8")).hexdigest()
        path = os.path.join(CACHE_DIR, h + ".pt")
        if os.path.exists(path):
            try:
                cond = torch.load(path, map_location="cpu", weights_only=False)
                os.utime(path)
                return (cond,)
            except Exception:
                pass
        cond = nodes.CLIPTextEncode().encode(clip, text)[0]
        try:
            tmp = path + ".tmp"
            torch.save(cond, tmp)
            os.replace(tmp, path)
            _prune()
        except Exception:
            pass
        return (cond,)


NODE_CLASS_MAPPINGS = {"CLIPTextEncodeCached": CLIPTextEncodeCached}
NODE_DISPLAY_NAME_MAPPINGS = {"CLIPTextEncodeCached": "CLIPTextEncode (disk cached)"}
