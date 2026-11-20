"""CleanCudaCache: pass-through IMAGE node that releases leftover CUDA cache right after VAE decode (keeps loaded models cached)."""
import gc
import torch


class CleanCudaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"
    CATEGORY = "utils"

    def go(self, images):
        gc.collect()
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache(True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (images,)


class CleanCudaCacheCond:
    """Pass-through CONDITIONING: releases leftover CUDA cache after the text reader has run and before the sampler loads the DiT."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"
    CATEGORY = "utils"

    def go(self, conditioning):
        gc.collect()
        try:
            import comfy.model_management as mm
            mm.soft_empty_cache(True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (conditioning,)


class UnloadAllCond:
    """Pass-through CONDITIONING: unloads every model (the previous job's DiT mapping and anything left by the text reader) so the sampler maps a fresh DiT."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "go"
    CATEGORY = "utils"

    def go(self, conditioning):
        try:
            import comfy.model_management as mm
            reader_ran = False
            for lm in list(getattr(mm, "current_loaded_models", [])):
                n = type(getattr(lm, "model", None)).__name__.lower() + " " + type(getattr(getattr(lm, "model", None), "model", None)).__name__.lower()
                if any(t in n for t in ("clip", "qwen", "llama", "textencoder", "te")) and "unet" not in n and "flux" not in n:
                    reader_ran = True
            if not reader_ran:      # prompt came from the cache: the previous job's DiT mapping is still resident -> reset everything (prevents the every-2nd-job crash)
                mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache(True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (conditioning,)


NODE_CLASS_MAPPINGS = {"CleanCudaCache": CleanCudaCache, "CleanCudaCacheCond": CleanCudaCacheCond, "UnloadAllCond": UnloadAllCond}
NODE_DISPLAY_NAME_MAPPINGS = {"CleanCudaCache": "Clean CUDA cache (pass-through)", "CleanCudaCacheCond": "Clean CUDA cache after text (pass-through)"}
