"""Lets ComfyUI load a *pruned* Qwen3-8B reader for FLUX.2 Klein 9B.
Klein only reads hidden states at layers 9/18/27, i.e. the output of layer 26, so layers 27-35, the final norm and lm_head are unused.
When any text-encoder file name contains 'pruned27', the 8B model is built with 27 layers, no final norm and no lm_head.
Other (full) 8B files are untouched."""
import comfy.sd
import comfy.text_encoders.llama as _llama

_state = {"prune": False}
_orig_load_clip = comfy.sd.load_clip
_orig_init = _llama.Qwen3_8B.__init__


def _load_clip(ckpt_paths, *a, **k):
    _state["prune"] = any("pruned27" in str(p) for p in (ckpt_paths or []))
    try:
        return _orig_load_clip(ckpt_paths, *a, **k)
    finally:
        _state["prune"] = False


def _init(self, config_dict, dtype, device, operations):
    if _state["prune"]:
        config_dict = {**config_dict, "num_hidden_layers": 27, "final_norm": False, "lm_head": False}
    _orig_init(self, config_dict, dtype, device, operations)


comfy.sd.load_clip = _load_clip
_llama.Qwen3_8B.__init__ = _init
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
