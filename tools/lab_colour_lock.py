"""Lab colour lock: keep an edited image's luminance and line-work, pull its chroma (and a little tone) back to the original photo's colours at a coarse spatial scale.
Usage: python lab_colour_lock.py OUT.png SRC.png [DST.png]   (OUT is the generated image, SRC the original photo)"""
import numpy as np
from PIL import Image

def _srgb2lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def _lab2srgb(lab):
    fy = (lab[..., 0] + 16) / 116; fx = fy + lab[..., 1] / 500; fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Minv = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]))
    c = np.clip(xyz @ Minv.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)

def _low(x, grid):
    """coarse low-pass: box-average to grid cells, bilinear back up (x: H,W,C float32)"""
    H, W, C = x.shape
    gh, gw = max(2, round(grid * H / max(H, W))), max(2, round(grid * W / max(H, W)))
    chans = []
    for k in range(C):
        im = Image.fromarray(x[..., k].astype(np.float32), mode="F")
        chans.append(np.asarray(im.resize((gw, gh), Image.BOX).resize((W, H), Image.BILINEAR), dtype=np.float32))
    return np.stack(chans, -1)

def colour_lock(out_path, src_path, dst_path=None, grid=20, chroma=1.0, tone=0.35):
    out = Image.open(out_path).convert("RGB"); src = Image.open(src_path).convert("RGB").resize(out.size, Image.BOX)
    lo, ls = _srgb2lab(np.asarray(out, dtype=np.float32)), _srgb2lab(np.asarray(src, dtype=np.float32))
    res = lo.copy()
    d_ab = _low(ls[..., 1:], grid) - _low(lo[..., 1:], grid)
    res[..., 1:] = lo[..., 1:] + chroma * d_ab
    d_l = _low(ls[..., :1], grid) - _low(lo[..., :1], grid)
    res[..., :1] = lo[..., :1] + tone * d_l
    Image.fromarray(_lab2srgb(res).astype(np.uint8)).save(dst_path or out_path)




if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    colour_lock(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
