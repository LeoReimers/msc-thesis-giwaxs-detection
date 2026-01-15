# -*- coding: utf-8 -*-
import numpy as np

def _to_uint8(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)

    # GIWAXS: oft sehr sparse ? Quantile auf Nicht-Null-Werten stabiler
    x = np.clip(x, 0.0, None)
    nz = x[x > 0]

    if nz.size > 100:
        lo = float(np.quantile(nz, 0.01))
        hi = float(np.quantile(nz, 0.99))
    else:
        lo = float(np.min(x))
        hi = float(np.max(x))

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo) < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + eps)
    return (255.0 * x).round().astype(np.uint8)


def clahe_2d_numpy(img_hw: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    """
    img_hw: (H,W) numpy float or uint8
    return: (H,W) float32 in [0,1]
    """
    img_u8 = _to_uint8(img_hw)

    import cv2
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1]))
    )
    out_u8 = clahe.apply(img_u8)
    return out_u8.astype(np.float32) / 255.0
