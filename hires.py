"""High-resolution terrain export.

The simulation runs once, at preview size. Export upscales that world. Widths
up to 8K are one image. Wider maps are tiles, with a little seeded height
detail so the extra pixels are not a flat blur. 256K is 262144 pixels wide.
"""
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from terrain.export import METRES_PER_UNIT, _write_png

PREVIEW_WIDTH = 640
DIRECT_MAX = 8192
TILE = 4096
GAME_MAX = 2048
EXPORT_WIDTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
EXPORT_LABELS = ("1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K")

# Added on top of the simulated height, as a fraction of that height's range.
DETAIL_FRACTION = 0.02
DETAIL_PERIODS = (64, 32, 16, 8)


def map_size(width):
    """4:3, even height, so block reductions in the simulator stay aligned."""
    return int(width), int(width * 3 / 4) // 2 * 2


def describe(width):
    w, h = map_size(width)
    if width > DIRECT_MAX:
        tiles = math.ceil(w / TILE) * math.ceil(h / TILE)
        text = f"{w}×{h}, {tiles} tiles upscaled from the {PREVIEW_WIDTH}-wide preview"
        if width >= 131072:
            text += ", tens of gigabytes"
        return text
    return f"{w}×{h}, upscaled from the {PREVIEW_WIDTH}-wide preview"


def downsample(image, max_edge):
    """Stride down to max_edge. Nearest sample, so it stays cheap on big maps."""
    image = np.asarray(image)
    h, w = image.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_edge)))
    if step == 1:
        return image
    return np.ascontiguousarray(image[::step, ::step])


def _hash(y, x, seed):
    y = y.astype(np.int64)
    x = x.astype(np.int64)
    n = (y * np.int64(374761393) + x * np.int64(668265263) + np.int64(seed)) & np.int64(0xFFFFFFFF)
    n ^= n >> np.int64(13)
    n = (n * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    return (n & np.int64(0xFFFFFF)).astype(np.float32) * np.float32(1.0 / 0xFFFFFF)


def _blend(grid, fy, fx):
    """Bilinear sample. `fy` is per row, `fx` per column, both in grid index space."""
    iy = np.floor(fy).astype(np.int32)
    ix = np.floor(fx).astype(np.int32)
    ty = (fy - iy).astype(np.float32)
    tx = (fx - ix).astype(np.float32)
    iy = np.clip(iy, 0, grid.shape[0] - 1)
    ix = np.clip(ix, 0, grid.shape[1] - 1)
    iy1 = np.clip(iy + 1, 0, grid.shape[0] - 1)
    ix1 = np.clip(ix + 1, 0, grid.shape[1] - 1)
    wide = grid[:, ix]
    wide = wide + (grid[:, ix1] - wide) * tx
    top = wide[iy]
    return top + (wide[iy1] - top) * ty[:, None]


def _value_noise(yy, xx, period, full_w, seed):
    """Smooth value noise on global pixel coordinates. x wraps with the world.

    The hash is evaluated on the lattice only (one value every `period` pixels)
    and then blended, so a 4K tile does not build a full-size integer grid.
    """
    cells = max(1, int(full_w) // int(period))
    y_start = math.floor(float(yy[0, 0]) / period)
    y_end = math.floor(float(yy[-1, 0]) / period) + 1
    x_start = math.floor(float(xx[0, 0]) / period)
    x_end = math.floor(float(xx[0, -1]) / period) + 1
    ys = np.arange(y_start, y_end + 1, dtype=np.int64)
    xs = np.arange(x_start, x_end + 1, dtype=np.int64)
    grid = _hash(ys[:, None], (xs % cells)[None, :], seed)
    fy = (yy[:, 0] / period - y_start).astype(np.float32)
    fx = (xx[0, :] / period - x_start).astype(np.float32)
    ty = fy - np.floor(fy)
    tx = fx - np.floor(fx)
    fy = np.floor(fy) + ty * ty * (3.0 - 2.0 * ty)
    fx = np.floor(fx) + tx * tx * (3.0 - 2.0 * tx)
    return _blend(grid, fy, fx)


def detail(y0, x0, tile_h, tile_w, full_w, seed):
    """0..1 height grain for one tile. Neighbouring tiles meet without a seam."""
    yy = (y0 + np.arange(tile_h, dtype=np.float32))[:, None]
    xx = (x0 + np.arange(tile_w, dtype=np.float32))[None, :]
    acc = np.zeros((tile_h, tile_w), np.float32)
    amp, norm = 1.0, 0.0
    for i, period in enumerate(DETAIL_PERIODS):
        acc += np.float32(amp) * _value_noise(yy, xx, period, full_w, seed + i * 101)
        norm += amp
        amp *= 0.5
    acc /= np.float32(norm)
    return acc


def _sample_plane(field, y0, x0, tile_h, tile_w, full_h, full_w):
    """Bilinear sample of one plane. x wraps, y clamps. Neighbouring tiles meet."""
    ch, cw = field.shape
    ys = (y0 + np.arange(tile_h, dtype=np.float32) + 0.5) * (ch / full_h) - 0.5
    xs = (x0 + np.arange(tile_w, dtype=np.float32) + 0.5) * (cw / full_w) - 0.5
    iy = np.floor(ys).astype(np.int32)
    ix = np.floor(xs).astype(np.int32)
    ty = (ys - iy).astype(np.float32)
    tx = (xs - ix).astype(np.float32)
    wide = field[:, ix % cw]
    wide = wide + (field[:, (ix + 1) % cw] - wide) * tx
    y0i = np.clip(iy, 0, ch - 1)
    y1i = np.clip(iy + 1, 0, ch - 1)
    top = wide[y0i]
    return (top + (wide[y1i] - top) * ty[:, None]).astype(np.float32)


def _sample_tile(field, y0, x0, tile_h, tile_w, full_h, full_w):
    if field.ndim == 2:
        return _sample_plane(field, y0, x0, tile_h, tile_w, full_h, full_w)
    planes = [_sample_plane(field[..., c], y0, x0, tile_h, tile_w, full_h, full_w)
              for c in range(field.shape[-1])]
    return np.stack(planes, axis=-1)


def _limits(height):
    hmin = float(height.min())
    hmax = float(height.max())
    span = max(hmax - hmin, 1e-9)
    detail_amp = DETAIL_FRACTION * span
    lo, hi = hmin - detail_amp, hmax + detail_amp
    return lo, hi, detail_amp, max(hi - lo, 1e-9)


def _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled):
    meta = {
        "width": int(full_w),
        "height": int(full_h),
        "seed": int(seed),
        "simulated": [int(height.shape[1]), int(height.shape[0])],
        "wrap": "x",
        "height_bits": 16,
        "lo": lo,
        "hi": hi,
        "units_per_code": scale / 65535.0,
        "metres_per_unit": METRES_PER_UNIT,
        "sea_code": (0.0 - lo) / scale * 65535.0,
        "detail": f"seeded value noise, +/- {DETAIL_FRACTION:.0%} of the simulated height range",
        "material_mask": "RGBA, R rock, G ground, B cover, A accent",
        "flow_mask": "RGB, R wetness",
    }
    if tiled:
        meta.update({
            "tile": TILE,
            "tiles_x": math.ceil(full_w / TILE),
            "tiles_y": math.ceil(full_h / TILE),
            "files": {"height": "height/y######_x######.png",
                      "material_mask": "mask/y######_x######.png",
                      "flow": "flow/y######_x######.png"},
        })
    return meta


def _as_float(image):
    return np.ascontiguousarray(image, dtype=np.float32)


def _write_color(image, path, mode):
    Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode).save(
        path, compress_level=1)


def _apply_detail(sampled, y0, x0, full_w, seed, detail_amp):
    grain = detail(y0, x0, sampled.shape[0], sampled.shape[1], full_w, seed)
    return sampled + (grain - np.float32(0.5)) * np.float32(2.0 * detail_amp)


def write_direct(height, mask, flow, full_w, full_h, seed, dest):
    """One image per map. Used up to 8K, where the upscale still fits in memory."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    height_f, mask_f, flow_f = _as_float(height), _as_float(mask), _as_float(flow)
    lo, hi, detail_amp, scale = _limits(height_f)
    sampled = _apply_detail(
        _sample_tile(height_f, 0, 0, full_h, full_w, full_h, full_w),
        0, 0, full_w, seed, detail_amp)
    coded = np.rint(np.clip((sampled - lo) / scale, 0, 1) * 65535).astype(np.uint16)
    _write_png(dest / "height16.png", coded, 16, level=1)
    _write_color(_sample_tile(mask_f, 0, 0, full_h, full_w, full_h, full_w),
                 dest / "material_mask.png", "RGBA")
    _write_color(_sample_tile(flow_f, 0, 0, full_h, full_w, full_h, full_w),
                 dest / "flow_mask.png", "RGB")
    (dest / "manifest.json").write_text(json.dumps(
        _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled=False), indent=1))
    return dest


def write_tiles(height, mask, flow, full_w, full_h, seed, dest, on_status=None, on_tile=None):
    """Write a map wider than memory as 4096 tiles under `dest`.

    `height` is the simulated float field. `mask` is RGBA uint8, `flow` is
    RGB uint8. Every height tile shares one 16-bit scale, recorded in
    manifest.json. Tiles are written on a few threads.
    """
    dest = Path(dest)
    height_dir, mask_dir, flow_dir = dest / "height", dest / "mask", dest / "flow"
    for folder in (height_dir, mask_dir, flow_dir):
        folder.mkdir(parents=True, exist_ok=True)

    height_f, mask_f, flow_f = _as_float(height), _as_float(mask), _as_float(flow)
    lo, hi, detail_amp, scale = _limits(height_f)
    tiles_y = math.ceil(full_h / TILE)
    tiles_x = math.ceil(full_w / TILE)
    total = tiles_y * tiles_x
    state = {"done": 0}
    lock = threading.Lock()

    def one(ty, tx):
        y0, x0 = ty * TILE, tx * TILE
        th, tw = min(TILE, full_h - y0), min(TILE, full_w - x0)
        name = f"y{y0:06d}_x{x0:06d}.png"
        sampled = _apply_detail(
            _sample_tile(height_f, y0, x0, th, tw, full_h, full_w),
            y0, x0, full_w, seed, detail_amp)
        coded = np.rint(np.clip((sampled - lo) / scale, 0, 1) * 65535).astype(np.uint16)
        _write_png(height_dir / name, coded, 16, level=1)
        _write_color(_sample_tile(mask_f, y0, x0, th, tw, full_h, full_w), mask_dir / name, "RGBA")
        _write_color(_sample_tile(flow_f, y0, x0, th, tw, full_h, full_w), flow_dir / name, "RGB")
        with lock:
            state["done"] += 1
            done = state["done"]
        if on_tile:
            on_tile(done, total)
        if on_status and (done == total or done % 8 == 0):
            on_status(f"writing tiles {done}/{total}")

    workers = min(4, os.cpu_count() or 2, total)
    coords = [(ty, tx) for ty in range(tiles_y) for tx in range(tiles_x)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: one(*pair), coords))

    (dest / "manifest.json").write_text(json.dumps(
        _manifest(height, full_w, full_h, seed, lo, hi, scale, tiled=True), indent=1))
    return dest
