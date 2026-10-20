"""Asset studio for the terrain generator.

The view is a 640-wide preview. Export upscales that preview to the chosen
resolution, up to 256K (262144 pixels wide). Widths through 8K are one image.
Wider maps are tiles.

Export writes the chosen size under library/terrain/hires/, a game-sized copy
at library/terrain/ (2048 wide at most), and the generator bundle under out/export/.

    python gui.py
"""
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

import hires
from terrain import Config, climate, export, generate, grid, render

ASSETS = Path(__file__).resolve().parents[2]
LIBRARY_TERRAIN = ASSETS / "library" / "terrain"
BUNDLE_PARENT = Path(__file__).resolve().parent / "out" / "export"

HEIGHT_STAGES = {"relief", "hotspots", "texture", "coastline", "erosion", "rivers"}

COVER = {
    climate.ICE, climate.TUNDRA, climate.TAIGA, climate.TEMPERATE_FOREST,
    climate.TEMPERATE_RAINFOREST, climate.TROPICAL_SEASONAL,
    climate.TROPICAL_RAINFOREST,
}
SPARSE = {climate.SHRUBLAND, climate.SAVANNA, climate.STEPPE}

BG = (18, 20, 24)
PANEL = (28, 31, 38)
LINE = (68, 74, 86)
TEXT = (230, 232, 236)
MUTED = (150, 156, 168)
ACCENT = (120, 176, 214)
WARN = (230, 186, 110)
BAR_H = 36

# Share of a typical simulation. A tiled export then counts each tile as 1,
# so a 256K job (thousands of tiles) spends almost the whole bar on writing.
STAGE_WEIGHTS = (
    ("plates", 2),
    ("relief", 6),
    ("hotspots", 2),
    ("texture", 6),
    ("coastline", 5),
    ("pre-rain", 4),
    ("erosion", 28),
    ("temperature", 3),
    ("climate", 5),
    ("rivers", 12),
    ("biomes", 4),
    ("trees", 2),
)


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image.astype(np.uint8))


def _surface(image):
    """pygame wants width, height, 3."""
    ordered = np.transpose(_rgb(image), (1, 0, 2))
    return pygame.surfarray.make_surface(np.ascontiguousarray(ordered))


def _preview_height(stage, value):
    if stage not in HEIGHT_STAGES:
        return None
    height = value[0] if isinstance(value, tuple) else value
    if not isinstance(height, np.ndarray) or height.ndim != 2:
        return None
    height = hires.downsample(height, hires.PREVIEW_WIDTH)
    return render.relief(height).astype(np.uint8)


def material_mask(world):
    """R rock, G ground, B cover, A accent. Matches library/terrain/layout.json."""
    height = world.height
    land = height > 0
    slope = np.hypot(*grid.gradient(height))
    steep = np.percentile(slope[land], 88) if land.any() else 1.0
    rock = np.where(land, np.clip(slope / max(steep, 1e-6), 0, 1), 0).astype(np.float32)

    cover_amt = np.zeros(height.shape, np.float32)
    biome = world.biome
    cover_amt[np.isin(biome, [climate.ICE, climate.TUNDRA])] = 0.92
    cover_amt[np.isin(biome, list(COVER - {climate.ICE, climate.TUNDRA}))] = 0.74
    cover_amt[np.isin(biome, list(SPARSE))] = 0.30
    cover_amt[biome == climate.DESERT] = 0.06
    cover_amt *= land

    canopy = world.trees.cover if world.trees is not None else 0.0
    accent = np.clip(canopy, 0, 1).astype(np.float32) * land
    accent = np.maximum(accent, ((biome == climate.ICE) & land).astype(np.float32))
    accent *= 1.0 - rock

    cover = (1.0 - rock) * cover_amt
    ground = land.astype(np.float32) * (1.0 - rock) * (1.0 - cover_amt)
    total = np.maximum(rock + ground + cover + accent, 1e-6)
    stacked = np.stack((rock, ground, cover, accent), axis=-1) / total[..., None]
    return np.rint(np.clip(stacked, 0, 1) * 255).astype(np.uint8)


def flow_mask(world):
    """Red channel is wetness: rivers, lakes, and heavy runoff."""
    height = world.height
    land = height > 0
    wet = np.zeros(height.shape, np.float32)
    water = world.water
    if water is not None:
        wet = np.clip(water.width / max(world.cfg.river_width_max, 1e-6), 0, 1)
        wet = np.maximum(wet, water.lake_mask.astype(np.float32) * 0.55)
    if world.runoff is not None and land.any():
        scale = max(float(np.percentile(world.runoff[land], 95)), 1e-6)
        wet = np.maximum(wet, np.clip(world.runoff / scale, 0, 1) * 0.4 * land)
    red = np.rint(np.clip(wet, 0, 1) * 255).astype(np.uint8)
    rgb = np.zeros(height.shape + (3,), np.uint8)
    rgb[..., 0] = red
    return rgb


def height_rgb(world):
    span = max(float(world.height.max() - world.height.min()), 1e-9)
    grey = np.rint(np.clip((world.height - world.height.min()) / span, 0, 1) * 255).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def assets_of(world):
    biome = render.layer(world, next(i for i, (name, _) in enumerate(render.LAYERS) if name == "biomes"))[1]
    canopy = render.layer(world, next(i for i, (name, _) in enumerate(render.LAYERS) if name == "canopy"))[1]
    mask = material_mask(world)
    return {
        "world": render.water_map(world).astype(np.uint8),
        "height": height_rgb(world),
        "biomes": biome,
        "material mask": mask[:, :, :3],
        "flow": flow_mask(world),
        "canopy": canopy,
        "_mask": mask,
    }


def _save_rgb(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb(image), "RGB").save(path)


def write_game_copy(made, library_dir):
    """8-bit maps the game can load as one mesh. Never wider than GAME_MAX."""
    library_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(hires.downsample(made["height"], hires.GAME_MAX), library_dir / "zemia_height.png")
    mask = hires.downsample(made["_mask"], hires.GAME_MAX)
    Image.fromarray(np.ascontiguousarray(mask), "RGBA").save(library_dir / "material_mask.png")
    _save_rgb(hires.downsample(made["flow"], hires.GAME_MAX), library_dir / "flow_mask.png")
    _save_rgb(hires.downsample(made["world"], hires.PREVIEW_WIDTH), library_dir / "preview.png")
    return library_dir


def _same_settings(world, settings):
    cfg = world.cfg
    return (cfg.seed == settings["seed"]
            and cfg.n_plates == settings["n_plates"]
            and abs(float(cfg.land_fraction) - settings["land_fraction"]) < 1e-6
            and abs(float(cfg.temp_offset) - settings["temp_offset"]) < 1e-6)


def preview_thumbs(made):
    names = ("world", "height", "biomes", "material mask", "flow", "canopy")
    return {name: hires.downsample(made[name], hires.PREVIEW_WIDTH) for name in names}


class Slider:
    def __init__(self, label, lo, hi, value, fmt=str):
        self.label, self.lo, self.hi, self.fmt = label, lo, hi, fmt
        self.value = value
        self.rect = pygame.Rect(0, 0, 180, 14)
        self.grabbed = False

    def set_from(self, x):
        t = np.clip((x - self.rect.x) / max(1, self.rect.w), 0, 1)
        self.value = int(round(self.lo + (self.hi - self.lo) * t))

    def draw(self, screen, font):
        t = (self.value - self.lo) / max(1, self.hi - self.lo)
        y = self.rect.centery
        knob = int(self.rect.x + t * self.rect.w)
        pygame.draw.line(screen, (60, 64, 74), self.rect.midleft, self.rect.midright, 3)
        pygame.draw.line(screen, ACCENT, self.rect.midleft, (knob, y), 3)
        pygame.draw.circle(screen, (240, 244, 248), (knob, y), 6)
        screen.blit(font.render(f"{self.label}  {self.fmt(self.value)}", True, TEXT),
                    (self.rect.x, self.rect.y - 18))


class Button:
    def __init__(self, label, width=120):
        self.label = label
        self.rect = pygame.Rect(0, 0, width, 28)
        self.enabled = True

    def draw(self, screen, font, hot=False):
        color = (46, 78, 108) if self.enabled else (40, 42, 48)
        if hot and self.enabled:
            color = (64, 104, 140)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, LINE, self.rect, 1, border_radius=4)
        img = font.render(self.label, True, TEXT if self.enabled else MUTED)
        screen.blit(img, img.get_rect(center=self.rect.center))


class Progress:
    """Weighted steps for a job, plus a time estimate from the steps already finished."""

    def __init__(self):
        self.active = False
        self.title = ""
        self.steps = []
        self.index = -1
        self.done_weight = 0.0
        self.total_weight = 1.0
        self.started = 0.0
        self.step_started = 0.0
        self.finished_at = None
        self.label = ""
        self.tile_frac = None

    def begin(self, title, steps):
        self.active = True
        self.title = title
        self.steps = [(name, float(weight)) for name, weight in steps]
        self.index = -1
        self.done_weight = 0.0
        self.total_weight = max(1e-6, sum(weight for _, weight in self.steps))
        now = time.perf_counter()
        self.started = now
        self.step_started = now
        self.finished_at = None
        self.label = "starting"
        self.tile_frac = None

    def advance(self, name):
        """`name` is starting, so the step before it has finished."""
        now = time.perf_counter()
        if self.index >= 0:
            self.done_weight += self.steps[self.index][1]
        found = next((i for i, (step, _) in enumerate(self.steps)
                      if step == name and i > self.index), None)
        if found is None:
            self.steps.append((name, 3.0))
            self.total_weight += 3.0
            found = len(self.steps) - 1
        while self.index + 1 < found:
            self.index += 1
            self.done_weight += self.steps[self.index][1]
        self.index = found
        self.step_started = now
        self.label = name
        self.tile_frac = None

    def tiles(self, done, total):
        if self.index < 0 or self.steps[self.index][0] != "tiles":
            self.advance("tiles")
        total = max(1, total)
        self.tile_frac = min(1.0, done / total)
        self.label = f"tiles {done}/{total}"
        if done >= total:
            self.done_weight += self.steps[self.index][1]
            self.steps[self.index] = ("tiles", 0.0)
            self.tile_frac = 1.0

    def complete(self):
        if self.index >= 0 and self.index < len(self.steps):
            self.done_weight += self.steps[self.index][1]
            self.steps[self.index] = (self.steps[self.index][0], 0.0)
        self.done_weight = self.total_weight
        self.active = False
        self.finished_at = time.perf_counter()
        self.label = "done"
        self.tile_frac = 1.0

    def stop(self):
        self.active = False
        self.finished_at = None
        self.label = ""

    def snapshot(self):
        """Fill fraction, elapsed seconds, seconds remaining, and the step name."""
        now = time.perf_counter()
        if not self.started:
            return 0.0, 0.0, None, ""
        elapsed = (self.finished_at or now) - self.started
        if not self.active:
            if self.finished_at:
                return 1.0, elapsed, 0.0, "done"
            return 0.0, 0.0, None, ""
        current = self.steps[self.index][1] if 0 <= self.index < len(self.steps) else 0.0
        if self.tile_frac is not None:
            creep = self.tile_frac
        else:
            time_in = now - self.step_started
            if self.done_weight > 0:
                expected = current * elapsed / self.done_weight
                creep = min(0.92, time_in / max(expected, 1e-3))
            else:
                creep = min(0.35, time_in / 8.0)
        credited = self.done_weight + current * creep
        fraction = min(0.995, credited / self.total_weight)
        eta = None
        if self.done_weight > 0 and elapsed > 0.3:
            eta = elapsed * max(0.0, self.total_weight - credited) / credited
        return fraction, elapsed, eta, self.label


class Studio:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Terrain asset studio")
        self.screen = pygame.display.set_mode((1360, 860), pygame.RESIZABLE)
        self.font = pygame.font.SysFont("segoeui,calibri,consolas", 16)
        self.small = pygame.font.SysFont("segoeui,calibri,consolas", 13)
        self.big = pygame.font.SysFont("segoeui,calibri,consolas", 20, bold=True)
        self.seed = 7
        self.seed_text = "7"
        self.res_i = 2
        self.job = None
        self.plates = Slider("plates", 4, 48, 18)
        self.land = Slider("land", 15, 60, 29, fmt=lambda v: f"{v}%")
        self.temp = Slider("temperature", -20, 20, 0, fmt=lambda v: f"{v:+d} C")
        self.generate_btn = Button("Generate preview", 252)
        self.export_btn = Button("Export maps", 252)
        self.export_btn.enabled = False
        self.size_rects = []
        self.thumb_rects = []
        self.view = pygame.Rect(300, 16, 1040, 680)
        self.zoom = 1.0
        self.cam = [0.0, 0.0]
        self.drag = None
        self.image = None
        self.image_name = ""
        self.thumbs = {}
        self.world = None
        self.made = None
        self.busy = False
        self.status = "Generate a preview. The view stays downscaled. Export upscales that preview, up to 256K."
        self.seed_box = pygame.Rect(16, 72, 252, 28)
        self.events = queue.Queue()
        self.clock = pygame.time.Clock()
        self.progress = Progress()

    def settings(self):
        return dict(
            seed=self.seed,
            n_plates=self.plates.value,
            land_fraction=self.land.value / 100,
            temp_offset=float(self.temp.value),
        )

    def config(self):
        width, height = hires.map_size(hires.PREVIEW_WIDTH)
        return Config(width=width, height=height, **self.settings())

    def start(self):
        if self.busy:
            return
        try:
            self.seed = int(self.seed_text)
        except ValueError:
            self.status = "Seed needs to be a whole number."
            return
        self.busy = True
        self.job = "preview"
        self.export_btn.enabled = False
        self.thumbs = {}
        self.world = None
        self.status = "starting preview"
        self.progress.begin("Generating preview", STAGE_WEIGHTS + (("maps", 4),))
        cfg = self.config()
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg):
        def on_stage(name):
            self.events.put(("stage", name))

        def on_preview(name, value):
            image = _preview_height(name, value)
            if image is not None:
                self.events.put(("image", name, image))

        try:
            world = generate(cfg, verbose=False, on_stage=on_stage, on_preview=on_preview)
            self.events.put(("stage", "maps"))
            made = assets_of(world)
        except Exception as err:
            self.events.put(("error", f"generation failed: {err}"))
            return
        self.events.put(("done", world, made))

    def pump(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.status = event[1]
                elif kind == "stage":
                    self.progress.advance(event[1])
                    self.status = event[1]
                elif kind == "tiles":
                    done, total = event[1], event[2]
                    self.progress.tiles(done, total)
                    if done == total or done % 8 == 0:
                        self.status = f"writing tiles {done}/{total}"
                elif kind == "image":
                    self._set_image(event[1], event[2], fit=self.image is None)
                elif kind == "preview_ready":
                    self.world, made = event[1], event[2]
                    self.made = made
                    self.thumbs = {name: made[name] for name in
                                   ("world", "height", "biomes", "material mask", "flow", "canopy")}
                    self._set_image("world", made["world"], fit=True)
                elif kind == "done":
                    self.world, made = event[1], event[2]
                    self.made = made
                    self.thumbs = {name: made[name] for name in
                                   ("world", "height", "biomes", "material mask", "flow", "canopy")}
                    self._set_image("world", made["world"], fit=True)
                    self.progress.complete()
                    self.busy = False
                    self.job = None
                    self.export_btn.enabled = True
                    h, w = self.world.height.shape
                    chosen = hires.EXPORT_WIDTHS[self.res_i]
                    self.status = (f"preview {w}×{h}, land {self.world.land.mean():.0%}. "
                                   f"Export writes {hires.describe(chosen)}")
                elif kind == "exported":
                    self.progress.complete()
                    self.busy = False
                    self.job = None
                    self.export_btn.enabled = self.world is not None
                    self.status = event[1]
                    self.thumbs = event[2]
                    self._set_image("world", self.thumbs["world"], fit=True)
                elif kind == "error":
                    self.progress.stop()
                    self.busy = False
                    self.job = None
                    self.export_btn.enabled = self.world is not None
                    self.status = event[1]
        except queue.Empty:
            pass

    def _set_image(self, name, image, fit=False):
        self.image_name = name
        self.image = _surface(hires.downsample(image, hires.PREVIEW_WIDTH))
        if fit:
            self.fit()

    def fit(self):
        if self.image is None:
            return
        mw, mh = self.image.get_width(), self.image.get_height()
        self.zoom = min(self.view.w / mw, self.view.h / mh)
        self.cam = [(mw - self.view.w / self.zoom) / 2, (mh - self.view.h / self.zoom) / 2]

    def export(self):
        if self.world is None or self.busy:
            return
        try:
            self.seed = int(self.seed_text)
        except ValueError:
            self.status = "Seed needs to be a whole number."
            return
        self.busy = True
        self.job = "export"
        self.export_btn.enabled = False
        width = hires.EXPORT_WIDTHS[self.res_i]
        settings = self.settings()
        resim = not _same_settings(self.world, settings)
        w, h = hires.map_size(width)
        steps = list(STAGE_WEIGHTS) + [("maps", 4)] if resim else []
        if width > hires.DIRECT_MAX:
            ntiles = math.ceil(w / hires.TILE) * math.ceil(h / hires.TILE)
            steps.append(("tiles", ntiles))
        else:
            steps.append(("writing", 8))
        self.progress.begin(f"Exporting {hires.EXPORT_LABELS[self.res_i]}", steps)
        threading.Thread(
            target=self._export_worker, args=(width, settings, resim), daemon=True).start()

    def _simulate(self, width, height, settings):
        cfg = Config(width=width, height=height, **settings)

        def on_stage(name):
            self.events.put(("stage", f"{name}"))

        def on_preview(name, value):
            image = _preview_height(name, value)
            if image is not None:
                self.events.put(("image", name, image))

        return generate(cfg, verbose=False, on_stage=on_stage, on_preview=on_preview)

    def _export_worker(self, width, settings, resim):
        w, h = hires.map_size(width)
        try:
            if resim:
                pw, ph = hires.map_size(hires.PREVIEW_WIDTH)
                self.events.put(("status", f"rebuilding preview at {pw}×{ph}"))
                world = self._simulate(pw, ph, settings)
                self.events.put(("stage", "maps"))
                made = assets_of(world)
                self.events.put(("preview_ready", world, made))
            else:
                world, made = self.world, self.made
            folder = LIBRARY_TERRAIN / "hires" / f"seed{settings['seed']}_{w}x{h}"
            self.events.put(("status", f"upscaling preview to {w}×{h}"))
            if width > hires.DIRECT_MAX:
                hires.write_tiles(
                    world.height, made["_mask"], made["flow"], w, h, settings["seed"], folder,
                    on_tile=lambda done, total: self.events.put(("tiles", done, total)),
                )
            else:
                self.events.put(("stage", "writing"))
                hires.write_direct(
                    world.height, made["_mask"], made["flow"], w, h, settings["seed"], folder)
            game = write_game_copy(made, LIBRARY_TERRAIN)
            bundle = export.bundle(world, parent=BUNDLE_PARENT)
            Image.fromarray(_rgb(hires.downsample(made["world"], hires.PREVIEW_WIDTH)), "RGB").save(
                bundle / "main_view.png")
            note = (f"exported {w}×{h} to {folder.as_posix()}. "
                    f"Game copy is in {game.as_posix()}. Bundle is {bundle.as_posix()}")
            self.events.put(("exported", note, preview_thumbs(made)))
        except Exception as err:
            self.events.put(("error", f"export failed: {err}"))

    def draw_map(self):
        pygame.draw.rect(self.screen, (10, 12, 16), self.view)
        if self.image is None:
            return
        surf, z = self.image, self.zoom
        mw, mh = surf.get_width(), surf.get_height()
        x0 = self.view.x - self.cam[0] * z
        y0 = self.view.y - self.cam[1] * z
        self.screen.set_clip(self.view)
        # The map wraps east to west, so draw a second copy when the seam is in view.
        for shift in (0, mw):
            dest = pygame.Rect(int(x0 + shift * z), int(y0), max(1, int(mw * z)), max(1, int(mh * z)))
            if dest.colliderect(self.view):
                self.screen.blit(pygame.transform.smoothscale(surf, dest.size), dest.topleft)
        self.screen.set_clip(None)
        label = self.big.render(self.image_name or "preview", True, (255, 226, 150))
        self.screen.blit(label, (self.view.x + 12, self.view.y + 10))
        note = self.small.render("downscaled preview", True, (255, 226, 150))
        self.screen.blit(note, (self.view.x + 12, self.view.y + 34))

    def draw_panel(self):
        panel = pygame.Rect(0, 0, 284, self.screen.get_height())
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.line(self.screen, LINE, (284, 0), (284, self.screen.get_height()))
        y = 18
        self.screen.blit(self.big.render("Terrain assets", True, TEXT), (16, y))
        y += 36
        self.screen.blit(self.small.render("seed", True, MUTED), (16, y))
        box = pygame.Rect(16, y + 18, 252, 28)
        pygame.draw.rect(self.screen, (16, 18, 22), box, border_radius=4)
        pygame.draw.rect(self.screen, LINE, box, 1, border_radius=4)
        self.screen.blit(self.font.render(self.seed_text or " ", True, TEXT), (24, y + 22))
        self.seed_box = box
        y += 62
        self.screen.blit(self.small.render("export resolution", True, MUTED), (16, y))
        y += 18
        self.size_rects = []
        bw, bh, gap, cols = 46, 24, 4, 5
        for i, label in enumerate(hires.EXPORT_LABELS):
            row, col = divmod(i, cols)
            rect = pygame.Rect(16 + col * (bw + gap), y + row * (bh + gap), bw, bh)
            on = i == self.res_i
            pygame.draw.rect(self.screen, (46, 78, 108) if on else (36, 40, 48), rect, border_radius=3)
            img = self.small.render(label, True, TEXT)
            self.screen.blit(img, img.get_rect(center=rect.center))
            self.size_rects.append(rect)
        y += 2 * (bh + gap) + 8
        chosen = hires.EXPORT_WIDTHS[self.res_i]
        y = self._wrapped(hires.describe(chosen), 16, y, 252, WARN if chosen >= 131072 else MUTED)
        y += 10
        for slider in (self.plates, self.land, self.temp):
            slider.rect.topleft = (16, y + 18)
            slider.rect.w = 248
            slider.draw(self.screen, self.small)
            y += 48
        y += 8
        self.generate_btn.rect.topleft = (16, y)
        self.generate_btn.enabled = not self.busy
        self.generate_btn.label = "Generating..." if self.busy and self.job == "preview" else "Generate preview"
        self.generate_btn.draw(self.screen, self.font)
        y += 38
        self.export_btn.rect.topleft = (16, y)
        self.export_btn.label = "Exporting..." if self.busy and self.job == "export" else "Export maps"
        self.export_btn.draw(self.screen, self.font)
        y += 46
        self._wrapped(self.status, 16, y, 252)

    def _wrapped(self, text, x, y, width, color=MUTED):
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if self.small.size(trial)[0] > width and line:
                self.screen.blit(self.small.render(line, True, color), (x, y))
                y += 18
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.small.render(line, True, color), (x, y))
            y += 18
        return y

    def draw_thumbs(self):
        names = ["world", "height", "biomes", "material mask", "flow", "canopy"]
        self.thumb_rects = []
        y = self.screen.get_height() - 128 - BAR_H
        x = 300
        for name in names:
            rect = pygame.Rect(x, y, 150, 112)
            pygame.draw.rect(self.screen, (12, 14, 18), rect)
            pygame.draw.rect(self.screen, ACCENT if name == self.image_name else LINE, rect, 1)
            image = self.thumbs.get(name)
            if image is not None:
                thumb = pygame.transform.smoothscale(_surface(image), (148, 88))
                self.screen.blit(thumb, (rect.x + 1, rect.y + 1))
            caption = self.small.render(name, True, TEXT)
            self.screen.blit(caption, (rect.x + 6, rect.bottom - 20))
            self.thumb_rects.append((rect, name))
            x += 162

    def draw_bar(self):
        width, height = self.screen.get_size()
        rect = pygame.Rect(0, height - BAR_H, width, BAR_H)
        pygame.draw.rect(self.screen, (12, 14, 18), rect)
        pygame.draw.line(self.screen, LINE, (0, rect.y), (width, rect.y))
        fraction, elapsed, eta, label = self.progress.snapshot()
        recent = self.progress.finished_at and (time.perf_counter() - self.progress.finished_at < 10)
        track = pygame.Rect(12, rect.y + 16, width - 24, 12)
        pygame.draw.rect(self.screen, (36, 40, 48), track, border_radius=4)
        if self.progress.active or recent:
            fill = track.copy()
            fill.w = max(0, int(track.w * fraction))
            if fill.w:
                pygame.draw.rect(self.screen, ACCENT, fill, border_radius=4)
            if self.progress.active:
                title = f"{self.progress.title}   {label}   {fraction * 100:.0f}%"
                if eta is None:
                    right = f"{format_duration(elapsed)} elapsed   estimating"
                else:
                    right = f"{format_duration(elapsed)} elapsed   {format_duration(eta)} left"
            else:
                title = f"{self.progress.title}   finished in {format_duration(elapsed)}"
                right = ""
        else:
            title, right = "Ready", ""
        self.screen.blit(self.small.render(title, True, TEXT), (12, rect.y + 1))
        if right:
            img = self.small.render(right, True, MUTED)
            self.screen.blit(img, (width - 12 - img.get_width(), rect.y + 1))

    def draw(self):
        self.screen.fill(BG)
        self.view = pygame.Rect(300, 12, self.screen.get_width() - 316,
                                self.screen.get_height() - 156 - BAR_H)
        self.draw_map()
        self.draw_thumbs()
        self.draw_panel()
        self.draw_bar()
        pygame.display.flip()

    def on_click(self, pos):
        if self.seed_box.collidepoint(pos):
            return
        if any(slider.rect.inflate(0, 10).collidepoint(pos) for slider in
               (self.plates, self.land, self.temp)):
            slider = next(s for s in (self.plates, self.land, self.temp)
                          if s.rect.inflate(0, 10).collidepoint(pos))
            slider.grabbed = True
            slider.set_from(pos[0])
            return
        for i, rect in enumerate(self.size_rects):
            if rect.collidepoint(pos) and not self.busy:
                self.res_i = i
                return
        if self.generate_btn.rect.collidepoint(pos):
            self.start()
            return
        if self.export_btn.rect.collidepoint(pos) and self.export_btn.enabled:
            self.export()
            return
        for rect, name in self.thumb_rects:
            if rect.collidepoint(pos) and name in self.thumbs:
                self._set_image(name, self.thumbs[name], fit=True)
                return
        if self.view.collidepoint(pos):
            self.drag = pos

    def on_key(self, event):
        if event.key == pygame.K_BACKSPACE:
            self.seed_text = self.seed_text[:-1]
        elif event.key == pygame.K_RETURN:
            self.start()
        elif event.unicode.isascii() and event.unicode.isdigit() and len(self.seed_text) < 10:
            self.seed_text += event.unicode
        elif event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def run(self):
        while True:
            self.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                if event.type == pygame.KEYDOWN:
                    self.on_key(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.drag = None
                    for slider in (self.plates, self.land, self.temp):
                        slider.grabbed = False
                if event.type == pygame.MOUSEMOTION:
                    grabbed = next((s for s in (self.plates, self.land, self.temp) if s.grabbed), None)
                    if grabbed:
                        grabbed.set_from(event.pos[0])
                    elif self.drag and self.image is not None:
                        self.cam[0] -= (event.pos[0] - self.drag[0]) / self.zoom
                        self.cam[1] -= (event.pos[1] - self.drag[1]) / self.zoom
                        self.drag = event.pos
                if event.type == pygame.MOUSEWHEEL and self.image is not None:
                    self.zoom = float(np.clip(self.zoom * (1.12 ** event.y), 0.15, 12))
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    Studio().run()
