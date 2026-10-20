"""Asset studio for the terrain generator.

Shows the land as each stage finishes, then the maps the game imports.
Export writes library/terrain/zemia_height.png, material_mask.png and
flow_mask.png, plus the generator's own bundle under out/export/.

    python gui.py
"""
import queue
import threading
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from terrain import Config, climate, export, generate, grid, render

ASSETS = Path(__file__).resolve().parents[2]
LIBRARY_TERRAIN = ASSETS / "library" / "terrain"

SIZES = [(320, 224), (480, 336), (640, 448), (1024, 768)]
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


def export_assets(world, library_dir):
    library_dir.mkdir(parents=True, exist_ok=True)
    made = assets_of(world)
    Image.fromarray(made["height"], "RGB").save(library_dir / "zemia_height.png")
    Image.fromarray(made["_mask"], "RGBA").save(library_dir / "material_mask.png")
    Image.fromarray(made["flow"], "RGB").save(library_dir / "flow_mask.png")
    Image.fromarray(_rgb(made["world"]), "RGB").save(library_dir / "preview.png")
    folder = export.bundle(world, parent=Path(__file__).resolve().parent / "out" / "export")
    Image.fromarray(_rgb(made["world"]), "RGB").save(folder / "main_view.png")
    return library_dir, folder


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
        self.size_i = 1
        self.plates = Slider("plates", 4, 48, 18)
        self.land = Slider("land", 15, 60, 29, fmt=lambda v: f"{v}%")
        self.temp = Slider("temperature", -20, 20, 0, fmt=lambda v: f"{v:+d} C")
        self.generate_btn = Button("Generate", 200)
        self.export_btn = Button("Export to game", 200)
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
        self.busy = False
        self.status = "Generate a world. The map updates as each stage finishes."
        self.seed_box = pygame.Rect(16, 72, 252, 28)
        self.events = queue.Queue()
        self.clock = pygame.time.Clock()

    def config(self):
        width, height = SIZES[self.size_i]
        return Config(
            width=width,
            height=height,
            seed=self.seed,
            n_plates=self.plates.value,
            land_fraction=self.land.value / 100,
            temp_offset=float(self.temp.value),
        )

    def start(self):
        if self.busy:
            return
        try:
            self.seed = int(self.seed_text)
        except ValueError:
            self.status = "Seed needs to be a whole number."
            return
        self.busy = True
        self.export_btn.enabled = False
        self.thumbs = {}
        self.world = None
        self.status = "starting"
        cfg = self.config()
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg):
        def on_stage(name):
            self.events.put(("status", f"{name}..."))

        def on_preview(name, value):
            image = _preview_height(name, value)
            if image is not None:
                self.events.put(("image", name, image))

        try:
            world = generate(cfg, verbose=False, on_stage=on_stage, on_preview=on_preview)
        except Exception as err:
            self.events.put(("error", str(err)))
            return
        made = assets_of(world)
        self.events.put(("done", world, made))

    def pump(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "status":
                    self.status = event[1]
                elif kind == "image":
                    self._set_image(event[1], event[2], fit=self.image is None)
                elif kind == "done":
                    self.world, made = event[1], event[2]
                    self.thumbs = {name: made[name] for name in
                                   ("world", "height", "biomes", "material mask", "flow", "canopy")}
                    self._set_image("world", made["world"], fit=True)
                    self.busy = False
                    self.export_btn.enabled = True
                    h, w = self.world.height.shape
                    self.status = (f"seed {self.world.cfg.seed}  {w}x{h}  "
                                   f"land {self.world.land.mean():.0%}  ready to export")
                elif kind == "error":
                    self.busy = False
                    self.status = f"generation failed: {event[1]}"
        except queue.Empty:
            pass

    def _set_image(self, name, image, fit=False):
        self.image_name = name
        self.image = _surface(image)
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
        library, folder = export_assets(self.world, LIBRARY_TERRAIN)
        self.status = f"exported {library.relative_to(ASSETS).as_posix()} and {folder.as_posix()}"

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
        self.screen.blit(self.small.render("map size", True, MUTED), (16, y))
        y += 20
        self.size_rects = []
        x = 16
        for i, (w, h) in enumerate(SIZES):
            rect = pygame.Rect(x, y, 60, 26)
            on = i == self.size_i
            pygame.draw.rect(self.screen, (46, 78, 108) if on else (36, 40, 48), rect, border_radius=3)
            img = self.small.render(f"{w}", True, TEXT)
            self.screen.blit(img, img.get_rect(center=rect.center))
            self.size_rects.append(rect)
            x += 64
        y += 44
        for slider in (self.plates, self.land, self.temp):
            slider.rect.topleft = (16, y + 18)
            slider.rect.w = 248
            slider.draw(self.screen, self.small)
            y += 48
        y += 8
        self.generate_btn.rect.topleft = (16, y)
        self.generate_btn.enabled = not self.busy
        self.generate_btn.label = "Generating..." if self.busy else "Generate"
        self.generate_btn.draw(self.screen, self.font)
        y += 38
        self.export_btn.rect.topleft = (16, y)
        self.export_btn.draw(self.screen, self.font)
        y += 46
        self._wrapped(self.status, 16, y, 252)

    def _wrapped(self, text, x, y, width):
        words, line = text.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if self.small.size(trial)[0] > width and line:
                self.screen.blit(self.small.render(line, True, MUTED), (x, y))
                y += 18
                line = word
            else:
                line = trial
        if line:
            self.screen.blit(self.small.render(line, True, MUTED), (x, y))

    def draw_thumbs(self):
        names = ["world", "height", "biomes", "material mask", "flow", "canopy"]
        self.thumb_rects = []
        y = self.screen.get_height() - 128
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

    def draw(self):
        self.screen.fill(BG)
        self.view = pygame.Rect(300, 12, self.screen.get_width() - 316, self.screen.get_height() - 156)
        self.draw_map()
        self.draw_thumbs()
        self.draw_panel()
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
                self.size_i = i
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
