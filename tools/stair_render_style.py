"""Render-only scene dressing and the right-hand load HUD for stair mission videos.

Nothing here changes bodies, joints, geoms or contact parameters: the studio adds a sky
texture, a checker floor material and one shadow-casting light before compilation, and the
HUD is drawn on finished RGB frames.
"""

from __future__ import annotations

import mujoco
import numpy as np

WIDTH, HEIGHT = 1280, 720
PANEL_X, PANEL_W = 958, 310
INSET_W, INSET_H = 290, 172
INSET_SLOTS = ((968, 46), (968, 280))
RANK_TOP, RANK_ROWS, RANK_STEP = 516, 5, 36
BAR_X0, BAR_X1 = 968, 1262
LEGEND_X0, LEGEND_X1 = 112, 434
LEGEND_Y = 680  # top edge of the single shared colour bar
TEXT_DARK = "&H00332017"
TEXT_MUTED = "&H00665A50"


def studio_decor(spec: mujoco.MjSpec) -> None:
    """Add a light gradient sky, a checker floor and a directional shadow light."""
    before = (len(spec.bodies), len(spec.geoms))
    sky = spec.add_texture()
    sky.name = "studio_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.97, 0.98, 1.0]
    sky.rgb2 = [0.66, 0.75, 0.86]
    sky.width, sky.height = 512, 3072
    floor_texture = spec.add_texture()
    floor_texture.name = "studio_floor"
    floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_texture.mark = mujoco.mjtMark.mjMARK_EDGE
    floor_texture.rgb1 = [0.90, 0.91, 0.92]
    floor_texture.rgb2 = [0.82, 0.84, 0.86]
    floor_texture.markrgb = [0.68, 0.71, 0.75]
    floor_texture.width = floor_texture.height = 512
    material = spec.add_material()
    material.name = "studio_floor"
    textures = list(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "studio_floor"
    material.textures = textures
    material.texrepeat = [1.0, 1.0]
    material.texuniform = True
    material.reflectance = 0.0
    spec.geom("floor").material = "studio_floor"
    light = spec.worldbody.add_light()
    light.name = "studio_sun"
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0.0, 0.0, 6.0]
    light.dir = [0.35, 0.25, -1.0]
    light.diffuse = [0.55, 0.55, 0.55]
    light.specular = [0.15, 0.15, 0.15]
    light.castshadow = True
    if (len(spec.bodies), len(spec.geoms)) != before:
        raise RuntimeError("studio dressing must not add bodies or geoms")


def configure_visual(model: mujoco.MjModel) -> None:
    model.vis.headlight.ambient[:] = [0.38, 0.38, 0.40]
    model.vis.headlight.diffuse[:] = [0.42, 0.42, 0.42]
    model.vis.headlight.specular[:] = [0.05, 0.05, 0.05]
    model.vis.quality.shadowsize = 4096
    # A tight shadow frustum removes floor acne; follow_sun keeps it centred on the robot.
    model.vis.map.shadowclip = 0.3
    model.vis.map.shadowscale = 0.3


SUN_OFFSET_M = np.array([-0.35, -0.25, 1.0]) * 3.0


def follow_sun(model: mujoco.MjModel, base_position: np.ndarray) -> None:
    model.light_pos[model.light("studio_sun").id] = np.asarray(base_position) + SUN_OFFSET_M


def frost_panel(frame: np.ndarray) -> None:
    """Lighten the right column so dark HUD text and insets stay readable."""
    region = frame[:, PANEL_X : PANEL_X + PANEL_W].astype(float)
    frame[:, PANEL_X : PANEL_X + PANEL_W] = np.rint(region * 0.18 + 246 * 0.82).astype(np.uint8)
    frame[:, PANEL_X : PANEL_X + 2] = (200, 206, 214)


def paste_inset(frame: np.ndarray, inset: np.ndarray, slot: int, border_rgb) -> None:
    x, y = INSET_SLOTS[slot]
    frame[y - 3 : y + INSET_H + 3, x - 3 : x + INSET_W + 3] = np.rint(
        np.asarray(border_rgb) * 255
    ).astype(np.uint8)
    frame[y : y + INSET_H, x : x + INSET_W] = inset


def paint_rank_bars(frame: np.ndarray, values: np.ndarray, colour, maxima: np.ndarray) -> None:
    """Draw one horizontal bar per ranked component; ``maxima`` sets each bar's full scale."""
    for row, (value, top) in enumerate(zip(values, maxima, strict=True)):
        y = RANK_TOP + row * RANK_STEP + 22
        frame[y : y + 8, BAR_X0:BAR_X1] = (222, 226, 232)
        end = BAR_X0 + int(round((BAR_X1 - BAR_X0) * min(max(value / top, 0.0), 1.0)))
        if end > BAR_X0:
            frame[y : y + 8, BAR_X0:end] = np.rint(colour(value, top) * 255).astype(np.uint8)


def paint_legend(frame: np.ndarray, colour, maximum: float) -> None:
    """One shared bar: trunk, legs and arm use the same MPa scale."""
    y0 = LEGEND_Y
    y1 = y0 + 14
    values = np.linspace(0, maximum, LEGEND_X1 - LEGEND_X0)
    frame[y0:y1, LEGEND_X0:LEGEND_X1] = np.rint(colour(values, maximum) * 255).astype(np.uint8)
    frame[y0 - 2 : y0, LEGEND_X0 - 2 : LEGEND_X1 + 2] = 255
    frame[y1 : y1 + 2, LEGEND_X0 - 2 : LEGEND_X1 + 2] = 255
    for tick in np.linspace(LEGEND_X0, LEGEND_X1 - 1, 5).astype(int):
        frame[y0 - 2 : y1 + 2, tick : tick + 2] = 255


def legend_events(prefix: str, maximum: float) -> list[str]:
    lines = [f"{prefix},LegendName,{{\\an6\\pos({LEGEND_X0 - 10},{LEGEND_Y + 7})}}응력 MPa"]
    for x, value in zip(np.linspace(LEGEND_X0, LEGEND_X1, 5), np.linspace(0, maximum, 5)):
        lines.append(f"{prefix},LegendTick,{{\\an2\\pos({int(x)},{LEGEND_Y - 3})}}{value:g}")
    return lines


def panel_static_events(prefix: str) -> list[str]:
    return [
        f"{prefix},HudTitle,{{\\an7\\pos({PANEL_X + 12},12)}}하중 집중 부위 확대",
        f"{prefix},HudTitle,{{\\an7\\pos({PANEL_X + 12},{RANK_TOP - 26})}}지금 응력이 큰 부위",
    ]


def inset_camera(
    lookat: np.ndarray, base_xmat: np.ndarray, left_side: bool, distance: float = 0.62
) -> mujoco.MjvCamera:
    """Side view of a link: from the robot's left for left-side links, right otherwise."""
    rotation = np.asarray(base_xmat).reshape(3, 3)
    yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = yaw - 90 if left_side else yaw + 90
    camera.elevation = -8
    return camera


def hud_styles() -> list[str]:
    return [
        (f"Style: HudTitle,Malgun Gothic,19,{TEXT_DARK},&H00FFFFFF,&H00000000,1,1,0,0,7,0,0,0"),
        (f"Style: HudText,Malgun Gothic,16,{TEXT_DARK},&H00FFFFFF,&H00000000,0,1,0,0,7,0,0,0"),
        (f"Style: HudMuted,Malgun Gothic,14,{TEXT_MUTED},&H00FFFFFF,&H00000000,0,1,0,0,7,0,0,0"),
    ]
