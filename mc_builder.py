"""Module for converting Minecraft heightmaps into actionable commands and scripts."""

import json
import numpy as np

BLOCK_SURFACE = "minecraft:grass_block"

def build_commands(mc_y: np.ndarray, origin_x: int, origin_z: int):
    """
    Builds commands for a seamless surface. Checks neighboring blocks to seal
    any visible holes caused by vertical drop-offs.
    """
    rows, cols = mc_y.shape
    cmds: list[str] = []

    for r in range(rows):
        for c in range(cols):
            mx = origin_x + c
            mz = origin_z + r
            sy = int(mc_y[r, c])

            # Look at adjacent blocks to find the lowest neighbor
            min_y = sy
            if r > 0: min_y = min(min_y, int(mc_y[r-1, c]))
            if r < rows - 1: min_y = min(min_y, int(mc_y[r+1, c]))
            if c > 0: min_y = min(min_y, int(mc_y[r, c-1]))
            if c < cols - 1: min_y = min(min_y, int(mc_y[r, c+1]))

            # If neighbors are level or higher, standard block placement is fine
            if sy == min_y:
                cmds.append(f"/setblock {mx} {sy} {mz} {BLOCK_SURFACE}")
            else:
                # If a neighbor is lower, fill downwards to bridge the gap and seal the hole
                cmds.append(f"/fill {mx} {min_y} {mz} {mx} {sy} {mz} {BLOCK_SURFACE}")
                
    return cmds

def build_js_macro(cmds: list[str], region_w: int, region_h: int, origin_x: int, origin_z: int) -> str:
    """Wraps the generated commands into a JsMacros-compatible JavaScript file."""
    lines = [
        "// ============================================================",
        "// Terrain surface paste script",
        f"// Region : {region_w} × {region_h} blocks",
        f"// Origin : X={origin_x}  Z={origin_z}",
        "// ============================================================",
        "",
        # Inject the python list directly as a JSON array literal
        f"var cmds = {json.dumps(cmds)};",
        "",
        "var sent = 0;",
        "for (var i = 0; i < cmds.length; i++) {",
        "    Chat.say(cmds[i]);",
        "    sent++;",
        "}",
        "",
        "log('Terrain surface complete — sent ' + sent + ' commands.');",
    ]
    return "\n".join(lines)