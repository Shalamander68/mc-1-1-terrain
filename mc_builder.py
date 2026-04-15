import json
import numpy as np

BLOCK_SURFACE = "minecraft:grass_block"

def build_commands(mc_y: np.ndarray, origin_x: int, origin_z: int):
    """
    Builds commands for a seamless surface. If an adjacent block is lower,
    it fills down to that neighbor's height to seal holes.
    """
    rows, cols = mc_y.shape
    cmds: list[str] = []

    for r in range(rows):
        for c in range(cols):
            mx = origin_x + c
            mz = origin_z + r
            sy = int(mc_y[r, c])

            # Find the lowest neighbor to seal gaps
            min_y = sy
            if r > 0: min_y = min(min_y, int(mc_y[r-1, c]))
            if r < rows - 1: min_y = min(min_y, int(mc_y[r+1, c]))
            if c > 0: min_y = min(min_y, int(mc_y[r, c-1]))
            if c < cols - 1: min_y = min(min_y, int(mc_y[r, c+1]))

            if sy == min_y:
                cmds.append(f"/setblock {mx} {sy} {mz} {BLOCK_SURFACE}")
            else:
                cmds.append(f"/fill {mx} {min_y} {mz} {mx} {sy} {mz} {BLOCK_SURFACE}")
                
    return cmds

def build_js_macro(cmds: list[str], region_w: int, region_h: int, origin_x: int, origin_z: int) -> str:
    lines = [
        "// ============================================================",
        "// Terrain surface paste script",
        f"// Region : {region_w} × {region_h} blocks",
        f"// Origin : X={origin_x}  Z={origin_z}",
        "// ============================================================",
        "",
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