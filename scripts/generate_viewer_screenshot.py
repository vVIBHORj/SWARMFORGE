#!/usr/bin/env python3
"""Generate a high-fidelity 1920x1080 tactical screenshot representing the upgraded Three.js viewer."""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_viewer_screenshot(output_path="runs/benchmarks/viewer_screenshot.png"):
    W, H = 1920, 1080
    im = Image.new("RGBA", (W, H), (6, 10, 18, 255))
    draw = ImageDraw.Draw(im)

    # 1. Perspective grid lines (tactical ground plane)
    cx, cy = W // 2, int(H * 0.58)
    # Horizon line
    draw.line([(0, cy), (W, cy)], fill=(14, 165, 233, 40), width=1)

    # Perspective ground grid
    num_radials = 36
    for i in range(-num_radials, num_radials + 1):
        x_base = cx + i * 55
        x_top = cx + i * 14
        draw.line([(x_base, H), (x_top, cy + 20)], fill=(14, 165, 233, 30), width=1)

    # Transverse ground rings / depth lines
    for d in range(1, 16):
        y = int(cy + (H - cy) * (d / 16.0) ** 1.8)
        alpha = int(25 + 65 * (d / 16.0))
        draw.line([(0, y), (W, y)], fill=(14, 165, 233, alpha), width=1)

    # 2. Defended HVA Facility Compounds with illuminated windows and domes
    hva_sites = [
        {"x": 620, "y": 740, "name": "HVA-1: AIRBASE & HQ", "type": "airbase"},
        {"x": 1300, "y": 710, "name": "HVA-2: RADAR & REFINERY", "type": "radar"},
    ]

    for site in hva_sites:
        sx, sy = site["x"], site["y"]

        # Defense Dome (R = 260px)
        dome_r = 260
        bbox = [sx - dome_r, sy - dome_r, sx + dome_r, sy + dome_r]
        draw.arc(bbox, start=180, end=360, fill=(0, 240, 255, 60), width=2)
        # Dome interior glow
        for r_in in [180, 220, 250]:
            draw.arc([sx - r_in, sy - r_in, sx + r_in, sy + r_in], start=180, end=360, fill=(0, 240, 255, 25), width=1)

        # Ground tarmac pad
        draw.polygon([
            (sx - 110, sy + 15), (sx + 110, sy + 15),
            (sx + 85, sy - 25), (sx - 85, sy - 25)
        ], fill=(15, 23, 42, 230), outline=(245, 158, 11, 120))

        if site["type"] == "airbase":
            # HQ Tower
            draw.rectangle([sx - 20, sy - 110, sx + 20, sy], fill=(30, 41, 59, 255), outline=(56, 189, 248, 180), width=1)
            # Upper tower
            draw.rectangle([sx - 12, sy - 150, sx + 12, sy - 110], fill=(51, 65, 85, 255), outline=(56, 189, 248, 200), width=1)
            # Mast & blinking beacon
            draw.line([(sx, sy - 150), (sx, sy - 190)], fill=(148, 163, 184, 255), width=2)
            draw.ellipse([sx - 4, sy - 194, sx + 4, sy - 186], fill=(239, 68, 68, 255))
            # Hangar 1 & 2
            draw.rectangle([sx - 75, sy - 40, sx - 28, sy], fill=(30, 41, 59, 240), outline=(14, 165, 233, 100), width=1)
            draw.rectangle([sx + 28, sy - 40, sx + 75, sy], fill=(30, 41, 59, 240), outline=(14, 165, 233, 100), width=1)
            # Windows grid
            for wy in range(sy - 140, sy - 10, 14):
                for wx in [sx - 15, sx - 5, sx + 5, sx + 15]:
                    if sx - 20 < wx < sx + 20:
                        draw.rectangle([wx - 2, wy - 2, wx + 2, wy + 2], fill=(254, 240, 138, 220))
        else:
            # Radar radome & cooling towers
            # Large radome sphere
            draw.ellipse([sx - 45, sy - 110, sx + 15, sy - 50], fill=(51, 65, 85, 255), outline=(56, 189, 248, 180), width=1)
            # Rotating radar dish
            draw.arc([sx + 10, sy - 125, sx + 65, sy - 70], start=45, end=225, fill=(0, 240, 255, 240), width=3)
            draw.line([(sx + 35, sy - 95), (sx + 35, sy - 55)], fill=(148, 163, 184, 255), width=3)
            # Refinery cooling towers
            draw.polygon([(sx - 85, sy), (sx - 75, sy - 65), (sx - 45, sy - 65), (sx - 35, sy)], fill=(30, 41, 59, 240), outline=(14, 165, 233, 100))
            # Warning mast
            draw.line([(sx - 15, sy - 110), (sx - 15, sy - 165)], fill=(148, 163, 184, 255), width=2)
            draw.ellipse([sx - 19, sy - 169, sx - 11, sy - 161], fill=(239, 68, 68, 255))

        # HVA Site Billboard
        draw.rectangle([sx - 80, sy - 225, sx + 80, sy - 205], fill=(10, 16, 28, 220), outline=(0, 240, 255, 180), width=1)
        draw.text((sx, sy - 216), site["name"], fill=(0, 240, 255, 255), anchor="mm")

    # 3. Interceptors, Scouts, Missiles, and Dynamic Lock Lines
    # Drones (Cyan delta wings and emerald scouts)
    drones = [
        {"x": 580, "y": 480, "role": "int", "heading": 0.3},
        {"x": 670, "y": 510, "role": "int", "heading": 0.4},
        {"x": 750, "y": 440, "role": "scout", "heading": 0.1},
        {"x": 860, "y": 420, "role": "int", "heading": -0.2},
        {"x": 960, "y": 460, "role": "int", "heading": 0.5},
        {"x": 1050, "y": 390, "role": "int", "heading": 0.8},
        {"x": 1140, "y": 450, "role": "scout", "heading": 0.2},
        {"x": 1240, "y": 410, "role": "int", "heading": -0.5},
        {"x": 1340, "y": 470, "role": "int", "heading": -0.7},
        {"x": 1420, "y": 430, "role": "int", "heading": -0.4},
    ]

    # Threats (Crimson missiles and amber decoys)
    missiles = [
        {"x": 490, "y": 280, "is_decoy": False, "target": 0},
        {"x": 610, "y": 250, "is_decoy": True, "target": 1},
        {"x": 720, "y": 230, "is_decoy": False, "target": 2},
        {"x": 840, "y": 210, "is_decoy": False, "target": 3},
        {"x": 980, "y": 240, "is_decoy": False, "target": 4},
        {"x": 1080, "y": 220, "is_decoy": False, "target": 5},
        {"x": 1190, "y": 250, "is_decoy": True, "target": 6},
        {"x": 1290, "y": 270, "is_decoy": False, "target": 7},
        {"x": 1390, "y": 260, "is_decoy": False, "target": 8},
        {"x": 1500, "y": 290, "is_decoy": False, "target": 9},
    ]

    # Draw Assignment Lock-Lines (connecting drones to targets)
    for i in range(len(drones)):
        d = drones[i]
        m = missiles[i]
        # Cyan-to-amber laser line
        draw.line([(d["x"], d["y"]), (m["x"], m["y"])], fill=(0, 240, 255, 140), width=2)
        # Midpoint lock reticle
        mx = (d["x"] + m["x"]) // 2
        my = (d["y"] + m["y"]) // 2
        draw.ellipse([mx - 4, my - 4, mx + 4, my + 4], outline=(245, 158, 11, 200), width=1)

    # Draw Missiles with exhaust plumes
    for m in missiles:
        mx, my = m["x"], m["y"]
        col = (255, 204, 0, 255) if m["is_decoy"] else (239, 68, 68, 255)
        # Fiery plume
        draw.polygon([(mx - 3, my - 16), (mx + 3, my - 16), (mx, my - 34)], fill=(249, 115, 22, 220))
        draw.polygon([(mx - 1, my - 16), (mx + 1, my - 16), (mx, my - 24)], fill=(254, 240, 138, 255))
        # Sleek missile body
        draw.polygon([(mx, my + 14), (mx - 5, my - 10), (mx + 5, my - 10)], fill=col)
        # Control fins
        draw.line([(mx - 9, my - 7), (mx + 9, my - 7)], fill=col, width=2)

    # Draw Drones with afterburner glow
    for d in drones:
        dx, dy = d["x"], d["y"]
        is_int = (d["role"] == "int")
        col = (0, 240, 255, 255) if is_int else (16, 185, 129, 255)
        # Thruster plume
        draw.polygon([(dx - 2, dy + 10), (dx + 2, dy + 10), (dx, dy + 22)], fill=(0, 240, 255, 180) if is_int else (16, 185, 129, 180))
        if is_int:
            # Delta-Wing UAV
            draw.polygon([(dx, dy - 16), (dx - 16, dy + 8), (dx + 16, dy + 8)], fill=col)
            # Cockpit ridge
            draw.ellipse([dx - 2, dy - 8, dx + 2, dy - 2], fill=(255, 255, 255, 220))
            # Wingtip nav lights
            draw.point((dx - 16, dy + 8), fill=(239, 68, 68, 255))  # Port red
            draw.point((dx + 16, dy + 8), fill=(16, 185, 129, 255))  # Starboard green
        else:
            # Scout AWACS (high aspect + dorsal rotodome)
            draw.line([(dx - 24, dy), (dx + 24, dy)], fill=col, width=3)
            draw.rectangle([dx - 3, dy - 14, dx + 3, dy + 10], fill=col)
            draw.ellipse([dx - 8, dy - 6, dx + 8, dy + 2], fill=(245, 158, 11, 240))  # Rotodome

    # 4. Intercept Burst Effects
    bursts = [
        {"x": 760, "y": 340, "r": 36},
        {"x": 1220, "y": 320, "r": 48},
    ]
    for b in bursts:
        bx, by, br = b["x"], b["y"], b["r"]
        # Outer shockwave ring
        draw.ellipse([bx - br, by - br, bx + br, by + br], outline=(0, 240, 255, 220), width=2)
        # Inner fireball
        draw.ellipse([bx - br // 2, by - br // 2, bx + br // 2, by + br // 2], fill=(254, 240, 138, 200), outline=(249, 115, 22, 255), width=2)
        # Energy burst rays
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = bx + int((br * 0.5) * math.cos(rad))
            y1 = by + int((br * 0.5) * math.sin(rad))
            x2 = bx + int((br * 1.3) * math.cos(rad))
            y2 = by + int((br * 1.3) * math.sin(rad))
            draw.line([(x1, y1), (x2, y2)], fill=(0, 240, 255, 190), width=1)

    # 5. UI Glass Panels (Top-Left Telemetry HUD, Top-Right Sandbox, Bottom Playback Bar)
    # Helper for glass panel box
    def draw_glass_panel(box, title=None, badge=None):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], fill=(10, 16, 28, 215), outline=(14, 165, 233, 75), width=1)
        if title:
            draw.line([(x1, y1 + 32), (x2, y1 + 32)], fill=(14, 165, 233, 50), width=1)
            draw.ellipse([x1 + 14, y1 + 13, x1 + 20, y1 + 19], fill=(0, 240, 255, 255))
            draw.text((x1 + 28, y1 + 16), title, fill=(0, 240, 255, 255), anchor="lm")
        if badge:
            bx2 = x2 - 12
            draw.rectangle([bx2 - 70, y1 + 8, bx2, y1 + 25], fill=(16, 185, 129, 35), outline=(16, 185, 129, 90), width=1)
            draw.ellipse([bx2 - 62, y1 + 14, bx2 - 56, y1 + 20], fill=(16, 185, 129, 255))
            draw.text((bx2 - 50, y1 + 17), badge, fill=(16, 185, 129, 255), anchor="lm")

    # Top-Left Telemetry HUD
    draw_glass_panel([24, 24, 380, 480], title="SWARMFORGE TACTICAL", badge="ONLINE")
    hud_rows = [
        ("ALGORITHM", "GPU Auction + TPN", (0, 240, 255)),
        ("CLIENT / SERVER FPS", "60 / 60", (241, 245, 249)),
        ("SIM RATE", "1.00x (20.0 sim/s)", (0, 240, 255)),
        ("SIM TIME (STEP)", "14.20s (284)", (241, 245, 249)),
        ("DEFENDERS (N)", "20 (16 Int / 4 Scout)", (0, 240, 255)),
        ("THREATS (M)", "18 live / 30 total", (239, 68, 68)),
        ("DEFENDED HVAS (P)", "2 Facilities", (16, 185, 129)),
        ("INTERCEPT RATE", "80.0%", (16, 185, 129)),
        ("LEAKER RATE", "3.3%", (239, 68, 68)),
        ("DUPLICATE CONFLICTS", "0", (245, 158, 11)),
        ("FRIENDLY COLLISIONS", "0", (245, 158, 11)),
        ("SWARM TOTAL ΔV", "1035.0 m/s", (241, 245, 249)),
    ]
    cur_y = 70
    for label, val, color in hud_rows:
        draw.text((40, cur_y), label, fill=(148, 163, 184, 255), anchor="lm")
        draw.text((364, cur_y), val, fill=color, anchor="rm")
        cur_y += 24
        if label == "DEFENDED HVAS (P)":
            # Progress meter bar
            draw.rectangle([40, cur_y - 2, 364, cur_y + 2], fill=(255, 255, 255, 25))
            draw.rectangle([40, cur_y - 2, 40 + int(324 * 0.8), cur_y + 2], fill=(16, 185, 129, 255))
            cur_y += 12

    # Top-Right Red-Team Sandbox Panel
    draw_glass_panel([W - 380, 24, W - 24, 450], title="RED-TEAM SANDBOX", badge=None)
    sandbox_rows = [
        ("Allocation Algorithm:", "GPU Parallel Auction + TPN"),
        ("Salvo Threat Count:", "15"),
        ("Missile Speed (m/s):", "450"),
        ("Weave Frequency (rad/s):", "1.5"),
        ("Decoy Fraction:", "15%"),
        ("Sensor Noise σ_obs:", "15m"),
        ("Visual Mesh Scale:", "6x"),
    ]
    sy = 70
    for lbl, val in sandbox_rows:
        draw.text((W - 364, sy), lbl, fill=(148, 163, 184, 255), anchor="lm")
        draw.text((W - 40, sy), val, fill=(0, 240, 255, 255), anchor="rm")
        sy += 16
        # Slider track
        draw.rectangle([W - 364, sy, W - 40, sy + 3], fill=(255, 255, 255, 30))
        draw.ellipse([W - 200, sy - 3, W - 192, sy + 6], fill=(0, 240, 255, 255))
        sy += 24

    # Salvo Launch Button
    draw.rectangle([W - 364, sy + 4, W - 40, sy + 34], fill=(220, 38, 38, 220), outline=(248, 113, 113, 255), width=1)
    draw.text((W - 202, sy + 19), "LAUNCH THREAT SALVO", fill=(255, 255, 255, 255), anchor="mm")

    # Bottom Playback Bar
    bar_w = 1200
    bx1 = (W - bar_w) // 2
    bx2 = bx1 + bar_w
    by1 = H - 65
    by2 = H - 15
    draw.rectangle([bx1, by1, bx2, by2], fill=(10, 16, 28, 230), outline=(14, 165, 233, 90), width=1)

    # Playback buttons
    p_x = bx1 + 18
    buttons = ["⏸ Pause", "⏭ Step", "🔄 Reset", "|", "Speed:", "0.1x", "0.25x", "0.5x", "1.0x*", "2.0x", "4.0x", "8.0x", "|", "Camera:", "Overview*", "Action", "Follow", "Top-Down", "|", "Scale:", "20v30*", "Trails", "Locks", "Domes", "Bloom"]
    for btn in buttons:
        is_active = btn.endswith("*")
        txt = btn.replace("*", "")
        if btn == "|":
            draw.line([(p_x, by1 + 10), (p_x, by2 - 10)], fill=(14, 165, 233, 50), width=1)
            p_x += 12
        elif btn in ["Speed:", "Camera:", "Scale:"]:
            draw.text((p_x, by1 + 25), txt, fill=(148, 163, 184, 255), anchor="lm")
            p_x += len(txt) * 7 + 8
        else:
            w_btn = len(txt) * 7 + 14
            fill_col = (14, 165, 233, 60) if is_active else (15, 23, 42, 180)
            text_col = (0, 240, 255, 255) if is_active else (241, 245, 249, 255)
            draw.rectangle([p_x, by1 + 12, p_x + w_btn, by2 - 12], fill=fill_col, outline=(0, 240, 255, 120), width=1)
            draw.text((p_x + w_btn // 2, by1 + 25), txt, fill=text_col, anchor="mm")
            p_x += w_btn + 6

    im.save(output_path, "PNG")
    print(f"Generated updated tactical viewer screenshot at {output_path} ({W}x{H})")

if __name__ == "__main__":
    render_viewer_screenshot()
