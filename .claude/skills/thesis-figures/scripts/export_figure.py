#!/usr/bin/env python3
"""Export a chat-style SVG diagram as a thesis-ready figure (SVG + PNG + PDF).

The inline diagrams Claude draws in chat use CSS variables, theme classes
(c-teal, ts, th ...) and a few SVG features that standalone renderers and
Word do not understand. This script bakes all of that into plain SVG with a
white background, then renders it at print resolution.

Usage:
  python export_figure.py INPUT.svg OUTDIR [--name fig-3-1] [--width-in 6]
         [--dpi 300] [--formats svg,png,pdf] [--font sans|serif|lmodern]
         [--grayscale] [--transparent] [--no-trim]

Requires: cairosvg  (pip install cairosvg --break-system-packages)
"""
import argparse
import os
import re
import sys

THEME = {
    "--p": "#2C2C2A", "--s": "#5F5E5A", "--t": "#888780",
    "--bg2": "#F1EFE8", "--b": "#B4B2A9",
    "--text-primary": "#2C2C2A", "--text-secondary": "#5F5E5A",
    "--text-muted": "#888780", "--surface-1": "#F1EFE8",
    "--surface-2": "#FFFFFF", "--surface-0": "#FFFFFF",
    "--border": "#D3D1C7", "--border-strong": "#888780",
}

# ramp: (fill, stroke, title text, subtitle text)
RAMPS = {
    "purple": ("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"),
    "teal":   ("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"),
    "coral":  ("#FAECE7", "#993C1D", "#712B13", "#993C1D"),
    "pink":   ("#FBEAF0", "#993556", "#72243E", "#993556"),
    "gray":   ("#F1EFE8", "#5F5E5A", "#444441", "#5F5E5A"),
    "blue":   ("#E6F1FB", "#185FA5", "#0C447C", "#185FA5"),
    "green":  ("#EAF3DE", "#3B6D11", "#27500A", "#3B6D11"),
    "amber":  ("#FAEEDA", "#854F0B", "#633806", "#854F0B"),
    "red":    ("#FCEBEB", "#A32D2D", "#791F1F", "#A32D2D"),
}

FONTS = {
    "sans": "'Liberation Sans', Arial, Helvetica, sans-serif",
    "serif": "'Liberation Serif', 'Times New Roman', Times, serif",
    # Added for the LaTeX thesis: matches the body font (lmodern package).
    "lmodern": "'Latin Modern Roman', 'LM Roman 10', serif",
}


def build_css(font: str) -> str:
    css = [
        f"text{{font-family:{FONTS[font]};}}",
        ".t{font-size:14px;fill:%s;}" % THEME["--p"],
        ".th{font-size:14px;font-weight:bold;fill:%s;}" % THEME["--p"],
        ".ts{font-size:12px;fill:%s;}" % THEME["--s"],
        ".box{fill:%s;stroke:%s;}" % (THEME["--surface-1"], THEME["--border-strong"]),
        ".arr{stroke:#5F5E5A;stroke-width:1.5;fill:none;}",
        ".leader{stroke:#888780;stroke-width:0.8;stroke-dasharray:4 3;fill:none;}",
    ]
    for name, (fill, stroke, title, sub) in RAMPS.items():
        shapes = ["rect", "circle", "ellipse", "polygon"]
        direct = ",".join(f".c-{name}>{s}" for s in shapes)
        self_ = ",".join(f"{s}.c-{name}" for s in shapes)
        css.append(f"{direct},{self_}{{fill:{fill};stroke:{stroke};}}")
        css.append(f".c-{name}>.t,.c-{name}>.th{{fill:{title};}}")
        css.append(f".c-{name}>.ts{{fill:{sub};}}")
    return "\n".join(css)


def fix_text_baseline(svg: str) -> str:
    """cairosvg ignores dominant-baseline; emulate 'central' with dy."""
    def repl(m):
        tag = m.group(0)
        if 'dominant-baseline="central"' not in tag:
            return tag
        tag = tag.replace(' dominant-baseline="central"', "")
        if " dy=" not in tag:
            tag = tag[:-1] + ' dy="0.35em">'
        return tag
    return re.sub(r"<text\b[^>]*>", repl, svg)


def to_gray(svg: str) -> str:
    def repl(m):
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        y = round(0.299 * r + 0.587 * g + 0.114 * b)
        return "#%02X%02X%02X" % (y, y, y)
    return re.sub(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})\b", repl, svg)


def prepare(svg: str, width_in: float, font: str, grayscale: bool,
            transparent: bool) -> str:
    vb = re.search(r'viewBox="\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*"', svg)
    if not vb:
        sys.exit("error: root <svg> needs a viewBox")
    vx, vy, vw, vh = (float(v) for v in vb.groups())

    # variables and unsupported features
    svg = re.sub(r"var\((--[a-z0-9-]+)\)", lambda m: THEME.get(m.group(1), "#444441"), svg)
    svg = svg.replace("context-stroke", "#5F5E5A")
    svg = svg.replace("auto-start-reverse", "auto")
    svg = fix_text_baseline(svg)

    # root element: explicit size, namespace
    px_w = width_in * 96
    px_h = px_w * vh / vw
    root = re.search(r"<svg\b[^>]*>", svg)
    new_root = root.group(0)
    new_root = re.sub(r'\swidth="[^"]*"', "", new_root)
    new_root = re.sub(r'\sheight="[^"]*"', "", new_root)
    if "xmlns=" not in new_root:
        new_root = new_root.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    new_root = new_root.replace("<svg", f'<svg width="{px_w:.0f}" height="{px_h:.0f}"', 1)

    bg = "" if transparent else f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" fill="#FFFFFF"/>'
    style = f"<style>{build_css(font)}</style>"
    svg = svg.replace(root.group(0), new_root + style + bg, 1)

    if grayscale:
        svg = to_gray(svg)
    return svg


def trim_viewbox(svg_transparent: str, pad: float = 10.0):
    """Render once with a transparent background and return a tight viewBox."""
    import io
    import cairosvg
    from PIL import Image
    vb = re.search(r'viewBox="\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*"', svg_transparent)
    vx, vy, vw, vh = (float(v) for v in vb.groups())
    scale = 3.0
    png = cairosvg.svg2png(bytestring=svg_transparent.encode("utf-8"), output_width=int(vw * scale))
    alpha = Image.open(io.BytesIO(png)).convert("RGBA").split()[3]
    box = alpha.point(lambda a: 255 if a > 8 else 0).getbbox()
    if not box:
        return vx, vy, vw, vh
    x0, y0, x1, y1 = box
    nx = max(vx, vx + x0 / scale - pad)
    ny = max(vy, vy + y0 / scale - pad)
    nx1 = min(vx + vw, vx + x1 / scale + pad)
    ny1 = min(vy + vh, vy + y1 / scale + pad)
    return nx, ny, nx1 - nx, ny1 - ny


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("outdir")
    ap.add_argument("--name", help="output file stem (default: input name)")
    ap.add_argument("--width-in", type=float, default=6.0, help="printed width in inches (default 6)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--formats", default="svg,png,pdf")
    ap.add_argument("--font", choices=list(FONTS), default="sans")
    ap.add_argument("--grayscale", action="store_true", help="for black-and-white printing")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--no-trim", action="store_true", help="keep the original canvas margins")
    a = ap.parse_args()

    try:
        import cairosvg
    except ImportError:
        sys.exit("error: pip install cairosvg --break-system-packages")

    with open(a.input, encoding="utf-8") as f:
        raw = f.read()
    if not a.no_trim:
        probe = prepare(raw, a.width_in, a.font, False, True)
        nx, ny, nw, nh = trim_viewbox(probe)
        raw = re.sub(r'viewBox="[^"]*"', f'viewBox="{nx:.1f} {ny:.1f} {nw:.1f} {nh:.1f}"', raw, count=1)
    svg = prepare(raw, a.width_in, a.font, a.grayscale, a.transparent)

    os.makedirs(a.outdir, exist_ok=True)
    stem = a.name or os.path.splitext(os.path.basename(a.input))[0]
    wanted = {x.strip() for x in a.formats.split(",")}
    written = []

    if "svg" in wanted:
        p = os.path.join(a.outdir, stem + ".svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg)
        written.append(p)
    data = svg.encode("utf-8")
    if "png" in wanted:
        p = os.path.join(a.outdir, stem + ".png")
        cairosvg.svg2png(bytestring=data, write_to=p,
                         output_width=int(a.width_in * a.dpi),
                         background_color=None if a.transparent else "white")
        written.append(p)
    if "pdf" in wanted:
        p = os.path.join(a.outdir, stem + ".pdf")
        cairosvg.svg2pdf(bytestring=data, write_to=p)
        written.append(p)
    print("\n".join(written))


if __name__ == "__main__":
    main()
