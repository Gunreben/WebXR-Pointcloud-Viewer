#!/usr/bin/env python3
"""
Interactive point cloud transform editor with 3D preview.

Usage
-----
python pointcloud_editor.py input.pcd output.pcd
python pointcloud_editor.py input.ply output.ply

Adjust origin mode, rotation, and translation with sliders,
then press "Save & Exit" to write the transformed file.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

from pointcloud_transform import (
    compute_origin_from_mode,
    parse_pcd_metadata,
    parse_ply_ascii_records,
    parse_ply_binary_records,
    parse_ply_metadata,
    pcd_column_layout,
    read_all_bytes,
    transform_pcd,
    transform_ply,
)

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider
except ImportError:
    print(
        "matplotlib is required for the editor.\n"
        "Install it with:  pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1)

MAX_PREVIEW_POINTS = 8_000


# ── Point extraction ────────────────────────────────────────


def extract_xyz_pcd(path: Path) -> list[list[float]]:
    raw = read_all_bytes(path)
    meta = parse_pcd_metadata(raw)
    names, fmt_parts = pcd_column_layout(meta)
    xi, yi, zi = names.index("x"), names.index("y"), names.index("z")

    if meta.data_kind == "ascii":
        body = raw[meta.data_offset :].decode("utf-8")
        rows = [ln for ln in body.splitlines() if ln.strip()]
        return [
            [float(p[xi]), float(p[yi]), float(p[zi])]
            for p in (line.split() for line in rows)
        ]

    rec = struct.Struct("<" + "".join(fmt_parts))
    data = raw[meta.data_offset :]
    stride = rec.size
    return [
        [float(v[xi]), float(v[yi]), float(v[zi])]
        for i in range(meta.points)
        for v in [rec.unpack_from(data, i * stride)]
    ]


def extract_xyz_ply(path: Path) -> list[list[float]]:
    raw = read_all_bytes(path)
    meta = parse_ply_metadata(raw)
    payload = raw[meta.data_offset :]

    vert_idx = next(
        (i for i, el in enumerate(meta.elements) if el.name == "vertex"), None
    )
    if vert_idx is None:
        raise ValueError("PLY has no vertex element.")

    props = [p.name for p in meta.elements[vert_idx].properties]
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")

    if meta.fmt == "ascii":
        lines = [ln for ln in payload.decode("utf-8").splitlines() if ln.strip()]
        records, _ = parse_ply_ascii_records(lines, meta)
    else:
        records = parse_ply_binary_records(payload, meta)

    return [
        [float(r[xi]), float(r[yi]), float(r[zi])]
        for r in records[vert_idx]
    ]


# ── Helpers ─────────────────────────────────────────────────


def subsample(pts: list, n: int) -> list:
    if len(pts) <= n:
        return pts
    step = len(pts) / n
    return [pts[int(i * step)] for i in range(n)]


def bounds(pts: list[list[float]]):
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for p in pts:
        for a in range(3):
            if p[a] < lo[a]:
                lo[a] = p[a]
            if p[a] > hi[a]:
                hi[a] = p[a]
    return lo, hi


def transform_preview(pts, origin, rot_deg):
    rx, ry, rz = (math.radians(v) for v in rot_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    out = []
    for p in pts:
        x, y, z = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        out.append((x, y, z))
    return out


# ── Editor UI ───────────────────────────────────────────────


class Editor:
    ORIGIN_MODES = ("bottom-center", "center", "min")

    def __init__(self, input_path: Path, output_path: Path, raw_pts: list):
        self.input_path = input_path
        self.output_path = output_path
        self.raw_pts = raw_pts
        self.preview_pts = subsample(raw_pts, MAX_PREVIEW_POINTS)
        self.lo, self.hi = bounds(raw_pts)

        self.origin_mode = "bottom-center"
        self.origin = compute_origin_from_mode(self.origin_mode, self.lo, self.hi)
        self.rot = [0.0, 0.0, 0.0]
        self.saved = False

        self._build()
        plt.show()

    # ── layout ──────────────────────────────────────────────

    def _build(self):
        self.fig = plt.figure(figsize=(13, 9), facecolor="#1a1a1a")
        self.fig.canvas.manager.set_window_title("Point Cloud Transform Editor")

        # 3D scatter — upper-left area
        self.ax3d = self.fig.add_axes(
            [0.02, 0.32, 0.62, 0.65], projection="3d", facecolor="#111111"
        )

        # origin-mode radio — right side
        rax = self.fig.add_axes(
            [0.70, 0.72, 0.26, 0.20], facecolor="#222222"
        )
        rax.set_title("Origin mode", color="white", fontsize=10, pad=6)
        self.radio = RadioButtons(
            rax, self.ORIGIN_MODES, active=0, activecolor="#44ffaa"
        )
        for lbl in self.radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(10)
        self.radio.on_clicked(self._on_mode)

        # rotation sliders
        kw = dict(color="#44ffaa", track_color="#333333")
        s = lambda y, label: Slider(
            self.fig.add_axes([0.12, y, 0.52, 0.025]),
            label,
            -180,
            180,
            valinit=0,
            valstep=1,
            **kw,
        )
        self.s_rx = s(0.24, "Rot X °")
        self.s_ry = s(0.19, "Rot Y °")
        self.s_rz = s(0.14, "Rot Z °")
        for sl in (self.s_rx, self.s_ry, self.s_rz):
            sl.label.set_color("white")
            sl.valtext.set_color("white")
            sl.on_changed(self._on_slider)

        # translate sliders
        span = [self.hi[i] - self.lo[i] for i in range(3)]
        lim = max(span) * 1.5 if max(span) > 0 else 10
        st = lambda y, label: Slider(
            self.fig.add_axes([0.72, y, 0.24, 0.025]),
            label,
            -lim,
            lim,
            valinit=0,
            valfmt="%.2f",
            color="#66aaff",
            track_color="#333333",
        )
        self.s_tx = st(0.24, "Tr X")
        self.s_ty = st(0.19, "Tr Y")
        self.s_tz = st(0.14, "Tr Z")
        for sl in (self.s_tx, self.s_ty, self.s_tz):
            sl.label.set_color("white")
            sl.valtext.set_color("white")
            sl.on_changed(self._on_slider)

        # buttons
        self.btn_reset = Button(
            self.fig.add_axes([0.70, 0.56, 0.11, 0.05]),
            "Reset",
            color="#333333",
            hovercolor="#555555",
        )
        self.btn_reset.label.set_color("white")
        self.btn_reset.on_clicked(self._on_reset)

        self.btn_save = Button(
            self.fig.add_axes([0.83, 0.56, 0.14, 0.05]),
            "Save & Exit",
            color="#225533",
            hovercolor="#338844",
        )
        self.btn_save.label.set_color("white")
        self.btn_save.on_clicked(self._on_save)

        # info text area
        iax = self.fig.add_axes([0.03, 0.01, 0.94, 0.10])
        iax.axis("off")
        self.info = iax.text(
            0,
            0.5,
            "",
            transform=iax.transAxes,
            fontsize=9,
            family="monospace",
            color="white",
            va="center",
        )

        self._refresh()

    # ── callbacks ───────────────────────────────────────────

    def _on_mode(self, label):
        self.origin_mode = label
        self.origin = compute_origin_from_mode(label, self.lo, self.hi)
        self._refresh()

    def _on_slider(self, _val):
        self.rot = [self.s_rx.val, self.s_ry.val, self.s_rz.val]
        self._refresh()

    def _on_reset(self, _evt):
        for sl in (self.s_rx, self.s_ry, self.s_rz, self.s_tx, self.s_ty, self.s_tz):
            sl.set_val(0)
        self.radio.set_active(0)
        self.origin_mode = "bottom-center"
        self.origin = compute_origin_from_mode(self.origin_mode, self.lo, self.hi)
        self.rot = [0.0, 0.0, 0.0]
        self._refresh()

    def _on_save(self, _evt):
        translate = (self.s_tx.val, self.s_ty.val, self.s_tz.val)
        rot = tuple(self.rot)
        suffix = self.input_path.suffix.lower()
        func = transform_pcd if suffix == ".pcd" else transform_ply
        summary = func(
            input_path=self.input_path,
            output_path=self.output_path,
            origin_mode=None,
            origin_value=self.origin,
            rotate_deg=rot,
            translate=translate,
        )

        ox, oy, oz = self.origin
        rx, ry, rz = rot
        tx, ty, tz = translate
        cli_parts = [
            "python pointcloud_transform.py",
            f'"{self.input_path}" "{self.output_path}"',
            f"--origin {ox:.6g} {oy:.6g} {oz:.6g}",
        ]
        if any(v != 0 for v in rot):
            cli_parts.append(f"--rotate {rx:.6g} {ry:.6g} {rz:.6g}")
        if any(v != 0 for v in translate):
            cli_parts.append(f"--translate {tx:.6g} {ty:.6g} {tz:.6g}")

        print()
        print(f"  Saved → {self.output_path}")
        print(f"  Points:  {summary['points']:,}")
        print(f"  Origin:  ({ox:.4f}, {oy:.4f}, {oz:.4f})  [{self.origin_mode}]")
        print(f"  Rotate:  ({rx:.1f}, {ry:.1f}, {rz:.1f}) deg")
        print(f"  Translate: ({tx:.4f}, {ty:.4f}, {tz:.4f})")
        print()
        print("  Equivalent CLI command:")
        print("  " + " \\\n    ".join(cli_parts))
        print()

        self.saved = True
        plt.close(self.fig)

    # ── drawing ─────────────────────────────────────────────

    def _refresh(self):
        translate = (self.s_tx.val, self.s_ty.val, self.s_tz.val)
        transformed = transform_preview(self.preview_pts, self.origin, self.rot)
        # apply translate
        xs = [p[0] + translate[0] for p in transformed]
        ys = [p[1] + translate[1] for p in transformed]
        zs = [p[2] + translate[2] for p in transformed]

        ax = self.ax3d
        elev, azim = ax.elev, ax.azim
        ax.cla()

        # matplotlib Z-up → Three.js Y-up: plot (X, Z, Y)
        ax.scatter(xs, zs, ys, s=0.4, c=ys, cmap="viridis", alpha=0.7)

        # ground grid reference
        lo_t, hi_t = bounds([list(t) for t in zip(xs, ys, zs)])
        extent = max(hi_t[a] - lo_t[a] for a in range(3)) * 0.6
        if extent < 0.01:
            extent = 1.0
        g = extent
        for v in [-g, -g / 2, 0, g / 2, g]:
            ax.plot([-g, g], [v, v], [0, 0], color="#333333", lw=0.4, alpha=0.5)
            ax.plot([v, v], [-g, g], [0, 0], color="#333333", lw=0.4, alpha=0.5)

        # axis arrows
        al = extent * 0.25
        ax.quiver(0, 0, 0, al, 0, 0, color="red", arrow_length_ratio=0.15, lw=1.5)
        ax.quiver(0, 0, 0, 0, al, 0, color="blue", arrow_length_ratio=0.15, lw=1.5)
        ax.quiver(0, 0, 0, 0, 0, al, color="green", arrow_length_ratio=0.15, lw=1.5)

        mid = [(lo_t[a] + hi_t[a]) * 0.5 for a in range(3)]
        half = max(hi_t[a] - lo_t[a] for a in range(3)) * 0.55
        if half < 0.01:
            half = 1.0
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[2] - half, mid[2] + half)
        ax.set_zlim(mid[1] - half, mid[1] + half)

        ax.set_xlabel("X", color="red", fontsize=8)
        ax.set_ylabel("Z", color="blue", fontsize=8)
        ax.set_zlabel("Y", color="green", fontsize=8)
        ax.tick_params(colors="#666666", labelsize=6)
        ax.set_title(
            f"{len(self.raw_pts):,} pts  ({len(self.preview_pts):,} shown)",
            color="white",
            fontsize=9,
        )

        if elev is not None:
            ax.view_init(elev=elev, azim=azim)

        ox, oy, oz = self.origin
        rx, ry, rz = self.rot
        tx, ty, tz = translate
        self.info.set_text(
            f"Origin: ({ox:.3f}, {oy:.3f}, {oz:.3f})  [{self.origin_mode}]    "
            f"Rot: ({rx:.0f}°, {ry:.0f}°, {rz:.0f}°)    "
            f"Translate: ({tx:.2f}, {ty:.2f}, {tz:.2f})\n"
            f"Input: {self.input_path.name}    Output: {self.output_path.name}"
        )

        self.fig.canvas.draw_idle()


# ── CLI entry ───────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive point cloud transform editor with 3D preview."
    )
    parser.add_argument("input", type=Path, help="Input .pcd or .ply file")
    parser.add_argument("output", type=Path, help="Output .pcd or .ply file")
    args = parser.parse_args(argv)

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.exists():
        parser.error(f"File not found: {inp}")

    suffix = inp.suffix.lower()
    if suffix not in (".pcd", ".ply"):
        parser.error("Only .pcd and .ply are supported.")
    if out.suffix.lower() != suffix:
        parser.error("Input and output extensions must match.")

    print(f"Loading {inp.name} …")
    pts = extract_xyz_pcd(inp) if suffix == ".pcd" else extract_xyz_ply(inp)
    print(f"  {len(pts):,} points loaded.  Opening editor …")

    editor = Editor(inp, out, pts)
    if not editor.saved:
        print("Closed without saving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
