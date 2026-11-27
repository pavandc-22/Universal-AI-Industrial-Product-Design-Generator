"""
Parametric Outline and Edge Map Renderer for ControlNet and CAD Visualizations.
Uses Pillow, OpenCV, and SVG generation.
"""

import math
import io
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from .geometry_engine import (
    ParametricProductModel,
    CirclePrimitive,
    RectanglePrimitive,
    PolygonPrimitive,
    TeethPrimitive,
    HolePatternPrimitive,
    ArcCurvePrimitive,
    DimensionAnnotation
)

class OutlineRenderer:
    """Renders parametric geometric models into ControlNet-ready edge maps and CAD drawings."""

    def __init__(self, default_size: Tuple[int, int] = (512, 512)):
        self.default_size = default_size

    def render_controlnet_edge_map(self, model: ParametricProductModel, invert: bool = False) -> Image.Image:
        """Generates clean high-contrast structural outline image suitable for ControlNet Canny/LineArt."""
        width, height = model.width, model.height
        # Black background for ControlNet edge map by default (white lines on black)
        bg_color = (255, 255, 255) if invert else (0, 0, 0)
        line_color = (0, 0, 0) if invert else (255, 255, 255)

        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        for prim in model.primitives:
            w = max(1, prim.width)
            if isinstance(prim, CirclePrimitive):
                bbox = [prim.cx - prim.radius, prim.cy - prim.radius, prim.cx + prim.radius, prim.cy + prim.radius]
                if prim.fill:
                    draw.ellipse(bbox, fill=line_color)
                else:
                    draw.ellipse(bbox, outline=line_color, width=w)

            elif isinstance(prim, RectanglePrimitive):
                bbox = [prim.x, prim.y, prim.x + prim.w, prim.y + prim.h]
                if prim.corner_radius > 0:
                    draw.rounded_rectangle(bbox, radius=int(prim.corner_radius), outline=line_color, width=w)
                else:
                    draw.rectangle(bbox, outline=line_color, width=w)

            elif isinstance(prim, PolygonPrimitive):
                if len(prim.points) > 1:
                    draw.polygon(prim.points, outline=line_color, fill=line_color if prim.fill else None)

            elif isinstance(prim, TeethPrimitive):
                points = self._generate_gear_teeth_points(prim.cx, prim.cy, prim.outer_r, prim.root_r, prim.teeth_count)
                draw.polygon(points, outline=line_color)

            elif isinstance(prim, HolePatternPrimitive):
                for i in range(prim.hole_count):
                    ang = i * (2 * math.pi / prim.hole_count)
                    hx = prim.cx + prim.bolt_circle_r * math.cos(ang)
                    hy = prim.cy + prim.bolt_circle_r * math.sin(ang)
                    draw.ellipse([hx - prim.hole_r, hy - prim.hole_r, hx + prim.hole_r, hy + prim.hole_r], outline=line_color, width=w)

            elif isinstance(prim, ArcCurvePrimitive):
                bbox = [prim.cx - prim.r, prim.cy - prim.r, prim.cx + prim.r, prim.cy + prim.r]
                draw.arc(bbox, start=prim.start_deg, end=prim.end_deg, fill=line_color, width=w)

        return img

    def render_cad_drawing(self, model: ParametricProductModel) -> Image.Image:
        """Renders technical engineering CAD schematic drawing with dimension lines and annotations."""
        width, height = model.width, model.height
        # Engineering dark slate blue background
        img = Image.new("RGB", (width, height), (15, 23, 42))
        draw = ImageDraw.Draw(img)

        # Subtle CAD Grid Lines
        grid_color = (24, 38, 64)
        for gx in range(0, width, 32):
            draw.line([(gx, 0), (gx, height)], fill=grid_color, width=1)
        for gy in range(0, height, 32):
            draw.line([(0, gy), (width, gy)], fill=grid_color, width=1)

        cad_line_color = (56, 189, 248) # Cyan CAD line
        centerline_color = (239, 68, 68) # Red dashed centerline

        # Render geometric primitives
        for prim in model.primitives:
            w = max(1, prim.width)
            if isinstance(prim, CirclePrimitive):
                bbox = [prim.cx - prim.radius, prim.cy - prim.radius, prim.cx + prim.radius, prim.cy + prim.radius]
                draw.ellipse(bbox, outline=cad_line_color, width=w)
            elif isinstance(prim, RectanglePrimitive):
                bbox = [prim.x, prim.y, prim.x + prim.w, prim.y + prim.h]
                if prim.corner_radius > 0:
                    draw.rounded_rectangle(bbox, radius=int(prim.corner_radius), outline=cad_line_color, width=w)
                else:
                    draw.rectangle(bbox, outline=cad_line_color, width=w)
            elif isinstance(prim, PolygonPrimitive):
                if len(prim.points) > 1:
                    draw.polygon(prim.points, outline=cad_line_color)
            elif isinstance(prim, TeethPrimitive):
                points = self._generate_gear_teeth_points(prim.cx, prim.cy, prim.outer_r, prim.root_r, prim.teeth_count)
                draw.polygon(points, outline=cad_line_color)
            elif isinstance(prim, HolePatternPrimitive):
                for i in range(prim.hole_count):
                    ang = i * (2 * math.pi / prim.hole_count)
                    hx = prim.cx + prim.bolt_circle_r * math.cos(ang)
                    hy = prim.cy + prim.bolt_circle_r * math.sin(ang)
                    draw.ellipse([hx - prim.hole_r, hy - prim.hole_r, hx + prim.hole_r, hy + prim.hole_r], outline=cad_line_color, width=w)
            elif isinstance(prim, ArcCurvePrimitive):
                bbox = [prim.cx - prim.r, prim.cy - prim.r, prim.cx + prim.r, prim.cy + prim.r]
                draw.arc(bbox, start=prim.start_deg, end=prim.end_deg, fill=cad_line_color, width=w)

        # Centerlines
        cx, cy = width / 2, height / 2
        draw.line([(cx - 210, cy), (cx + 210, cy)], fill=centerline_color, width=1)
        draw.line([(cx, cy - 210), (cx, cy + 210)], fill=centerline_color, width=1)

        # Render Dimension Annotations
        anno_color = (250, 204, 21) # Yellow dimension lines
        for anno in model.annotations:
            (x1, y1), (x2, y2) = anno.p1, anno.p2
            # Offset dimension line
            dx = -(y2 - y1)
            dy = (x2 - x1)
            length = math.hypot(dx, dy)
            if length > 0:
                ux, uy = (dx / length) * anno.offset, (dy / length) * anno.offset
                ax1, ay1 = x1 + ux, y1 + uy
                ax2, ay2 = x2 + ux, y2 + uy
                # Leader lines
                draw.line([(x1, y1), (ax1, ay1)], fill=anno_color, width=1)
                draw.line([(x2, y2), (ax2, ay2)], fill=anno_color, width=1)
                # Main dimension line
                draw.line([(ax1, ay1), (ax2, ay2)], fill=anno_color, width=1)
                # Label text
                mid_x, mid_y = (ax1 + ax2) / 2, (ay1 + ay2) / 2
                draw.text((mid_x - 30, mid_y - 12), anno.text, fill=anno_color)

        return img

    def extract_canny_from_image(self, image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
        """Extracts Canny edge map from a reference PIL image using OpenCV."""
        np_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return Image.fromarray(edges).convert("RGB")

    def to_svg(self, model: ParametricProductModel) -> str:
        """Exports parametric model to clean W3C SVG vector format."""
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {model.width} {model.height}" width="{model.width}" height="{model.height}">',
            f'  <rect width="100%" height="100%" fill="#0f172a" />'
        ]

        for prim in model.primitives:
            w = max(1, prim.width)
            if isinstance(prim, CirclePrimitive):
                fill = "white" if prim.fill else "none"
                svg_lines.append(f'  <circle cx="{prim.cx}" cy="{prim.cy}" r="{prim.radius}" stroke="#38bdf8" stroke-width="{w}" fill="{fill}" />')
            elif isinstance(prim, RectanglePrimitive):
                rx_attr = f'rx="{prim.corner_radius}"' if prim.corner_radius > 0 else ''
                svg_lines.append(f'  <rect x="{prim.x}" y="{prim.y}" width="{prim.w}" height="{prim.h}" {rx_attr} stroke="#38bdf8" stroke-width="{w}" fill="none" />')
            elif isinstance(prim, PolygonPrimitive):
                pts = " ".join(f"{x},{y}" for x, y in prim.points)
                svg_lines.append(f'  <polygon points="{pts}" stroke="#38bdf8" stroke-width="{w}" fill="none" />')
            elif isinstance(prim, TeethPrimitive):
                points = self._generate_gear_teeth_points(prim.cx, prim.cy, prim.outer_r, prim.root_r, prim.teeth_count)
                pts = " ".join(f"{x},{y}" for x, y in points)
                svg_lines.append(f'  <polygon points="{pts}" stroke="#38bdf8" stroke-width="{w}" fill="none" />')
            elif isinstance(prim, HolePatternPrimitive):
                for i in range(prim.hole_count):
                    ang = i * (2 * math.pi / prim.hole_count)
                    hx = prim.cx + prim.bolt_circle_r * math.cos(ang)
                    hy = prim.cy + prim.bolt_circle_r * math.sin(ang)
                    svg_lines.append(f'  <circle cx="{hx}" cy="{hy}" r="{prim.hole_r}" stroke="#38bdf8" stroke-width="{w}" fill="none" />')

        svg_lines.append('</svg>')
        return "\n".join(svg_lines)

    def _generate_gear_teeth_points(self, cx: float, cy: float, outer_r: float, root_r: float, teeth: int) -> List[Tuple[float, float]]:
        points = []
        d_theta = (2 * math.pi) / teeth
        for i in range(teeth):
            t0 = i * d_theta
            t1 = t0 + d_theta * 0.25
            t2 = t0 + d_theta * 0.50
            t3 = t0 + d_theta * 0.75

            # Root point 1
            points.append((cx + root_r * math.cos(t0), cy + root_r * math.sin(t0)))
            # Flank up to Outer Tip 1
            points.append((cx + outer_r * math.cos(t1), cy + outer_r * math.sin(t1)))
            # Outer Tip 2
            points.append((cx + outer_r * math.cos(t2), cy + outer_r * math.sin(t2)))
            # Flank down to Root point 2
            points.append((cx + root_r * math.cos(t3), cy + root_r * math.sin(t3)))
        return points
