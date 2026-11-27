"""
Universal Parametric Geometry Engine for Industrial Products.
Supports Circles, Rectangles, Polygons, Holes, Rings, Teeth, Cylinders,
Flanges, Slots, Curves, Multi-component Assemblies, and Custom shapes.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

class GeometryPrimitive:
    """Base class for parametric primitives."""
    def __init__(self, primitive_type: str, color: Tuple[int, int, int] = (255, 255, 255), width: int = 2):
        self.primitive_type = primitive_type
        self.color = color
        self.width = width

class CirclePrimitive(GeometryPrimitive):
    def __init__(self, cx: float, cy: float, radius: float, fill: bool = False, **kwargs):
        super().__init__("circle", **kwargs)
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.fill = fill

class RectanglePrimitive(GeometryPrimitive):
    def __init__(self, x: float, y: float, w: float, h: float, corner_radius: float = 0, fill: bool = False, **kwargs):
        super().__init__("rectangle", **kwargs)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.corner_radius = corner_radius
        self.fill = fill

class PolygonPrimitive(GeometryPrimitive):
    def __init__(self, points: List[Tuple[float, float]], fill: bool = False, **kwargs):
        super().__init__("polygon", **kwargs)
        self.points = points
        self.fill = fill

class TeethPrimitive(GeometryPrimitive):
    """Involute or radial gear teeth profile."""
    def __init__(self, cx: float, cy: float, outer_r: float, root_r: float, teeth_count: int, **kwargs):
        super().__init__("teeth", **kwargs)
        self.cx = cx
        self.cy = cy
        self.outer_r = outer_r
        self.root_r = root_r
        self.teeth_count = max(4, teeth_count)

class HolePatternPrimitive(GeometryPrimitive):
    """Bolt circle or linear hole pattern."""
    def __init__(self, cx: float, cy: float, bolt_circle_r: float, hole_r: float, hole_count: int, **kwargs):
        super().__init__("hole_pattern", **kwargs)
        self.cx = cx
        self.cy = cy
        self.bolt_circle_r = bolt_circle_r
        self.hole_r = hole_r
        self.hole_count = hole_count

class ArcCurvePrimitive(GeometryPrimitive):
    def __init__(self, cx: float, cy: float, r: float, start_deg: float, end_deg: float, **kwargs):
        super().__init__("arc", **kwargs)
        self.cx = cx
        self.cy = cy
        self.r = r
        self.start_deg = start_deg
        self.end_deg = end_deg

class DimensionAnnotation:
    def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float], text: str, offset: float = 20.0):
        self.p1 = p1
        self.p2 = p2
        self.text = text
        self.offset = offset


class ParametricProductModel:
    """Parametric industrial component model container."""
    def __init__(self, product_name: str, canvas_size: Tuple[int, int] = (512, 512)):
        self.product_name = product_name
        self.width, self.height = canvas_size
        self.primitives: List[GeometryPrimitive] = []
        self.annotations: List[DimensionAnnotation] = []
        self.metadata: Dict[str, Any] = {}

    def add(self, prim: GeometryPrimitive):
        self.primitives.append(prim)

    def add_annotation(self, anno: DimensionAnnotation):
        self.annotations.append(anno)


class UniversalGeometryEngine:
    """Builds parametric geometric models from arbitrary product specifications."""

    def __init__(self, canvas_size: Tuple[int, int] = (512, 512)):
        self.canvas_size = canvas_size

    def build_geometry(self, spec: Dict[str, Any]) -> ParametricProductModel:
        """Universal geometry builder. Dispatches to specific geometry builders or generalized layout."""
        product_name = spec.get("product_name", "").lower()
        category = spec.get("category", "").lower()
        geom = spec.get("primary_geometry", {})
        shape_type = geom.get("shape_type", "").lower()
        model = ParametricProductModel(product_name, self.canvas_size)

        # Dispatch based on specific product knowledge if available
        if "gear" in product_name or "sprocket" in product_name:
            self._build_gear_geometry(model, spec)
        elif "bearing" in product_name:
            self._build_bearing_geometry(model, spec)
        elif "pump" in product_name:
            self._build_pump_geometry(model, spec)
        elif "enclosure" in product_name or "box" in product_name or "cabinet" in product_name:
            self._build_enclosure_geometry(model, spec)
        elif "flange" in product_name:
            self._build_flange_geometry(model, spec)
        elif "valve" in product_name:
            self._build_valve_geometry(model, spec)
        elif "shaft" in product_name or "cylinder" in shape_type:
            self._build_shaft_cylinder_geometry(model, spec)
        else:
            # Unfamiliar product: Universal Generalized Shape Synthesizer
            self._build_universal_generalized_geometry(model, spec)

        return model

    def _build_gear_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        dims = spec.get("dimensions", {})
        teeth = dims.get("teeth_count") or 24
        outer_r = 170.0
        root_r = 135.0
        bore_r = 45.0

        # Gear teeth
        model.add(TeethPrimitive(cx, cy, outer_r, root_r, teeth, width=2))
        # Root circle
        model.add(CirclePrimitive(cx, cy, root_r, width=1))
        # Pitch circle (dashed concept rendered as thin circle)
        model.add(CirclePrimitive(cx, cy, (outer_r + root_r) / 2, width=1))
        # Central hub circle
        model.add(CirclePrimitive(cx, cy, 75.0, width=2))
        # Central bore hole
        model.add(CirclePrimitive(cx, cy, bore_r, fill=False, width=2))
        # Keyway slot
        kw_w, kw_h = 16.0, 10.0
        model.add(RectanglePrimitive(cx - kw_w/2, cy - bore_r - kw_h + 2, kw_w, kw_h, width=2))

        # Weight reduction cutouts
        for i in range(4):
            angle = i * (math.pi / 2) + (math.pi / 4)
            wx = cx + 105.0 * math.cos(angle)
            wy = cy + 105.0 * math.sin(angle)
            model.add(CirclePrimitive(wx, wy, 18.0, width=1))

        # Dimension annotations
        od_val = dims.get("outer_diameter") or 100
        bore_val = dims.get("central_bore") or 25
        model.add_annotation(DimensionAnnotation((cx - outer_r, cy), (cx + outer_r, cy), f"OD: {od_val} mm", offset=30))
        model.add_annotation(DimensionAnnotation((cx - bore_r, cy), (cx + bore_r, cy), f"Bore: {bore_val} mm", offset=-25))
        model.add_annotation(DimensionAnnotation((cx, cy - outer_r), (cx, cy - root_r), f"{teeth} Teeth", offset=30))

    def _build_bearing_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        outer_r = 180.0
        inner_r = 100.0
        ball_orbit_r = (outer_r + inner_r) / 2
        ball_r = 26.0

        # Outer ring
        model.add(CirclePrimitive(cx, cy, outer_r, width=3))
        model.add(CirclePrimitive(cx, cy, outer_r - 20, width=2))
        # Inner ring
        model.add(CirclePrimitive(cx, cy, inner_r, width=2))
        model.add(CirclePrimitive(cx, cy, inner_r - 35, width=3))
        # Rolling elements (balls) and cage separators
        ball_count = 8
        for i in range(ball_count):
            angle = i * (2 * math.pi / ball_count)
            bx = cx + ball_orbit_r * math.cos(angle)
            by = cy + ball_orbit_r * math.sin(angle)
            model.add(CirclePrimitive(bx, by, ball_r, width=2))
            # Cage connector
            next_angle = (i + 1) * (2 * math.pi / ball_count)
            mx = cx + ball_orbit_r * math.cos((angle + next_angle)/2)
            my = cy + ball_orbit_r * math.sin((angle + next_angle)/2)
            model.add(CirclePrimitive(mx, my, 4.0, fill=True, width=1))

        model.add_annotation(DimensionAnnotation((cx - outer_r, cy), (cx + outer_r, cy), "OD 120 mm", offset=30))
        model.add_annotation(DimensionAnnotation((cx - (inner_r - 35), cy), (cx + (inner_r - 35), cy), "ID 45 mm", offset=-30))

    def _build_pump_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2 + 10, self.canvas_size[1] / 2 + 15
        dims = spec.get("dimensions", {})
        holes = dims.get("mounting_holes_count") or 4

        # Volute spiral casing (approximated via compound arcs and offsets)
        volute_pts = []
        for deg in range(0, 360, 10):
            rad = math.radians(deg)
            r = 80.0 + (deg / 360.0) * 85.0
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            volute_pts.append((x, y))
        model.add(PolygonPrimitive(volute_pts, width=3))

        # Circular suction inlet (center)
        model.add(CirclePrimitive(cx, cy, 55.0, width=2))
        model.add(CirclePrimitive(cx, cy, 38.0, width=2))
        # Suction inlet bolt holes
        model.add(HolePatternPrimitive(cx, cy, 47.0, 4.0, 4, width=1))

        # Tangential discharge outlet nozzle (top right)
        nx, ny = cx + 80, cy - 130
        nw, nh = 65, 80
        model.add(RectanglePrimitive(nx, ny, nw, nh, width=2))
        model.add(RectanglePrimitive(nx - 10, ny, nw + 20, 18, width=3))  # Discharge flange
        # Discharge flange holes
        model.add(CirclePrimitive(nx - 4, ny + 9, 3.5, width=1))
        model.add(CirclePrimitive(nx + nw + 4, ny + 9, 3.5, width=1))

        # Mounting foot base (bottom)
        bx, by = cx - 120, cy + 130
        bw, bh = 240, 32
        model.add(RectanglePrimitive(bx, by, bw, bh, corner_radius=6, width=3))
        # Base mounting holes
        model.add(CirclePrimitive(bx + 25, by + 16, 6.0, width=2))
        model.add(CirclePrimitive(bx + bw - 25, by + 16, 6.0, width=2))

        model.add_annotation(DimensionAnnotation((cx, cy), (nx + nw/2, ny), "Discharge Port", offset=20))
        model.add_annotation(DimensionAnnotation((bx, by + bh), (bx + bw, by + bh), "Mounting Base", offset=20))

    def _build_enclosure_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        w, h = 300.0, 360.0
        x, y = cx - w/2, cy - h/2

        # Outer enclosure cabinet body
        model.add(RectanglePrimitive(x, y, w, h, corner_radius=12, width=3))
        # Door seam gasket line
        model.add(RectanglePrimitive(x + 18, y + 18, w - 36, h - 36, corner_radius=8, width=2))
        # Hinges (left side)
        model.add(RectanglePrimitive(x - 6, y + 60, 12, 35, corner_radius=3, width=2))
        model.add(RectanglePrimitive(x - 6, y + h - 95, 12, 35, corner_radius=3, width=2))
        # Quarter-turn latch lock (right side)
        model.add(CirclePrimitive(x + w - 35, cy, 14.0, width=2))
        model.add(RectanglePrimitive(x + w - 42, cy - 4, 14, 8, width=1))
        # Conduit knockouts (bottom plate)
        for i in range(3):
            kx = x + 60 + i * 90
            ky = y + h - 10
            model.add(CirclePrimitive(kx, ky, 12.0, width=1))
        # External wall mounting brackets (4 corners)
        model.add(RectanglePrimitive(x - 22, y - 10, 22, 28, width=2))
        model.add(CirclePrimitive(x - 11, y + 4, 4.0, width=1))
        model.add(RectanglePrimitive(x + w, y - 10, 22, 28, width=2))
        model.add(CirclePrimitive(x + w + 11, y + 4, 4.0, width=1))
        model.add(RectanglePrimitive(x - 22, y + h - 18, 22, 28, width=2))
        model.add(CirclePrimitive(x - 11, y + h - 4, 4.0, width=1))
        model.add(RectanglePrimitive(x + w, y + h - 18, 22, 28, width=2))
        model.add(CirclePrimitive(x + w + 11, y + h - 4, 4.0, width=1))

        model.add_annotation(DimensionAnnotation((x, y - 15), (x + w, y - 15), "300 mm Width", offset=15))
        model.add_annotation(DimensionAnnotation((x - 30, y), (x - 30, y + h), "360 mm Height", offset=-15))

    def _build_flange_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        dims = spec.get("dimensions", {})
        outer_r = 180.0
        bore_r = 55.0
        bolt_circle_r = 135.0
        bolt_count = dims.get("mounting_holes_count") or 8

        # Outer flange rim
        model.add(CirclePrimitive(cx, cy, outer_r, width=3))
        # Raised sealing face
        model.add(CirclePrimitive(cx, cy, 100.0, width=2))
        # Central pipe bore
        model.add(CirclePrimitive(cx, cy, bore_r, width=3))
        # Bolt hole pattern
        model.add(HolePatternPrimitive(cx, cy, bolt_circle_r, 9.0, bolt_count, width=2))

        model.add_annotation(DimensionAnnotation((cx - outer_r, cy), (cx + outer_r, cy), "Flange OD", offset=30))
        model.add_annotation(DimensionAnnotation((cx - bore_r, cy), (cx + bore_r, cy), "Bore ID", offset=-25))

    def _build_valve_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2 + 30
        # Central valve body (spherical / cast chamber)
        model.add(CirclePrimitive(cx, cy, 70.0, width=3))
        # Left flange neck & flange
        model.add(RectanglePrimitive(cx - 150, cy - 35, 80, 70, width=2))
        model.add(RectanglePrimitive(cx - 170, cy - 55, 20, 110, width=3))
        model.add(CirclePrimitive(cx - 160, cy - 35, 5.0, width=1))
        model.add(CirclePrimitive(cx - 160, cy + 35, 5.0, width=1))
        # Right flange neck & flange
        model.add(RectanglePrimitive(cx + 70, cy - 35, 80, 70, width=2))
        model.add(RectanglePrimitive(cx + 150, cy - 55, 20, 110, width=3))
        model.add(CirclePrimitive(cx + 160, cy - 35, 5.0, width=1))
        model.add(CirclePrimitive(cx + 160, cy + 35, 5.0, width=1))
        # Bonnet / stem neck (top)
        model.add(RectanglePrimitive(cx - 20, cy - 130, 40, 65, width=2))
        # Actuation Handwheel / lever (top)
        model.add(CirclePrimitive(cx, cy - 145, 45.0, width=3))
        model.add(CirclePrimitive(cx, cy - 145, 12.0, width=2))
        for deg in [0, 90, 180, 270]:
            rad = math.radians(deg)
            model.add(PolygonPrimitive([
                (cx, cy - 145),
                (cx + 45 * math.cos(rad), cy - 145 + 45 * math.sin(rad))
            ], width=2))

    def _build_shaft_cylinder_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        # Stepped industrial shaft
        # Section 1: drive end
        model.add(RectanglePrimitive(cx - 210, cy - 25, 90, 50, corner_radius=2, width=2))
        model.add(RectanglePrimitive(cx - 190, cy - 10, 50, 20, width=1)) # Keyway
        # Section 2: bearing seat
        model.add(RectanglePrimitive(cx - 120, cy - 40, 70, 80, corner_radius=2, width=2))
        # Section 3: main larger diameter center
        model.add(RectanglePrimitive(cx - 50, cy - 60, 140, 120, corner_radius=3, width=3))
        # Section 4: second bearing seat
        model.add(RectanglePrimitive(cx + 90, cy - 40, 70, 80, corner_radius=2, width=2))
        # Section 5: threaded output end
        model.add(RectanglePrimitive(cx + 160, cy - 20, 55, 40, corner_radius=2, width=2))
        for tx in range(int(cx + 165), int(cx + 210), 8):
            model.add(PolygonPrimitive([(tx, cy - 20), (tx, cy + 20)], width=1))

    def _build_universal_generalized_geometry(self, model: ParametricProductModel, spec: Dict[str, Any]):
        """Generalized layout generator for unfamiliar/arbitrary industrial components.
        Synthesizes a cohesive engineering representation using parsed dimensions, shapes, and features."""
        cx, cy = self.canvas_size[0] / 2, self.canvas_size[1] / 2
        geom = spec.get("primary_geometry", {})
        shape_type = geom.get("shape_type", "compound")
        dims = spec.get("dimensions", {})
        features = spec.get("features", [])

        # Determine main envelope from dimensions or defaults
        od = dims.get("outer_diameter")
        w = dims.get("width") or dims.get("length") or (od * 1.5 if od else None) or 260.0
        h = dims.get("height") or dims.get("thickness") or (od if od else None) or 200.0
        # scale to fit canvas comfortably
        max_dim = max(w, h, 1)
        scale = min(360.0 / max_dim, 1.2)
        sw, sh = w * scale, h * scale

        if "circular" in shape_type or "radial" in geom.get("symmetry", ""):
            r = min(sw, sh) / 2
            model.add(CirclePrimitive(cx, cy, r, width=3))
            model.add(CirclePrimitive(cx, cy, r * 0.7, width=2))
            model.add(CirclePrimitive(cx, cy, r * 0.35, width=2))
            # Radial features
            for i in range(6):
                ang = i * (math.pi / 3)
                fx = cx + (r * 0.7) * math.cos(ang)
                fy = cy + (r * 0.7) * math.sin(ang)
                model.add(CirclePrimitive(fx, fy, 6.0, width=1))
        else:
            # Rectangular / prism / compound assembly envelope
            x, y = cx - sw/2, cy - sh/2
            model.add(RectanglePrimitive(x, y, sw, sh, corner_radius=8, width=3))
            # Internal sub-assembly lines
            model.add(RectanglePrimitive(x + 15, y + 15, sw - 30, sh - 30, corner_radius=4, width=2))
            # Central functional interface
            model.add(CirclePrimitive(cx, cy, min(sw, sh) * 0.22, width=2))
            model.add(CirclePrimitive(cx, cy, min(sw, sh) * 0.12, width=1))
            # Corner mounting features
            corner_offset = 25
            for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                model.add(CirclePrimitive(cx + ox * (sw/2 - corner_offset), cy + oy * (sh/2 - corner_offset), 7.0, width=2))

        # Add engineering dimension annotations
        model.add_annotation(DimensionAnnotation((cx - sw/2, cy - sh/2 - 20), (cx + sw/2, cy - sh/2 - 20), f"W: {int(w)} mm", offset=15))
        model.add_annotation(DimensionAnnotation((cx - sw/2 - 20, cy - sh/2), (cx - sw/2 - 20, cy + sh/2), f"H: {int(h)} mm", offset=-15))
