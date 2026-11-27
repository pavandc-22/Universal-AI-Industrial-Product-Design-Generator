"""
Unit tests for Universal Geometry Engine and Outline Renderer.
"""
import pytest
from PIL import Image
from modules.universal_parser import UniversalSpecificationParser
from modules.geometry_engine import UniversalGeometryEngine
from modules.outline_renderer import OutlineRenderer

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def geom_engine():
    return UniversalGeometryEngine()

@pytest.fixture
def renderer():
    return OutlineRenderer()

def test_gear_geometry_and_outline_render(parser, geom_engine, renderer):
    spec = parser.parse("Create a mechanical gear with 36 teeth, 100 mm outer diameter, 25 mm bore.")
    model = geom_engine.build_geometry(spec)
    assert len(model.primitives) > 0

    edge_map = renderer.render_controlnet_edge_map(model)
    assert isinstance(edge_map, Image.Image)
    assert edge_map.size == (512, 512)

    cad_img = renderer.render_cad_drawing(model)
    assert isinstance(cad_img, Image.Image)
    assert cad_img.size == (512, 512)

    svg = renderer.to_svg(model)
    assert "<svg" in svg
    assert "</svg>" in svg

def test_pump_geometry_generation(parser, geom_engine, renderer):
    spec = parser.parse("Centrifugal water pump with circular inlet, side outlet, four mounting holes.")
    model = geom_engine.build_geometry(spec)
    edge_map = renderer.render_controlnet_edge_map(model)
    assert edge_map.size == (512, 512)

def test_unfamiliar_product_geometry(parser, geom_engine, renderer):
    spec = parser.parse("Custom industrial robotic gripper claw mechanism with titanium finish.")
    model = geom_engine.build_geometry(spec)
    assert len(model.primitives) > 0
    edge_map = renderer.render_controlnet_edge_map(model)
    assert edge_map is not None

def test_canny_extraction(renderer):
    test_img = Image.new("RGB", (200, 200), (0, 0, 0))
    canny = renderer.extract_canny_from_image(test_img)
    assert canny.size == (200, 200)
