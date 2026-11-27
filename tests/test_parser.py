"""
Unit tests for Universal Specification Parser.
"""
import pytest
from modules.universal_parser import UniversalSpecificationParser

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

def test_parse_pump_specification(parser):
    text = (
        "Generate an industrial centrifugal water pump with a stainless steel body, "
        "circular inlet, side outlet, four mounting holes, compact design and a professional product catalogue appearance."
    )
    spec = parser.parse(text)
    assert spec is not None
    assert "pump" in spec["product_name"].lower() or "centrifugal" in spec["product_name"].lower()
    assert spec["material"]["base_material"] == "stainless steel"
    assert spec["dimensions"]["mounting_holes_count"] == 4
    assert any("inlet" in f.lower() for f in spec["features"])
    assert any("outlet" in f.lower() for f in spec["features"])
    assert "catalogue" in spec["required_appearance"].lower()
    assert spec["parsing_source"] == "local_rule_fallback"
    assert parser.validate_spec(spec) is True

def test_parse_gear_specification(parser):
    text = (
        "Create a custom mechanical gear with 36 teeth, 100 mm outer diameter, "
        "25 mm central bore and hardened steel finish."
    )
    spec = parser.parse(text)
    assert spec is not None
    assert "gear" in spec["product_name"].lower()
    assert spec["dimensions"]["teeth_count"] == 36
    assert spec["dimensions"]["outer_diameter"] == 100.0
    assert spec["dimensions"]["central_bore"] == 25.0
    assert "hardened steel" in spec["material"]["base_material"] or spec["material"]["treatment"] == "Hardened"
    assert parser.validate_spec(spec) is True

def test_unfamiliar_product_parsing(parser):
    text = (
        "Manufacture an aerospace titanium micro-nozzle manifold with 12 radial spray orifices, "
        "electropolished finish, 45 mm diameter, and high pressure rating."
    )
    spec = parser.parse(text)
    assert spec is not None
    assert "titanium" in spec["material"]["base_material"]
    assert spec["dimensions"]["outer_diameter"] == 45.0
    assert parser.validate_spec(spec) is True

def test_missing_dimensions_detected_without_hallucination(parser):
    text = "Design a spur gear without specifying any dimensions."
    spec = parser.parse(text)
    assert spec["dimensions"]["outer_diameter"] is None
    assert spec["dimensions"]["teeth_count"] is None
    assert spec["clarification_needed"] is True
    assert len(spec["missing_information"]) > 0
