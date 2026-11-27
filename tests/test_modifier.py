"""
Unit tests for Natural Language Design Modifier.
"""
import pytest
from modules.universal_parser import UniversalSpecificationParser
from modules.design_modifier import NaturalLanguageDesignModifier

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def modifier():
    return NaturalLanguageDesignModifier()

def test_modify_material_and_color(parser, modifier):
    spec = parser.parse("Create a gear with 36 teeth, 100 mm outer diameter, 25 mm bore in steel.")
    assert spec["material"]["base_material"] == "steel"

    # Request change to aluminum
    updated, summary = modifier.modify(spec, "Change the material to aluminium")
    assert updated["material"]["base_material"] == "aluminum"
    assert updated["dimensions"]["teeth_count"] == 36 # Unchanged preserved
    assert updated["dimensions"]["outer_diameter"] == 100.0 # Unchanged preserved
    assert "Aluminum" in summary

    # Request change color to black
    updated2, summary2 = modifier.modify(updated, "Change the colour to black")
    assert "black" in updated2["color"].lower()
    assert updated2["material"]["base_material"] == "aluminum" # Previous change preserved

def test_modify_holes_and_scale(parser, modifier):
    spec = parser.parse("Industrial flange coupling with 120 mm outer diameter.")
    updated, summary = modifier.modify(spec, "Add four mounting holes and make it larger")

    assert updated["dimensions"]["mounting_holes_count"] == 4
    assert updated["dimensions"]["outer_diameter"] > 120.0
    assert any("mounting holes" in f for f in updated["features"])

def test_modify_viewing_angle(parser, modifier):
    spec = parser.parse("Centrifugal water pump.")
    updated, summary = modifier.modify(spec, "Show the product from the front")
    assert "front" in updated["viewing_angle"].lower()
