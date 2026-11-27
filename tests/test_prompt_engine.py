"""
Unit tests for Universal Prompt Engineering Engine.
"""
import pytest
from modules.universal_parser import UniversalSpecificationParser
from modules.prompt_engine import UniversalPromptEngine

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def prompt_engine():
    return UniversalPromptEngine()

def test_prompt_generation_gear(parser, prompt_engine):
    text = "Create a custom mechanical gear with 36 teeth, 100 mm outer diameter, 25 mm central bore and hardened steel finish."
    spec = parser.parse(text)
    prompts = prompt_engine.generate(spec)

    assert "positive_prompt" in prompts
    assert "negative_prompt" in prompts
    assert "structural_description" in prompts
    assert "geometric_feature_description" in prompts
    assert "material_texture_description" in prompts
    assert "camera_lighting_instructions" in prompts

    pos = prompts["positive_prompt"]
    assert "Gear" in pos or "gear" in pos
    assert "36 teeth" in pos or "36" in prompts["geometric_feature_description"]
    assert "hardened steel" in pos or "hardened" in prompts["material_texture_description"].lower()
    assert "missing teeth" in prompts["negative_prompt"]

def test_prompt_generation_pump(parser, prompt_engine):
    text = "Generate an industrial centrifugal water pump with a stainless steel body, circular inlet, side outlet, four mounting holes."
    spec = parser.parse(text)
    prompts = prompt_engine.generate(spec)

    assert "Pump" in prompts["positive_prompt"] or "pump" in prompts["positive_prompt"]
    assert "stainless steel" in prompts["material_texture_description"].lower()
    assert "leaking fluid" in prompts["negative_prompt"]
