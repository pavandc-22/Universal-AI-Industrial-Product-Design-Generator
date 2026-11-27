"""
Unit tests for Universal Image Generator.
"""
import pytest
from pathlib import Path
from modules.universal_parser import UniversalSpecificationParser
from modules.prompt_engine import UniversalPromptEngine
from modules.image_generator import UniversalImageGenerator

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def prompt_engine():
    return UniversalPromptEngine()

@pytest.fixture
def generator():
    return UniversalImageGenerator()

def test_generate_gear_candidates(parser, prompt_engine, generator):
    spec = parser.parse("Create a custom mechanical gear with 36 teeth, 100 mm outer diameter, 25 mm central bore.")
    prompts = prompt_engine.generate(spec)

    results = generator.generate(
        spec=spec,
        prompts=prompts,
        seed=101,
        num_candidates=2,
        condition="Condition_C_LLM_ControlNet"
    )

    assert len(results) == 2
    for r in results:
        assert "image" in r
        assert "path" in r
        assert Path(r["path"]).exists()
        assert r["image"].size == (512, 512)
        assert r["condition"] == "Condition_C_LLM_ControlNet"
        assert r["is_demonstration_mode"] is True

def test_generate_pump_candidates(parser, prompt_engine, generator):
    spec = parser.parse("Industrial centrifugal water pump with stainless steel body and circular inlet.")
    prompts = prompt_engine.generate(spec)

    results = generator.generate(
        spec=spec,
        prompts=prompts,
        seed=42,
        num_candidates=1,
        condition="Condition_B_LLM_Prompt"
    )

    assert len(results) == 1
    assert results[0]["image"].size == (512, 512)
