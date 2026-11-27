"""
Unit tests for Image Evaluator.
"""
import pytest
from PIL import Image
from modules.universal_parser import UniversalSpecificationParser
from modules.prompt_engine import UniversalPromptEngine
from modules.image_generator import UniversalImageGenerator
from modules.evaluator import UniversalEvaluator

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def prompt_engine():
    return UniversalPromptEngine()

@pytest.fixture
def generator():
    return UniversalImageGenerator()

@pytest.fixture
def evaluator():
    return UniversalEvaluator()

def test_evaluation_pipeline(parser, prompt_engine, generator, evaluator):
    spec = parser.parse("Centrifugal water pump with stainless steel body and circular inlet.")
    prompts = prompt_engine.generate(spec)
    gens = generator.generate(spec, prompts, seed=42, num_candidates=1)
    img = gens[0]["image"]

    eval_result = evaluator.evaluate(img, spec, prompts, human_rating=4)

    assert "clip_score" in eval_result
    assert 0.0 <= eval_result["clip_score"] <= 1.0
    assert "feature_checklist" in eval_result
    assert len(eval_result["feature_checklist"]) > 0
    assert eval_result["human_rating"] == 4
    assert "CRITICAL ENGINEERING NOTICE" in eval_result["dimensional_accuracy_disclaimer"]

def test_ssim_computation(evaluator):
    imgA = Image.new("RGB", (256, 256), (128, 128, 128))
    imgB = Image.new("RGB", (256, 256), (130, 130, 130))
    ssim_val = evaluator._compute_ssim(imgA, imgB)
    assert 0.9 <= ssim_val <= 1.0
