"""
Unit tests for Automatic Refinement Loop.
"""
import pytest
from modules.universal_parser import UniversalSpecificationParser
from modules.refinement import RefinementEngine

@pytest.fixture
def parser():
    return UniversalSpecificationParser(provider="local")

@pytest.fixture
def refinement_engine():
    return RefinementEngine()

def test_refinement_loop_execution(parser, refinement_engine):
    spec = parser.parse("Create an industrial spur gear with 36 teeth and hardened steel finish.")
    res = refinement_engine.run_refinement_loop(
        spec=spec,
        max_rounds=2,
        target_score=85.0,
        seed=42
    )

    assert "rounds_completed" in res
    assert 1 <= res["rounds_completed"] <= 2
    assert "history" in res
    assert len(res["history"]) == res["rounds_completed"]
    assert "latest_evaluation" in res
    assert "overall_design_fidelity" in res["latest_evaluation"]
