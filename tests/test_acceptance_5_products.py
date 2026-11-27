"""
Final Acceptance Test Suite for the 5 Mandatory Universal Product Types:
1. Deep-groove ball bearing
2. Mechanical gear
3. Industrial centrifugal pump
4. Custom electrical enclosure
5. Unfamiliar industrial product (not predefined in system)

Verifies independent interpretation, distinct geometry, prompts, image generation, and evaluation.
"""

import pytest
from pathlib import Path
from backend.api import IndustrialDesignPlatformService

@pytest.fixture
def service():
    return IndustrialDesignPlatformService()

FIVE_TEST_REQUESTS = [
    {
        "type": "bearing",
        "input": "Single-row deep groove ball bearing with 120 mm outer diameter, 45 mm inner bore, 20 mm width, hardened chrome steel construction and stamped steel cage with 8 precision ground steel balls.",
        "expected_name_keyword": "bearing",
        "expected_shape": "toroidal"
    },
    {
        "type": "gear",
        "input": "Create a custom mechanical gear with 36 teeth, 100 mm outer diameter, 25 mm central bore and hardened steel finish.",
        "expected_name_keyword": "gear",
        "expected_shape": "toothed circular disk"
    },
    {
        "type": "pump",
        "input": "Generate an industrial centrifugal water pump with a stainless steel body, circular inlet, side outlet, four mounting holes, compact design and a professional product catalogue appearance.",
        "expected_name_keyword": "pump",
        "expected_shape": "spiral volute with cylindrical housing"
    },
    {
        "type": "enclosure",
        "input": "Wall-mount industrial electrical enclosure box, 300 mm x 250 mm x 150 mm, 304 stainless steel with continuous polyurethane door gasket, quarter-turn key latch, external mounting feet.",
        "expected_name_keyword": "enclosure",
        "expected_shape": "rectangular prism"
    },
    {
        "type": "unfamiliar",
        "input": "Cryogenic liquid nitrogen vaporizing heat-sink module with micro-channel serpentine fins, vacuum jacket, 220 mm width, 140 mm height, electropolished titanium alloy finish.",
        "expected_name_keyword": "cryogenic",
        "expected_shape": "compound" # Unfamiliar custom shape
    }
]

@pytest.mark.parametrize("item", FIVE_TEST_REQUESTS)
def test_end_to_end_acceptance_per_product(service, item):
    user_input = item["input"]

    # 1. Universal Specification Parser
    spec = service.parse_specification(user_input)
    assert spec is not None
    assert item["expected_name_keyword"] in spec["product_name"].lower()
    assert spec["material"]["base_material"] is not None
    assert service.parser.validate_spec(spec) is True

    # 2. Universal Prompt Engineering
    prompts = service.generate_prompts(spec)
    assert "positive_prompt" in prompts
    assert "negative_prompt" in prompts
    assert item["expected_name_keyword"] in prompts["positive_prompt"].lower()

    # 3. Universal Geometry & Outlines
    outlines = service.generate_geometry_and_outlines(spec)
    assert outlines["model"] is not None
    assert outlines["edge_map"].size == (512, 512)
    assert outlines["cad_drawing"].size == (512, 512)
    assert "<svg" in outlines["svg_code"]

    # 4. Universal Image Generation
    results = service.generate_design(
        spec=spec,
        prompts=prompts,
        outline_model=outlines["model"],
        seed=42,
        num_candidates=1,
        condition="Condition_C_LLM_ControlNet"
    )
    assert len(results) == 1
    gen_result = results[0]
    img = gen_result["image"]
    assert img.size == (512, 512)
    assert Path(gen_result["path"]).exists()

    # 5. Multimodal Evaluation
    eval_result = service.evaluate_candidate(
        image=img,
        spec=spec,
        prompts=prompts,
        reference_image=outlines["cad_drawing"]
    )
    assert eval_result["clip_score"] > 0.0
    assert eval_result["overall_design_fidelity"] >= 60.0
    assert len(eval_result["feature_checklist"]) >= 3
    assert "CRITICAL ENGINEERING NOTICE" in eval_result["dimensional_accuracy_disclaimer"]
