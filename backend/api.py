"""
Unified Backend Service API for Controlled Industrial Product Design Generation.
Provides clean facade for parsing, prompt synthesis, outline rendering, image generation,
evaluation, automatic refinement, and conversational modification.
"""

from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import pandas as pd

from modules.universal_parser import UniversalSpecificationParser
from modules.prompt_engine import UniversalPromptEngine
from modules.geometry_engine import UniversalGeometryEngine, ParametricProductModel
from modules.outline_renderer import OutlineRenderer
from modules.image_generator import UniversalImageGenerator
from modules.evaluator import UniversalEvaluator
from modules.design_modifier import NaturalLanguageDesignModifier
from modules.refinement import RefinementEngine
from modules.experiment_runner import ExperimentRunner
from utils.hardware import detect_hardware

class IndustrialDesignPlatformService:
    """Singleton service coordinator for frontend and API clients."""

    def __init__(self):
        self.hardware_info = detect_hardware()
        self.parser = UniversalSpecificationParser(provider="local")
        self.prompt_engine = UniversalPromptEngine()
        self.geometry_engine = UniversalGeometryEngine()
        self.outline_renderer = OutlineRenderer()
        self.generator = UniversalImageGenerator()
        self.evaluator = UniversalEvaluator()
        self.modifier = NaturalLanguageDesignModifier()
        self.refinement_engine = RefinementEngine(
            generator=self.generator,
            evaluator=self.evaluator,
            prompt_engine=self.prompt_engine,
            geometry_engine=self.geometry_engine
        )
        self.experiment_runner = ExperimentRunner(
            parser=self.parser,
            prompt_engine=self.prompt_engine,
            geometry_engine=self.geometry_engine,
            generator=self.generator,
            evaluator=self.evaluator
        )

    def parse_specification(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.parser.parse(text, user_hints)

    def generate_prompts(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self.prompt_engine.generate(spec)

    def generate_geometry_and_outlines(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        model = self.geometry_engine.build_geometry(spec)
        edge_map = self.outline_renderer.render_controlnet_edge_map(model)
        cad_drawing = self.outline_renderer.render_cad_drawing(model)
        svg_code = self.outline_renderer.to_svg(model)
        return {
            "model": model,
            "edge_map": edge_map,
            "cad_drawing": cad_drawing,
            "svg_code": svg_code
        }

    def generate_design(
        self,
        spec: Dict[str, Any],
        prompts: Dict[str, Any],
        outline_model: Optional[ParametricProductModel] = None,
        seed: int = 42,
        num_candidates: int = 1,
        condition: str = "Condition_C_LLM_ControlNet",
        width: int = 512,
        height: int = 512
    ) -> List[Dict[str, Any]]:
        return self.generator.generate(
            spec=spec,
            prompts=prompts,
            outline_model=outline_model,
            seed=seed,
            num_candidates=num_candidates,
            condition=condition,
            width=width,
            height=height
        )

    def evaluate_candidate(
        self,
        image: Image.Image,
        spec: Dict[str, Any],
        prompts: Dict[str, Any],
        reference_image: Optional[Image.Image] = None,
        human_rating: Optional[int] = None
    ) -> Dict[str, Any]:
        return self.evaluator.evaluate(
            image=image,
            spec=spec,
            prompts=prompts,
            reference_image=reference_image,
            human_rating=human_rating
        )

    def modify_design_conversational(self, spec: Dict[str, Any], user_request: str) -> Tuple[Dict[str, Any], str]:
        return self.modifier.modify(spec, user_request)

    def run_refinement(
        self,
        spec: Dict[str, Any],
        max_rounds: int = 3,
        target_score: float = 80.0,
        seed: int = 42,
        condition: str = "Condition_C_LLM_ControlNet"
    ) -> Dict[str, Any]:
        return self.refinement_engine.run_refinement_loop(
            spec=spec,
            max_rounds=max_rounds,
            target_score=target_score,
            seed=seed,
            condition=condition
        )

    def run_experiments(
        self,
        sample_ids: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        conditions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        return self.experiment_runner.run_experiment(sample_ids, seeds, conditions)

    def get_dataset(self) -> List[Dict[str, Any]]:
        return self.experiment_runner.load_dataset()
