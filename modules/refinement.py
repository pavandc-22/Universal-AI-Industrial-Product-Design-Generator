"""
Automatic Refinement Engine for Controlled Industrial Product Design.
Executes an automated feedback and refinement loop (maximum 3 rounds).
Preserves the original engineering specification while optimizing prompt and layout fidelity.
"""

import copy
import logging
from typing import Dict, Any, List, Optional
from PIL import Image

from .prompt_engine import UniversalPromptEngine
from .geometry_engine import UniversalGeometryEngine
from .outline_renderer import OutlineRenderer
from .image_generator import UniversalImageGenerator
from .evaluator import UniversalEvaluator

logger = logging.getLogger("IndustrialDesignAI.refinement")

class RefinementEngine:
    """Manages iterative evaluation and automatic prompt/layout refinement loop."""

    def __init__(
        self,
        generator: Optional[UniversalImageGenerator] = None,
        evaluator: Optional[UniversalEvaluator] = None,
        prompt_engine: Optional[UniversalPromptEngine] = None,
        geometry_engine: Optional[UniversalGeometryEngine] = None
    ):
        self.generator = generator or UniversalImageGenerator()
        self.evaluator = evaluator or UniversalEvaluator()
        self.prompt_engine = prompt_engine or UniversalPromptEngine()
        self.geometry_engine = geometry_engine or UniversalGeometryEngine()

    def run_refinement_loop(
        self,
        spec: Dict[str, Any],
        initial_prompts: Optional[Dict[str, Any]] = None,
        max_rounds: int = 3,
        target_score: float = 80.0,
        seed: int = 42,
        condition: str = "Condition_C_LLM_ControlNet"
    ) -> Dict[str, Any]:
        """
        Runs evaluation and automatic refinement loop up to max_rounds.
        Returns final refined generation, evaluation, and full round history.
        """
        rounds_history = []
        current_spec = copy.deepcopy(spec)
        current_prompts = initial_prompts or self.prompt_engine.generate(current_spec)
        current_seed = seed

        final_result = None

        for r in range(1, max_rounds + 1):
            logger.info(f"Starting Refinement Round {r}/{max_rounds}")

            # 1. Outline model
            outline_model = self.geometry_engine.build_geometry(current_spec)

            # 2. Generate candidate
            gen_results = self.generator.generate(
                spec=current_spec,
                prompts=current_prompts,
                outline_model=outline_model,
                seed=current_seed,
                num_candidates=1,
                condition=condition
            )
            candidate = gen_results[0]
            current_image = candidate["image"]

            # 3. Evaluate candidate
            eval_metrics = self.evaluator.evaluate(
                image=current_image,
                spec=current_spec,
                prompts=current_prompts
            )

            score = eval_metrics["overall_design_fidelity"]
            checklist = eval_metrics["feature_checklist"]

            # 4. Failure identification
            failures = self._identify_failures(eval_metrics, current_spec)

            round_record = {
                "round": r,
                "seed": current_seed,
                "fidelity_score": score,
                "clip_score": eval_metrics["clip_score"],
                "features_verified": eval_metrics["features_verified"],
                "failures_identified": failures,
                "image_path": candidate["path"],
                "prompts_used": copy.deepcopy(current_prompts),
                "action_taken": "None (Target achieved)" if score >= target_score or r == max_rounds else "Prompt & Feature Reinforcement"
            }
            rounds_history.append(round_record)

            final_result = {
                "latest_image": current_image,
                "latest_image_path": candidate["path"],
                "latest_evaluation": eval_metrics,
                "latest_prompts": current_prompts,
                "final_spec": current_spec,
                "rounds_completed": r,
                "target_reached": score >= target_score,
                "history": rounds_history
            }

            # If target fidelity reached or last round, terminate
            if score >= target_score or r >= max_rounds:
                logger.info(f"Refinement loop completed at round {r} with score {score}")
                break

            # 5. Formulate targeted refinements for next round
            current_prompts = self._refine_prompts(current_prompts, failures, current_spec)
            current_seed += 10 # slight variation for generation exploration

        return final_result

    def _identify_failures(self, eval_metrics: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
        failures = []
        for item in eval_metrics.get("feature_checklist", []):
            if not item.get("verified", False):
                failures.append(f"Unverified feature: {item.get('feature')}")

        if eval_metrics.get("clip_score", 1.0) < 0.78:
            failures.append("Sub-optimal text-image semantic alignment")

        if not failures:
            failures.append("Minor specular and edge contrast optimization required")
        return failures

    def _refine_prompts(self, prompts: Dict[str, Any], failures: List[str], spec: Dict[str, Any]) -> Dict[str, Any]:
        """Refines prompt vocabulary based on detected failure cues."""
        refined = copy.deepcopy(prompts)
        pos = refined.get("positive_prompt", "")
        neg = refined.get("negative_prompt", "")

        refinement_tokens = []
        for f in failures:
            if "Material" in f:
                refinement_tokens.append("hyper-accurate specular metallic reflection, razor-sharp machined lathe texture")
            elif "Primary Geometry" in f:
                refinement_tokens.append("perfect geometric symmetry, crisp structural silhouette, sharp CAD boundaries")
            elif "Circular Interfaces" in f:
                refinement_tokens.append("high-precision concentric circular bore, perfectly drilled bolt hole circle")
            elif "alignment" in f.lower():
                refinement_tokens.append("pristine industrial product catalogue presentation, award-winning studio lighting")

        if refinement_tokens:
            refined["positive_prompt"] = f"{pos}, {', '.join(refinement_tokens)}"

        # Strengthen negative prompt against distortions
        refined["negative_prompt"] = f"{neg}, imperfect chamfer, asymmetrical holes, blurry edges, noisy background"

        return refined
