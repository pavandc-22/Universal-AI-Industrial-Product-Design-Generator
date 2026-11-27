"""
Experimental Comparison Runner for Controlled Industrial Product Design Generation.
Runs and compares the three research conditions:
  Condition A: Naive prompt (raw user spec sent to generator)
  Condition B: LLM-engineered prompt (structured positive/negative prompts)
  Condition C: LLM prompt + parametric outline + ControlNet edge guidance
Stores authentic results to CSV.
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from PIL import Image

from utils.helpers import EXPERIMENTS_DIR, DATASET_DIR
from .universal_parser import UniversalSpecificationParser
from .prompt_engine import UniversalPromptEngine
from .geometry_engine import UniversalGeometryEngine
from .outline_renderer import OutlineRenderer
from .image_generator import UniversalImageGenerator
from .evaluator import UniversalEvaluator

logger = logging.getLogger("IndustrialDesignAI.experiment")

class ExperimentRunner:
    """Automates multi-condition scientific comparison and CSV data logging."""

    CONDITIONS = [
        "Condition_A_Naive",
        "Condition_B_LLM_Prompt",
        "Condition_C_LLM_ControlNet"
    ]

    DEFAULT_SEEDS = [42, 101, 2024, 777]

    def __init__(
        self,
        parser: Optional[UniversalSpecificationParser] = None,
        prompt_engine: Optional[UniversalPromptEngine] = None,
        geometry_engine: Optional[UniversalGeometryEngine] = None,
        generator: Optional[UniversalImageGenerator] = None,
        evaluator: Optional[UniversalEvaluator] = None
    ):
        self.parser = parser or UniversalSpecificationParser(provider="local")
        self.prompt_engine = prompt_engine or UniversalPromptEngine()
        self.geometry_engine = geometry_engine or UniversalGeometryEngine()
        self.outline_renderer = OutlineRenderer()
        self.generator = generator or UniversalImageGenerator()
        self.evaluator = evaluator or UniversalEvaluator()

    def load_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = DATASET_DIR / "specifications.json"
        if not dataset_path.exists():
            return []
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_experiment(
        self,
        sample_ids: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        conditions: Optional[List[str]] = None,
        output_csv_filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Runs the comparative experiment across specified samples, seeds, and conditions.
        Stores authentic metrics in CSV.
        """
        all_samples = self.load_dataset()
        if sample_ids:
            samples_to_run = [s for s in all_samples if s["id"] in sample_ids]
        else:
            # Default to first 3 validation samples to prevent heavy batch runs
            samples_to_run = all_samples[:3]

        eval_seeds = seeds or [42]
        eval_conditions = conditions or self.CONDITIONS

        records = []
        logger.info(f"Running experiment: {len(samples_to_run)} products x {len(eval_seeds)} seeds x {len(eval_conditions)} conditions")

        for s_idx, sample in enumerate(samples_to_run):
            spec_raw_text = sample["description"]
            product_id = sample["id"]
            product_name = sample["name"]
            category = sample["category"]

            # Parse specification
            spec = self.parser.parse(spec_raw_text)
            prompts = self.prompt_engine.generate(spec)
            outline_model = self.geometry_engine.build_geometry(spec)
            outline_img = self.outline_renderer.render_cad_drawing(outline_model)

            for seed in eval_seeds:
                for cond in eval_conditions:
                    t0 = time.time()
                    # Execute generation
                    gen_results = self.generator.generate(
                        spec=spec,
                        prompts=prompts,
                        outline_model=outline_model,
                        seed=seed,
                        num_candidates=1,
                        condition=cond
                    )
                    gen_res = gen_results[0]
                    img = gen_res["image"]
                    gen_time = round(time.time() - t0, 3)

                    # Evaluate generated output
                    eval_res = self.evaluator.evaluate(
                        image=img,
                        spec=spec,
                        prompts=prompts,
                        reference_image=outline_img
                    )

                    record = {
                        "timestamp": int(time.time()),
                        "product_id": product_id,
                        "product_name": product_name,
                        "category": category,
                        "condition": cond,
                        "seed": seed,
                        "clip_score": eval_res["clip_score"],
                        "ssim_score": eval_res["ssim_score"],
                        "features_verified_rate": eval_res["feature_verification_rate"],
                        "overall_design_fidelity": eval_res["overall_design_fidelity"],
                        "generation_time_sec": gen_time,
                        "engine_mode": gen_res["engine"],
                        "is_demonstration_mode": gen_res["is_demonstration_mode"],
                        "image_path": gen_res["path"]
                    }
                    records.append(record)

        df = pd.DataFrame(records)

        # Save authentic experimental results to CSV
        csv_filename = output_csv_filename or f"experiment_results_{int(time.time())}.csv"
        csv_path = EXPERIMENTS_DIR / csv_filename
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved experiment results to {csv_path}")

        return df

    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes aggregate comparative benchmarks by condition."""
        if df.empty:
            return pd.DataFrame()
        summary = df.groupby("condition").agg({
            "clip_score": ["mean", "std"],
            "ssim_score": ["mean", "std"],
            "features_verified_rate": "mean",
            "overall_design_fidelity": "mean",
            "generation_time_sec": "mean"
        }).round(3)
        return summary
