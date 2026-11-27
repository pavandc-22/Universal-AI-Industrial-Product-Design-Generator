# Controlled Industrial Product Design Generation Architecture

## System Overview
This platform implements a universal, controlled AI-powered pipeline for converting natural language engineering instructions and dimensional specifications into high-fidelity industrial product mockups.

```
+--------------------------------------------------------------------------------------------------+
|                                    USER INPUT INTERFACE                                          |
| Natural Language Spec / Overrides / Ref Image / Viewing Angle / Conversational Modifications    |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                             UNIVERSAL SPECIFICATION PARSER                                       |
| - Extensible JSON Schema (Dimensions, Geometry, Materials, Components, Constraints)             |
| - Local Regex/NLP Fallback Parser + Extensible LLM API Support (OpenAI/Gemini/Anthropic)        |
| - Missing Specification Detector & Clarification Question Generator                             |
+--------------------------------------------------------------------------------------------------+
                                                 |
                        +------------------------+------------------------+
                        |                                                 |
                        v                                                 v
+----------------------------------------------+ +-------------------------------------------------+
|       UNIVERSAL PROMPT ENGINEERING           | |     PARAMETRIC GEOMETRY & OUTLINE ENGINE        |
| - Tailored Positive Prompts                  | | - Parametric Primitives (Circles, Teeth, etc.) |
| - Targeted Negative Anti-Artifact Prompts    | | - ControlNet Canny/LineArt Structural Edge Map  |
| - Micro-Surface Physical Texture Descriptions| | - Engineering CAD Blueprint & SVG Vector        |
| - Calibrated Studio Lighting & Camera Angle  | | - Canny Edge Extraction from User Ref Images    |
+----------------------------------------------+ +-------------------------------------------------+
                        |                                                 |
                        +------------------------+------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                            UNIVERSAL AI IMAGE GENERATION ENGINE                                  |
| Mode 1: Local Hugging Face Diffusers + ControlNet Pipeline (CUDA / CPU acceleration)             |
| Mode 2: High-Fidelity Parametric Procedural PBR CAD/Physics Shading (Demonstration Mode)         |
| Conditions Supported:                                                                            |
|   - Condition A: Naive prompt (uncalibrated baseline)                                            |
|   - Condition B: LLM-engineered prompt (calibrated materials & studio lighting)                 |
|   - Condition C: LLM prompt + Parametric Outline + ControlNet edge guidance                      |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                             MULTIMODAL EVALUATION & REFINEMENT                                   |
| - CLIPScore (Semantic Image-Text Alignment)                                                      |
| - SSIM (Structural Similarity against CAD reference)                                             |
| - VLM / Computer-Vision Feature Checklist Verification                                            |
| - Human Engineering Review & 1-5 Star Rating                                                     |
| - Automatic Refinement Feedback Loop (Up to 3 Iterative Rounds)                                  |
| - Critical Dimensional Accuracy vs Visual Similarity Disclaimer                                 |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                         CONVERSATIONAL NATURAL LANGUAGE MODIFIER                                 |
| Multi-turn natural language adjustments ("make it larger", "change material to aluminium", etc.)|
| Strict preservation of all unchanged dimensions and specifications                               |
+--------------------------------------------------------------------------------------------------+
```

## Key Modules
1. `modules/universal_parser.py`: Universal parser converting free-form text to structured JSON conforming to `SPEC_SCHEMA`.
2. `modules/prompt_engine.py`: Converts engineering specs into positive/negative prompts, camera setup, and material physical cues.
3. `modules/geometry_engine.py`: Parametric geometry engine supporting circles, teeth, flanges, slots, polygons, and assemblies.
4. `modules/outline_renderer.py`: Renders ControlNet edge maps, CAD blueprints with dimension leaders, and SVG vector files.
5. `modules/image_generator.py`: Dual-engine image generation with hardware detection (Diffusers ControlNet or PBR Shader).
6. `modules/controlnet_pipeline.py`: Diffusers ControlNet wrapper with model caching and CUDA/CPU placement.
7. `modules/evaluator.py`: Multimodal evaluation suite (CLIPScore, SSIM, feature verification, disclaimer).
8. `modules/refinement.py`: Automatic iterative feedback and refinement loop (maximum 3 rounds).
9. `modules/design_modifier.py`: Conversational design editor preserving unchanged parameters.
10. `modules/experiment_runner.py`: Scientific comparative benchmark comparing Conditions A, B, and C with CSV logging.
