# Universal AI Industrial Product Design Generator

Controlled Industrial Product Design Generation using Stable Diffusion, ControlNet, and LLM Prompting.

A universal platform for converting free-form natural language engineering instructions and dimensional specifications into high-fidelity industrial product mockups, parametric CAD edge outlines, and multimodal validation reports.

---

## 🚀 Quick Start

### 1. Launch the Web Application
```powershell
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

### 2. Run the Full Test Suite
```powershell
python -m pytest tests/
```
Runs 25 automated unit, integration, and final acceptance tests across all modules.

---

## 🌟 Key Capabilities

### 1. Universal Product Understanding (No Hardcoded Dropdowns)
Accepts any industrial product specification:
- Bearings, gears, shafts, centrifugal pumps, flanges, butterfly valves, enclosures, nozzles, robotic grippers, manifolds, or custom mechanical components.
- Extracts dimensions, materials, finishes, features, viewing angles, and constraints into an extensible JSON schema.
- Identifies missing engineering information and formulates clarification questions without hallucinating specs.

### 2. Universal Prompt Engineering Engine
Synthesizes dedicated prompts for each product:
- Positive prompts with micro-surface physical cues (lathe turning marks, brushed satin grain, anodized absorption).
- Targeted negative prompts excluding product-specific defects (e.g. broken cogs, leaking fluid, eccentric races).
- Structural and geometric descriptions with 5500K studio softbox illumination and macro lens angles.

### 3. Parametric Geometry & ControlNet Outline Engine
- Renders high-contrast edge maps for ControlNet conditioning.
- Renders technical CAD blueprints with centerlines, leader lines, dimension arrows, and annotations.
- Exports scalable W3C SVG vector files.
- Extracts Canny edge maps from user-uploaded reference images.

### 4. Dual-Engine Image Generation Architecture
- **CUDA/CPU Diffusers Pipeline**: Loads Stable Diffusion + ControlNet Canny models when available with local caching.
- **Parametric Procedural PBR CAD Shader Engine**: High-fidelity local physics engine producing authentic metallic 3D renders with ambient occlusion, drop shadows, and specular highlights when running in demonstration mode.
- Transparently labels hardware operating mode in UI and image stamps.

### 5. Conversational Natural Language Design Modification
Users can request modifications in natural language:
- *"Change the material to aluminium"*
- *"Add four mounting holes and scale up by 25%"*
- *"Make the colour matte black"*
- *"Show product from the front"*
- All unchanged dimensions and specifications are strictly preserved across turns.

### 6. Multimodal Evaluation & Automatic Refinement
- **CLIPScore**: Semantic image-prompt alignment.
- **SSIM**: Structural similarity index against CAD blueprints or reference images.
- **VLM / CV Feature Checklist**: Automated verification of holes, cogs, flanges, and materials.
- **Human Review**: 1 to 5 star rating interface with engineering reviewer notes.
- **Dimensional Accuracy Disclaimer**: Distinguishes visual mockup fidelity from physical manufacturing tolerances.
- **Automatic Refinement Loop**: Iteratively optimizes prompts and regenerated images up to 3 rounds.

### 7. Scientific A/B/C Experimental Comparison
Compares three research conditions across a 40-product industrial dataset:
- **Condition A**: Naive raw prompt.
- **Condition B**: LLM-engineered prompt.
- **Condition C**: LLM prompt + Parametric Outline + ControlNet.
- Logs authentic trial records directly to CSV in `outputs/experiments/`.

---

## 📁 Project Structure

```
prompt_en_project/
├── app.py                     # Streamlit Professional Web Application
├── requirements.txt           # Core and optional dependencies
├── PROGRESS.md                # Development phase tracker
├── README.md                  # System overview and quick start guide
├── backend/
│   ├── __init__.py
│   └── api.py                 # Unified service coordinator facade
├── modules/
│   ├── __init__.py
│   ├── universal_parser.py    # Extensible JSON parser with rule fallback & LLM API
│   ├── prompt_engine.py       # Tailored prompt synthesis engine
│   ├── geometry_engine.py     # Parametric geometry primitive builder
│   ├── outline_renderer.py    # ControlNet edge map, CAD blueprint & SVG renderer
│   ├── image_generator.py     # Dual-engine generation (Diffusers + PBR Shader)
│   ├── controlnet_pipeline.py # Hugging Face Diffusers ControlNet wrapper
│   ├── evaluator.py           # CLIPScore, SSIM, VLM checklist, human rating
│   ├── refinement.py          # Iterative feedback refinement loop (max 3 rounds)
│   ├── design_modifier.py     # Multi-turn conversational design modifier
│   └── experiment_runner.py   # A/B/C condition comparison & CSV logger
├── utils/
│   ├── __init__.py
│   ├── hardware.py            # GPU/CPU/Diffusers hardware detection
│   ├── helpers.py             # File I/O, base64 conversion, directory setup
│   └── logger.py              # Centralized logging configuration
├── dataset/
│   └── specifications.json    # Standard 40 industrial product specifications
├── outputs/
│   ├── generations/           # Saved product design images
│   ├── outlines/              # Saved CAD blueprints and edge maps
│   └── experiments/           # Exported experimental CSV files
├── tests/
│   ├── test_acceptance_5_products.py # Mandatory 5-product final acceptance suite
│   ├── test_parser.py
│   ├── test_prompt_engine.py
│   ├── test_geometry.py
│   ├── test_generator.py
│   ├── test_evaluator.py
│   ├── test_modifier.py
│   ├── test_refinement.py
│   └── test_experiment.py
└── docs/
    └── ARCHITECTURE.md        # Technical architecture specification
```
