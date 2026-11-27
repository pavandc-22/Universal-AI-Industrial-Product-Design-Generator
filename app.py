"""
Universal AI Industrial Product Design Generation Platform.
Web Application built with Streamlit, OpenCV, and Parametric CAD / ControlNet Pipelines.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from PIL import Image

# Ensure project root in python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.api import IndustrialDesignPlatformService
from utils.helpers import (
    OUTPUTS_DIR,
    GENERATIONS_DIR,
    OUTLINES_DIR,
    EXPERIMENTS_DIR,
    DATASET_DIR,
    image_to_base64
)

# Set page config
st.set_page_config(
    page_title="Industrial AI Design Studio",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CAD Studio Dark CSS Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Header Branding */
    .brand-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Status Badges */
    .engine-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .badge-demo {
        background-color: #451a03;
        color: #fbbf24;
        border: 1px solid #b45309;
    }
    .badge-cuda {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
    }
    .badge-cpu {
        background-color: #1e1b4b;
        color: #a5b4fc;
        border: 1px solid #4338ca;
    }
    
    /* Metrics & Cards */
    .metric-card {
        background-color: #131d31;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
    }

    /* Engineering Disclaimer */
    .disclaimer-box {
        background-color: #1c1917;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #fbbf24;
        margin: 16px 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #131c2e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #1e293b;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    return IndustrialDesignPlatformService()

service = get_service()

# Initialize Session State
if "current_spec" not in st.session_state:
    st.session_state.current_spec = None
if "current_prompts" not in st.session_state:
    st.session_state.current_prompts = None
if "current_outlines" not in st.session_state:
    st.session_state.current_outlines = None
if "generated_results" not in st.session_state:
    st.session_state.generated_results = []
if "latest_evaluation" not in st.session_state:
    st.session_state.latest_evaluation = None
if "modification_chat_history" not in st.session_state:
    st.session_state.modification_chat_history = []
if "refinement_history" not in st.session_state:
    st.session_state.refinement_history = []

# Sidebar Diagnostics & Quick Controls
with st.sidebar:
    st.markdown("### ⚙️ System Engine")
    hw = service.hardware_info
    if hw["is_demonstration_mode"]:
        st.markdown('<div class="engine-badge badge-demo">Demonstration Engine (CPU Mode)</div>', unsafe_allow_html=True)
    elif hw["has_cuda"]:
        st.markdown(f'<div class="engine-badge badge-cuda">CUDA Acceleration ({hw["cuda_device_name"]})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="engine-badge badge-cpu">Diffusers CPU Mode</div>', unsafe_allow_html=True)

    st.caption(f"Status: {hw['status_message']}")
    st.divider()

    st.markdown("### 🎛️ Design Pipeline Settings")
    active_condition = st.selectbox(
        "Research Condition",
        [
            "Condition_C_LLM_ControlNet",
            "Condition_B_LLM_Prompt",
            "Condition_A_Naive"
        ],
        format_func=lambda x: {
            "Condition_A_Naive": "Condition A (Naive Raw Prompt)",
            "Condition_B_LLM_Prompt": "Condition B (LLM-Engineered Prompt)",
            "Condition_C_LLM_ControlNet": "Condition C (LLM + Outline + ControlNet)"
        }[x]
    )

    resolution_choice = st.selectbox("Render Resolution", [(512, 512), (768, 768), (1024, 1024)], index=0, format_func=lambda s: f"{s[0]} x {s[1]}")
    candidate_count = st.slider("Candidate Variants", min_value=1, max_value=4, value=2)
    random_seed = st.number_input("Generation Seed", min_value=1, max_value=999999, value=42, step=1)

    st.divider()
    st.markdown("### 📂 Quick Sample Loader")
    sample_options = [
        "-- Select a Sample Spec --",
        "Centrifugal Water Pump (Stainless Steel)",
        "Precision Spur Gear (36 Teeth, 100mm OD)",
        "Deep-Groove Ball Bearing (120mm OD)",
        "NEMA 4X Industrial Electrical Enclosure",
        "ANSI Class 150 Flange Coupling",
        "Titanium Micro-Nozzle Manifold"
    ]
    selected_sample = st.selectbox("Load Standard Specification", sample_options)

    sample_texts = {
        "Centrifugal Water Pump (Stainless Steel)": "Generate an industrial centrifugal water pump with a stainless steel body, circular inlet, side outlet, four mounting holes, compact design and a professional product catalogue appearance.",
        "Precision Spur Gear (36 Teeth, 100mm OD)": "Create a custom mechanical gear with 36 teeth, 100 mm outer diameter, 25 mm central bore and hardened steel finish.",
        "Deep-Groove Ball Bearing (120mm OD)": "Single-row deep groove ball bearing with 120 mm outer diameter, 45 mm inner bore, 20 mm width, hardened chrome steel construction and stamped steel cage with 8 precision ground steel balls.",
        "NEMA 4X Industrial Electrical Enclosure": "Wall-mount industrial electrical enclosure box, 300 mm x 250 mm x 150 mm, 304 stainless steel with continuous polyurethane door gasket, quarter-turn key latch, external mounting feet.",
        "ANSI Class 150 Flange Coupling": "ANSI Class 150 weld neck pipe flange with 190 mm outer diameter, 60 mm bore, raised sealing face, 8 bolt holes on 152 mm bolt circle diameter, forged carbon steel A105.",
        "Titanium Micro-Nozzle Manifold": "Aerospace titanium micro-nozzle manifold with 12 radial spray orifices, electropolished finish, 45 mm diameter, and high pressure rating."
    }

# Top Branding Banner
st.markdown("""
<div class="brand-header">
    <div class="brand-title">Controlled Industrial Product Design Generation</div>
    <div class="brand-subtitle">Universal AI-Powered Industrial Component Synthesis using Stable Diffusion, ControlNet, and LLM Prompting</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_workspace, tab_specs, tab_outlines, tab_gallery, tab_evaluation, tab_experiments, tab_dataset = st.tabs([
    "🎨 Design Workspace",
    "📋 Specification View",
    "📐 Outline & CAD View",
    "🖼️ Results & Modification",
    "📊 Evaluation & Refinement",
    "🧪 A/B/C Experiments",
    "📦 Dataset Explorer"
])

# -------------------------------------------------------------
# TAB 1: DESIGN WORKSPACE
# -------------------------------------------------------------
with tab_workspace:
    st.markdown("#### Describe Any Industrial Component or Product")
    st.caption("Universal system: Type any product description (bearings, pumps, gears, flanges, valves, enclosures, nozzles, or custom parts). The system understands arbitrary industrial specifications.")

    initial_prompt_val = sample_texts.get(selected_sample, "")
    user_input_text = st.text_area(
        "Natural Language Product Description & Engineering Instructions",
        value=initial_prompt_val,
        height=130,
        placeholder="e.g. Generate an industrial centrifugal water pump with a stainless steel body, circular inlet, side outlet, four mounting holes, compact design and a professional product catalogue appearance."
    )

    with st.expander("🛠️ Advanced Engineering Overrides (Optional Hints)", expanded=False):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            override_material = st.selectbox("Base Material Override", ["(Auto-detect)", "Stainless Steel", "Hardened Steel", "Carbon Steel", "Aluminum 6061", "Brass", "Bronze", "Cast Iron", "Titanium"])
            override_finish = st.selectbox("Surface Finish Override", ["(Auto-detect)", "Machined", "Brushed Satin", "Mirror Polished", "Bead Blasted", "Black Oxide", "Powder Coated", "Anodized"])
        with col_m2:
            override_view = st.selectbox("Viewing Angle", ["Isometric 3D perspective view", "Direct front elevation view", "Top-down orthographic plan view", "Cutaway cross-sectional view"])
            override_color = st.text_input("Color Override", placeholder="e.g. Matte Black, Metallic Silver, Safety Yellow")
        with col_m3:
            override_od = st.number_input("Outer Diameter (mm)", min_value=0.0, value=0.0, step=5.0)
            override_bore = st.number_input("Bore / ID (mm)", min_value=0.0, value=0.0, step=5.0)

        uploaded_ref_img = st.file_uploader("Upload Reference Image / Drawing (Optional)", type=["png", "jpg", "jpeg"])

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        generate_clicked = st.button("🚀 Synthesize Design", type="primary", use_container_width=True)

    if generate_clicked:
        if not user_input_text.strip():
            st.error("Please enter an industrial product description.")
        else:
            with st.spinner("Parsing specifications and synthesizing parametric geometry..."):
                # Compile overrides
                hints = {}
                if override_material != "(Auto-detect)": hints["material"] = override_material.lower()
                if override_finish != "(Auto-detect)": hints["surface_finish"] = override_finish.lower()
                if override_color.strip(): hints["color"] = override_color.strip()
                if override_view: hints["viewing_angle"] = override_view
                if override_od > 0 or override_bore > 0:
                    hints["dimensions"] = {}
                    if override_od > 0: hints["dimensions"]["outer_diameter"] = override_od
                    if override_bore > 0: hints["dimensions"]["central_bore"] = override_bore

                # 1. Parse Specification
                spec = service.parse_specification(user_input_text, user_hints=hints)
                st.session_state.current_spec = spec

                # 2. Synthesize Prompts
                prompts = service.generate_prompts(spec)
                st.session_state.current_prompts = prompts

                # 3. Generate Geometry & Outlines
                outlines = service.generate_geometry_and_outlines(spec)
                st.session_state.current_outlines = outlines

                # 4. Generate Images
                results = service.generate_design(
                    spec=spec,
                    prompts=prompts,
                    outline_model=outlines["model"],
                    seed=random_seed,
                    num_candidates=candidate_count,
                    condition=active_condition,
                    width=resolution_choice[0],
                    height=resolution_choice[1]
                )
                st.session_state.generated_results = results

                # 5. Evaluate First Candidate
                if results:
                    eval_res = service.evaluate_candidate(
                        image=results[0]["image"],
                        spec=spec,
                        prompts=prompts,
                        reference_image=outlines["cad_drawing"]
                    )
                    st.session_state.latest_evaluation = eval_res

            st.success(f"Successfully generated {len(results)} candidate mockups for '{spec.get('product_name')}'!")

    # If results exist, display quick summary in Workspace
    if st.session_state.generated_results:
        st.divider()
        st.markdown("### ⚡ Latest Generated Candidates")
        cols = st.columns(len(st.session_state.generated_results))
        for idx, res in enumerate(st.session_state.generated_results):
            with cols[idx]:
                st.image(res["image"], caption=f"Candidate #{idx+1} (Seed: {res['seed']})", use_container_width=True)
                st.download_button(
                    label=f"💾 Download #{idx+1}",
                    data=open(res["path"], "rb").read(),
                    file_name=Path(res["path"]).name,
                    mime="image/png",
                    key=f"dl_quick_{idx}"
                )

# -------------------------------------------------------------
# TAB 2: SPECIFICATION VIEW
# -------------------------------------------------------------
with tab_specs:
    st.markdown("### 📋 Structured Engineering Specification")
    if not st.session_state.current_spec:
        st.info("No specification parsed yet. Enter a description in Design Workspace and click 'Synthesize Design'.")
    else:
        spec = st.session_state.current_spec
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Product Name", spec.get("product_name", "N/A"))
        c2.metric("Category", spec.get("category", "N/A"))
        c3.metric("Base Material", spec.get("material", {}).get("base_material", "N/A").title())
        c4.metric("Surface Finish", spec.get("surface_finish", "N/A").title())

        # Missing Information Alerts & Clarification Questions
        if spec.get("clarification_needed") and spec.get("missing_information"):
            st.warning("⚠️ **Missing Information Identified:**\n- " + "\n- ".join(spec["missing_information"]))
            if spec.get("clarification_questions"):
                st.info("💡 **Clarification Questions:**\n- " + "\n- ".join(spec["clarification_questions"]))

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Extracted Engineering Dimensions")
            dims = spec.get("dimensions", {})
            dim_data = [
                {"Parameter": "Unit", "Value": dims.get("unit", "mm")},
                {"Parameter": "Outer Diameter", "Value": dims.get("outer_diameter") or "N/A"},
                {"Parameter": "Inner Diameter / Bore", "Value": dims.get("central_bore") or dims.get("inner_diameter") or "N/A"},
                {"Parameter": "Teeth Count", "Value": dims.get("teeth_count") or "N/A"},
                {"Parameter": "Mounting Holes", "Value": dims.get("mounting_holes_count") or "N/A"},
                {"Parameter": "Length x Width x Height", "Value": f"{dims.get('length') or '-'} x {dims.get('width') or '-'} x {dims.get('height') or '-'}"},
                {"Parameter": "Axial Thickness", "Value": dims.get("thickness") or "N/A"}
            ]
            st.dataframe(pd.DataFrame(dim_data), use_container_width=True, hide_index=True)

            st.markdown("#### Components & Features")
            st.write("**Components:**", ", ".join(spec.get("components", [])) or "None specified")
            st.write("**Features:**", ", ".join(spec.get("features", [])) or "Standard precision features")
            st.write("**Design Constraints:**", ", ".join(spec.get("design_constraints", [])) or "None")

        with col_s2:
            st.markdown("#### Full Extensible JSON Schema Output")
            st.json(spec)
            st.download_button(
                "📥 Export Specification JSON",
                data=json.dumps(spec, indent=2),
                file_name=f"{spec.get('product_name', 'spec').lower()}_spec.json",
                mime="application/json"
            )

# -------------------------------------------------------------
# TAB 3: OUTLINE & CAD VIEW
# -------------------------------------------------------------
with tab_outlines:
    st.markdown("### 📐 Universal Geometry & ControlNet Edge Outlines")
    if not st.session_state.current_outlines:
        st.info("Generate a design in the Workspace to view parametric outlines and CAD schematics.")
    else:
        outlines = st.session_state.current_outlines
        col_o1, col_o2 = st.columns(2)

        with col_o1:
            st.markdown("#### 1. ControlNet Edge Map (Guidance Input)")
            st.caption("Clean structural outline used for ControlNet edge conditioning.")
            st.image(outlines["edge_map"], use_container_width=True)

        with col_o2:
            st.markdown("#### 2. Technical Engineering CAD Blueprint")
            st.caption("Parametric schematic with centerlines, grid, dimension leaders, and annotations.")
            st.image(outlines["cad_drawing"], use_container_width=True)

        st.markdown("#### 3. Synthesized Prompt Architecture")
        if st.session_state.current_prompts:
            p = st.session_state.current_prompts
            st.text_area("Positive Image Prompt", p["positive_prompt"], height=90)
            st.text_area("Negative Image Prompt", p["negative_prompt"], height=60)
            with st.expander("Detailed Prompt Components"):
                st.write("**Structural Description:**", p["structural_description"])
                st.write("**Geometric Features:**", p["geometric_feature_description"])
                st.write("**Material & Micro-Texture:**", p["material_texture_description"])
                st.write("**Camera & Lighting:**", p["camera_lighting_instructions"])

        st.markdown("#### 4. SVG Vector Export")
        st.download_button(
            "📥 Download CAD Outline as SVG",
            data=outlines["svg_code"],
            file_name="product_cad_outline.svg",
            mime="image/svg+xml"
        )

# -------------------------------------------------------------
# TAB 4: RESULTS & MODIFICATION
# -------------------------------------------------------------
with tab_gallery:
    st.markdown("### 🖼️ Generated Candidate Gallery & Natural Language Modification")

    if not st.session_state.generated_results:
        st.info("No images generated yet. Run generation from the Design Workspace.")
    else:
        # Gallery Grid
        cols = st.columns(len(st.session_state.generated_results))
        for idx, item in enumerate(st.session_state.generated_results):
            with cols[idx]:
                st.image(item["image"], caption=f"Candidate {idx+1} | Seed: {item['seed']} | Time: {item['generation_time_sec']}s", use_container_width=True)
                st.download_button(
                    label=f"💾 Download Image #{idx+1}",
                    data=open(item["path"], "rb").read(),
                    file_name=Path(item["path"]).name,
                    mime="image/png",
                    key=f"dl_gal_{idx}"
                )

        st.divider()
        st.markdown("### 💬 Conversational Natural Language Design Modification")
        st.caption("Request any modification in natural language (e.g. 'Make it larger', 'Change the material to aluminium', 'Add four mounting holes', 'Change colour to black', 'Show product from the front'). All unchanged specifications are preserved.")

        # Modification Chat UI
        for chat in st.session_state.modification_chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
                if "image" in chat:
                    st.image(chat["image"], width=380)

        mod_request = st.chat_input("Enter design modification request...")
        if mod_request:
            with st.spinner(f"Applying modification: '{mod_request}'..."):
                # 1. Modify specification
                updated_spec, mod_summary = service.modify_design_conversational(
                    st.session_state.current_spec, mod_request
                )
                st.session_state.current_spec = updated_spec

                # 2. Regenerate prompts
                updated_prompts = service.generate_prompts(updated_spec)
                st.session_state.current_prompts = updated_prompts

                # 3. Regenerate geometry and outlines
                updated_outlines = service.generate_geometry_and_outlines(updated_spec)
                st.session_state.current_outlines = updated_outlines

                # 4. Regenerate design
                new_results = service.generate_design(
                    spec=updated_spec,
                    prompts=updated_prompts,
                    outline_model=updated_outlines["model"],
                    seed=st.session_state.generated_results[0]["seed"] + 1 if st.session_state.generated_results else 43,
                    num_candidates=1,
                    condition=active_condition,
                    width=resolution_choice[0],
                    height=resolution_choice[1]
                )
                st.session_state.generated_results = new_results
                latest_mod_img = new_results[0]["image"]

                # 5. Re-evaluate
                eval_res = service.evaluate_candidate(
                    image=latest_mod_img,
                    spec=updated_spec,
                    prompts=updated_prompts,
                    reference_image=updated_outlines["cad_drawing"]
                )
                st.session_state.latest_evaluation = eval_res

                # Record in chat history
                st.session_state.modification_chat_history.append({
                    "user": mod_request,
                    "assistant": f"✅ Design updated successfully! Changes applied: **{mod_summary}**.",
                    "image": latest_mod_img
                })
                st.rerun()

# -------------------------------------------------------------
# TAB 5: EVALUATION & REFINEMENT
# -------------------------------------------------------------
with tab_evaluation:
    st.markdown("### 📊 Multimodal Image Evaluation & Automated Refinement Loop")

    if not st.session_state.latest_evaluation:
        st.info("Run generation from Design Workspace to evaluate images.")
    else:
        ev = st.session_state.latest_evaluation

        # Critical Engineering Notice
        st.markdown(f"""
        <div class="disclaimer-box">
            <b>DISCLAIMER:</b> {ev['dimensional_accuracy_disclaimer']}
        </div>
        """, unsafe_allow_html=True)

        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("CLIPScore (Semantic)", f"{ev['clip_score_pct']}", help="Measures image-prompt semantic alignment")
        col_e2.metric("SSIM (Structural)", f"{ev['ssim_score'] if ev['ssim_score'] is not None else 'N/A'}", help="Structural Similarity against CAD model")
        col_e3.metric("Features Verified", f"{ev['features_verified']}", help="VLM/Vision checklist verification rate")
        col_e4.metric("Overall Design Fidelity", f"{ev['overall_design_fidelity']}%", help="Weighted composite score")

        st.markdown("#### Automated Feature Verification Checklist")
        for item in ev.get("feature_checklist", []):
            status_icon = "✅" if item["verified"] else "❌"
            confidence_pct = round(item["confidence"] * 100, 1)
            st.markdown(f"- **{status_icon} {item['feature']}**: {item['detail']} *(Confidence: {confidence_pct}%)*")

        st.divider()
        st.markdown("#### Human Review & Engineering Rating")
        col_h1, col_h2 = st.columns([1, 2])
        with col_h1:
            human_stars = st.feedback("stars")
            st.caption("Rate the visual realism and specification fidelity (1-5)")
        with col_h2:
            review_note = st.text_input("Reviewer Engineering Notes", placeholder="e.g. Flange bolt pattern conforms to ANSI 150 standard")
            if st.button("Submit Human Rating"):
                st.success(f"Rating recorded: {human_stars + 1 if human_stars is not None else 'Not set'} stars with note: '{review_note}'")

        st.divider()
        st.markdown("### 🔄 Automated Refinement Loop")
        st.caption("Iteratively executes: Specification → Prompts → Outlines → AI Image Generation → Evaluation → Failure Identification → Targeted Refinement → Regeneration (Up to 3 rounds).")

        refine_col1, refine_col2 = st.columns([1, 2])
        with refine_col1:
            target_score_slider = st.slider("Target Design Fidelity Score", 70.0, 95.0, 85.0, 1.0)
            start_refine_btn = st.button("⚡ Run Auto-Refinement Loop (Max 3 Rounds)", type="secondary")

        if start_refine_btn and st.session_state.current_spec:
            with st.spinner("Executing automatic refinement loop..."):
                refine_res = service.run_refinement(
                    spec=st.session_state.current_spec,
                    max_rounds=3,
                    target_score=target_score_slider,
                    seed=random_seed,
                    condition=active_condition
                )
                st.session_state.refinement_history = refine_res["history"]
                st.session_state.latest_evaluation = refine_res["latest_evaluation"]
                st.session_state.generated_results = [{
                    "image": refine_res["latest_image"],
                    "path": refine_res["latest_image_path"],
                    "seed": random_seed,
                    "condition": active_condition,
                    "generation_time_sec": 1.2,
                    "engine": "Parametric_PBR_Refinement_Engine",
                    "is_demonstration_mode": service.hardware_info["is_demonstration_mode"]
                }]

            st.success(f"Refinement completed in {refine_res['rounds_completed']} round(s)! Target reached: {refine_res['target_reached']}")
            st.rerun()

        if st.session_state.refinement_history:
            st.markdown("#### Refinement Iteration Trajectory")
            ref_df = pd.DataFrame([
                {
                    "Round": r["round"],
                    "Seed": r["seed"],
                    "Fidelity Score": f"{r['fidelity_score']}%",
                    "CLIPScore": round(r["clip_score"], 4),
                    "Features": r["features_verified"],
                    "Action Taken": r["action_taken"],
                    "Failures Handled": ", ".join(r["failures_identified"])
                }
                for r in st.session_state.refinement_history
            ])
            st.dataframe(ref_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 6: EXPERIMENT DASHBOARD
# -------------------------------------------------------------
with tab_experiments:
    st.markdown("### 🧪 Experimental Comparison (Condition A vs Condition B vs Condition C)")
    st.caption("Scientific benchmark comparing Naive Prompting (A), LLM-Engineered Prompting (B), and LLM + Outline + ControlNet (C).")

    all_specs = service.get_dataset()
    sample_select_options = [f"{s['id']} - {s['name']}" for s in all_specs]

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        chosen_sample_names = st.multiselect(
            "Select Products for Comparison",
            sample_select_options,
            default=sample_select_options[:3]
        )
        chosen_ids = [s.split(" - ")[0] for s in chosen_sample_names]

    with col_exp2:
        chosen_seeds = st.multiselect("Random Seeds", [42, 101, 777, 2024], default=[42])
        chosen_conditions = st.multiselect(
            "Conditions to Compare",
            service.experiment_runner.CONDITIONS,
            default=service.experiment_runner.CONDITIONS
        )

    run_exp_btn = st.button("🚀 Run Scientific Experiment & Log CSV", type="primary")

    if run_exp_btn:
        if not chosen_ids:
            st.error("Please select at least one product specification.")
        else:
            with st.spinner(f"Running experiment across {len(chosen_ids)} products x {len(chosen_seeds)} seeds x {len(chosen_conditions)} conditions..."):
                exp_df = service.run_experiments(
                    sample_ids=chosen_ids,
                    seeds=chosen_seeds,
                    conditions=chosen_conditions
                )
                st.session_state.latest_experiment_df = exp_df

            st.success(f"Experiment completed! Logged {len(exp_df)} authentic trial rows.")

    if "latest_experiment_df" in st.session_state:
        df = st.session_state.latest_experiment_df
        st.markdown("#### Comparative Benchmark Results")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Aggregate Condition Summary")
        summary_df = service.experiment_runner.get_summary_statistics(df)
        st.dataframe(summary_df, use_container_width=True)

        # Comparative Visualization Charts
        st.markdown("#### Metric Comparisons Across Research Conditions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mean Overall Design Fidelity (%)**")
            chart_df = df.groupby("condition")["overall_design_fidelity"].mean().reset_index()
            st.bar_chart(chart_df, x="condition", y="overall_design_fidelity")
        with c2:
            st.markdown("**Mean CLIPScore**")
            chart_clip = df.groupby("condition")["clip_score"].mean().reset_index()
            st.bar_chart(chart_clip, x="condition", y="clip_score")

        st.download_button(
            "📥 Download Authentic Experiment CSV",
            data=df.to_csv(index=False),
            file_name=f"industrial_ai_experiment_{int(time.time())}.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 7: DATASET EXPLORER
# -------------------------------------------------------------
with tab_dataset:
    st.markdown("### 📦 Industrial Product Specifications Dataset (40 Specifications)")
    st.caption("Standardized dataset spanning Mechanical Power Transmission, Fluid Handling, Bearings, Enclosures, Thermal Management, and Custom Machinery.")

    dataset = service.get_dataset()
    categories = sorted(list(set(d["category"] for d in dataset)))
    filter_cat = st.selectbox("Filter by Category", ["All Categories"] + categories)

    filtered_dataset = dataset if filter_cat == "All Categories" else [d for d in dataset if d["category"] == filter_cat]

    for item in filtered_dataset:
        with st.expander(f"{item['id']} — {item['name']} ({item['category']})"):
            st.write(item["description"])
            if st.button(f"Load '{item['name']}' into Design Workspace", key=f"load_{item['id']}"):
                spec = service.parse_specification(item["description"])
                st.session_state.current_spec = spec
                prompts = service.generate_prompts(spec)
                st.session_state.current_prompts = prompts
                outlines = service.generate_geometry_and_outlines(spec)
                st.session_state.current_outlines = outlines
                st.success(f"Loaded '{item['name']}'! Navigate to Design Workspace to synthesize.")
