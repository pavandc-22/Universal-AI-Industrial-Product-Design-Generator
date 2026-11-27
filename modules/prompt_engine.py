"""
Universal Prompt Engineering Engine for Industrial Product Design.
Converts structured JSON specifications into tailored positive, negative,
structural, geometric, material, and camera lighting prompts.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger("IndustrialDesignAI.prompt_engine")

class UniversalPromptEngine:
    """Intelligent prompt synthesizer tailored to specific industrial products."""

    BASE_NEGATIVE_PROMPTS = [
        "cartoon", "anime", "sketch", "hand-drawn", "low quality", "blurry",
        "deformed geometry", "warped perspective", "plastic toy look", "bad proportions",
        "unrealistic lighting", "overexposed", "underexposed", "rust and dirt",
        "human hands", "amateur render", "cluttered background", "watermark",
        "text overlay", "jpeg artifacts", "oversaturated colors"
    ]

    MATERIAL_VISUAL_CUES = {
        "stainless steel": "machined austenitic stainless steel, uniform directional brushed satin grain, crisp specular highlights, neutral silver reflection",
        "hardened steel": "precision ground alloy steel, dark gunmetal tempered tone, subtle oil-quench sheen, micro-finished tooth flanks",
        "carbon steel": "cold-rolled structural steel, smooth turned finish, industrial anti-corrosion protective film, solid mass",
        "aluminum": "anodized 6061-T6 aerospace aluminum, velvety bead-blasted matte finish, sharp precision CNC chamfers, lightweight solid structure",
        "aluminum 6061": "billet CNC machined 6061 aluminum, micro-faceted tool marks, clear protective anodized layer",
        "brass": "solid naval brass, warm golden metallic reflections, diamond-turned face, subtle micro-polish",
        "bronze": "cast phosphor bronze, warm reddish-gold luster, dense industrial cast texture, machined seating surfaces",
        "cast iron": "heavy gray cast iron, subtle granular cast texture, machined flat mounting faces, durable industrial casting",
        "titanium": "Grade 5 titanium alloy (Ti-6Al-4V), high strength-to-weight metallic luster, precision wire-EDM edges",
        "copper": "pure electrolytic copper, high thermal conductivity sheen, warm amber metallic gloss",
        "ceramic": "technical high-purity alumina ceramic, smooth non-porous white surface, extreme wear resistance"
    }

    CATEGORY_LIGHTING = {
        "Bearings & Rotary Motion": "precision laboratory studio lighting, dual rim lights tracing outer ring curvature, subtle ground reflections, neutral 5000K",
        "Mechanical Power Transmission": "raking directional studio light revealing tooth profiles and flank geometry, controlled specular sheen, high-contrast studio",
        "Fluid Handling & Hydraulics": "clean industrial catalogue lighting, softbox illumination on curved volute body and flange faces, clean drop shadow",
        "Electrical & Industrial Enclosures": "architectural studio lighting, broad diffusion panel highlighting panel flatness, seam seals, and conduit knockouts",
        "Electromechanical Systems": "balanced technical product lighting, highlighting ribbed motor housing, drive shaft, and wiring junction enclosure",
        "Piping & Structural Couplings": "orthographic high-key industrial photography, precise reflection on raised face and bolt circle holes"
    }

    def generate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizes complete prompt suite from specification."""
        product_name = spec.get("product_name", "Industrial Product")
        category = spec.get("category", "General Industrial")
        dims = spec.get("dimensions", {})
        mat = spec.get("material", {})
        base_mat = mat.get("base_material", "steel").lower()
        finish = spec.get("surface_finish", "machined")
        color = spec.get("color", "metallic silver")
        features = spec.get("features", [])
        components = spec.get("components", [])
        constraints = spec.get("design_constraints", [])
        appearance = spec.get("required_appearance", "professional product catalogue appearance")
        viewing_angle = spec.get("viewing_angle", "isometric 3D perspective view")
        geom = spec.get("primary_geometry", {})

        # 1. Structural Description
        structural_desc = self._build_structural_description(product_name, geom, components, features)

        # 2. Geometric Feature Description
        geometric_desc = self._build_geometric_description(dims, geom, features)

        # 3. Material & Texture Description
        material_desc = self._build_material_description(base_mat, finish, color, mat.get("treatment"))

        # 4. Camera & Lighting Instructions
        camera_lighting = self._build_camera_lighting(category, viewing_angle, appearance)

        # 5. Composite Positive Prompt
        positive_prompt = self._build_positive_prompt(
            product_name, category, structural_desc, geometric_desc,
            material_desc, camera_lighting, appearance, constraints
        )

        # 6. Negative Prompt tailored to the product
        negative_prompt = self._build_negative_prompt(product_name, category, base_mat)

        return {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "structural_description": structural_desc,
            "geometric_feature_description": geometric_desc,
            "material_texture_description": material_desc,
            "camera_lighting_instructions": camera_lighting,
            "product_name": product_name,
            "category": category
        }

    def _build_structural_description(self, product_name: str, geom: Dict[str, Any], components: List[str], features: List[str]) -> str:
        shape = geom.get("shape_type", "cylindrical")
        comp_str = ", ".join(components) if components else "monolithic body"
        feat_str = ", ".join(features) if features else "engineered mating faces"
        return f"{shape} architecture for {product_name}, incorporating {comp_str}, featuring {feat_str}"

    def _build_geometric_description(self, dims: Dict[str, Any], geom: Dict[str, Any], features: List[str]) -> str:
        tokens = []
        unit = dims.get("unit", "mm")
        if dims.get("outer_diameter") is not None:
            tokens.append(f"{dims['outer_diameter']} {unit} outer diameter")
        if dims.get("inner_diameter") is not None or dims.get("central_bore") is not None:
            bore = dims.get("central_bore") or dims.get("inner_diameter")
            tokens.append(f"{bore} {unit} central bore")
        if dims.get("teeth_count") is not None:
            tokens.append(f"{dims['teeth_count']} teeth")
        if dims.get("mounting_holes_count") is not None:
            tokens.append(f"{dims['mounting_holes_count']} mounting holes")
        if dims.get("length") is not None and dims.get("width") is not None:
            dim_str = f"{dims['length']} x {dims['width']}"
            if dims.get("height") is not None:
                dim_str += f" x {dims['height']}"
            tokens.append(f"{dim_str} {unit} envelope")
        elif dims.get("thickness") is not None:
            tokens.append(f"{dims['thickness']} {unit} axial thickness")

        symmetry = geom.get("symmetry", "radial")
        tokens.append(f"{symmetry} symmetry")
        return ", ".join(tokens) if tokens else "standard precision metric engineering envelope"

    def _build_material_description(self, base_mat: str, finish: str, color: str, treatment: str = None) -> str:
        base_desc = self.MATERIAL_VISUAL_CUES.get(base_mat, f"{base_mat} with authentic solid density")
        treatment_str = f", {treatment.lower()} condition" if treatment else ""
        return f"{base_desc}{treatment_str}, {finish} surface finish, {color} coloration, realistic physical specular roughness and micro-facet reflection"

    def _build_camera_lighting(self, category: str, viewing_angle: str, appearance: str) -> str:
        lighting = self.CATEGORY_LIGHTING.get(category, "clean high-key 5500K studio softbox illumination, calibrated rim accents, soft contact ambient occlusion")
        return f"{viewing_angle}, {lighting}, neutral studio background, 85mm industrial macro lens perspective, crisp focus depth"

    def _build_positive_prompt(
        self,
        product_name: str,
        category: str,
        structural: str,
        geometric: str,
        material: str,
        camera: str,
        appearance: str,
        constraints: List[str]
    ) -> str:
        constraints_str = f", {', '.join(constraints)}" if constraints else ""
        prompt_parts = [
            f"Professional studio photograph of a real industrial {product_name}",
            f"Category: {category}",
            f"Structure: {structural}",
            f"Geometry: {geometric}",
            f"Material & Finish: {material}",
            f"Appearance & Lighting: {appearance}, {camera}{constraints_str}",
            "8k resolution, crisp photorealistic CAD render, sharp edges, clean engineering precision, isolated studio presentation"
        ]
        return ", ".join(prompt_parts)

    def _build_negative_prompt(self, product_name: str, category: str, base_mat: str) -> str:
        negatives = list(self.BASE_NEGATIVE_PROMPTS)
        combined = f"{product_name} {category}".lower()
        if "gear" in combined or "transmission" in combined:
            negatives.extend(["missing teeth", "irregular tooth spacing", "blunt teeth", "fused cogs"])
        if "fluid" in combined or "pump" in combined or "valve" in combined:
            negatives.extend(["leaking fluid", "cracked casing", "distorted flange", "misaligned port"])
        if "bearing" in combined:
            negatives.extend(["deformed ball bearings", "eccentric race", "broken cage"])
        if "enclosure" in combined or "box" in combined:
            negatives.extend(["bent sheet metal", "warped door", "missing hinges"])

        if "steel" in base_mat or "aluminum" in base_mat:
            negatives.extend(["plastic texture", "wood grain", "clay"])

        return ", ".join(dict.fromkeys(negatives))
