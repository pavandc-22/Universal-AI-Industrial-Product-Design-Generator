"""
Universal Industrial Product Specification Parser.
Converts free-form natural language and engineering descriptions into structured JSON.
Supports external LLM API calls with a comprehensive local rule-based fallback parser.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
import jsonschema

logger = logging.getLogger("IndustrialDesignAI.parser")

SPEC_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "IndustrialProductSpecification",
    "type": "object",
    "required": [
        "product_name",
        "category",
        "primary_geometry",
        "dimensions",
        "material",
        "features",
        "surface_finish",
        "missing_information"
    ],
    "properties": {
        "product_name": {"type": "string"},
        "category": {"type": "string"},
        "primary_geometry": {
            "type": "object",
            "properties": {
                "shape_type": {"type": "string"},
                "symmetry": {"type": "string"},
                "aspect_ratio": {"type": "string"}
            }
        },
        "dimensions": {
            "type": "object",
            "properties": {
                "unit": {"type": "string"},
                "outer_diameter": {"type": ["number", "null"]},
                "inner_diameter": {"type": ["number", "null"]},
                "central_bore": {"type": ["number", "null"]},
                "length": {"type": ["number", "null"]},
                "width": {"type": ["number", "null"]},
                "height": {"type": ["number", "null"]},
                "thickness": {"type": ["number", "null"]},
                "teeth_count": {"type": ["integer", "null"]},
                "mounting_holes_count": {"type": ["integer", "null"]},
                "bolt_circle_diameter": {"type": ["number", "null"]},
                "raw_dimension_strings": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "material": {
            "type": "object",
            "properties": {
                "base_material": {"type": "string"},
                "grade": {"type": ["string", "null"]},
                "treatment": {"type": ["string", "null"]}
            }
        },
        "components": {
            "type": "array",
            "items": {"type": "string"}
        },
        "features": {
            "type": "array",
            "items": {"type": "string"}
        },
        "quantity": {"type": ["integer", "string", "null"]},
        "surface_finish": {"type": "string"},
        "color": {"type": "string"},
        "design_constraints": {
            "type": "array",
            "items": {"type": "string"}
        },
        "required_appearance": {"type": "string"},
        "viewing_angle": {"type": "string"},
        "missing_information": {
            "type": "array",
            "items": {"type": "string"}
        },
        "clarification_needed": {"type": "boolean"},
        "clarification_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "raw_input": {"type": "string"},
        "parsing_source": {"type": "string"}
    }
}


class UniversalSpecificationParser:
    """Universal parser for arbitrary industrial product specifications."""

    KNOWN_MATERIALS = [
        "stainless steel", "hardened steel", "carbon steel", "tool steel",
        "aluminum 6061", "aluminum 7075", "aluminum", "brass", "bronze",
        "cast iron", "ductile iron", "titanium", "inconel", "tungsten",
        "copper", "nickel alloy", "polycarbonate", "peek", "delrin", "ptfe",
        "teflon", "nylon", "silicone", "carbon fiber", "ceramic", "zinc alloy"
    ]

    KNOWN_FINISHES = [
        "brushed", "machined", "mirror polished", "polished", "bead blasted",
        "sandblasted", "anodized", "black oxide", "powder coated", "electroplated",
        "zinc plated", "ground", "lapped", "cast as-is", "forged rough",
        "electropolished", "hard coat anodized", "passivated"
    ]

    KNOWN_COLORS = [
        "metallic silver", "matte black", "glossy black", "gunmetal grey",
        "industrial yellow", "safety orange", "industrial blue", "dark grey",
        "raw metallic", "bronze", "brass gold", "signal red", "white enamel"
    ]

    KNOWN_CATEGORIES = {
        "fluid": "Fluid Handling & Hydraulics",
        "pump": "Fluid Handling & Hydraulics",
        "valve": "Fluid Handling & Hydraulics",
        "gear": "Mechanical Power Transmission",
        "bearing": "Bearings & Rotary Motion",
        "shaft": "Mechanical Power Transmission",
        "motor": "Electromechanical Systems",
        "actuator": "Electromechanical Systems",
        "enclosure": "Electrical & Industrial Enclosures",
        "box": "Electrical & Industrial Enclosures",
        "cabinet": "Electrical & Industrial Enclosures",
        "flange": "Piping & Structural Couplings",
        "connector": "Fittings, Fasteners & Interconnects",
        "manifold": "Fluid Handling & Hydraulics",
        "nozzle": "Fluid Handling & Spray Systems",
        "heatsink": "Thermal Management",
        "bracket": "Structural Hardware",
        "housing": "Machine Housing & Enclosures",
        "turbine": "Turbomachinery & Power Generation",
        "compressor": "Fluid Handling & Compression"
    }

    def __init__(self, api_key: Optional[str] = None, provider: str = "local"):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider
        logger.info(f"Initialized UniversalSpecificationParser with provider: {provider}")

    def parse(self, text: str, user_hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main entry point: attempts LLM parsing if configured, otherwise falls back to robust local parsing."""
        if not text or not text.strip():
            return self._create_empty_spec("Empty specification provided")

        spec = None
        if self.api_key and self.provider != "local":
            try:
                spec = self._parse_with_llm(text)
            except Exception as e:
                logger.warning(f"LLM parsing failed, using local rule fallback: {e}")

        if not spec:
            spec = self._parse_local(text)

        # Merge explicit user hints if provided (e.g. from UI override inputs)
        if user_hints:
            spec = self._merge_user_hints(spec, user_hints)

        # Validate against JSON schema
        self.validate_spec(spec)
        return spec

    def validate_spec(self, spec: Dict[str, Any]) -> bool:
        """Validates structured JSON against schema."""
        try:
            jsonschema.validate(instance=spec, schema=SPEC_SCHEMA)
            return True
        except jsonschema.ValidationError as err:
            logger.warning(f"Specification JSON schema validation warning: {err.message}")
            return False

    def _parse_local(self, text: str) -> Dict[str, Any]:
        """Comprehensive rule-based universal parser for any industrial component."""
        cleaned_text = text.strip()
        lower_text = cleaned_text.lower()

        # 1. Product Name & Category detection
        product_name = self._extract_product_name(cleaned_text, lower_text)
        category = self._detect_category(lower_text, product_name)

        # 2. Material extraction
        material_info = self._extract_material(lower_text)

        # 3. Surface finish & Color
        surface_finish = self._extract_finish(lower_text)
        color = self._extract_color(lower_text, material_info["base_material"])

        # 4. Dimensions extraction (never invent specs)
        dimensions = self._extract_dimensions(cleaned_text, lower_text)

        # 5. Geometry analysis
        primary_geometry = self._extract_geometry(lower_text, product_name)

        # 6. Components and Features
        components = self._extract_components(lower_text, product_name)
        features = self._extract_features(lower_text, dimensions)

        # 7. Quantity & Constraints
        quantity = self._extract_quantity(lower_text)
        design_constraints = self._extract_constraints(lower_text)

        # 8. Appearance & Viewing Angle
        required_appearance = self._extract_appearance(lower_text)
        viewing_angle = self._extract_viewing_angle(lower_text)

        # 9. Missing information detection & Clarifications
        missing_info, clarif_needed, clarif_questions = self._check_missing_info(
            product_name, primary_geometry, dimensions, material_info
        )

        spec = {
            "product_name": product_name,
            "category": category,
            "primary_geometry": primary_geometry,
            "dimensions": dimensions,
            "material": material_info,
            "components": components,
            "features": features,
            "quantity": quantity,
            "surface_finish": surface_finish,
            "color": color,
            "design_constraints": design_constraints,
            "required_appearance": required_appearance,
            "viewing_angle": viewing_angle,
            "missing_information": missing_info,
            "clarification_needed": clarif_needed,
            "clarification_questions": clarif_questions,
            "raw_input": cleaned_text,
            "parsing_source": "local_rule_fallback"
        }
        return spec

    def _extract_product_name(self, text: str, lower_text: str) -> str:
        # Match common intro phrases: "generate an? ...", "create a? ...", "design an? ..."
        pattern = r'^(?:generate|create|design|produce|build|render|make)\s+(?:an?|the)?\s*([a-zA-Z0-9\-\s]+?)(?:\s+with|\s+having|\s+composed|\s+for|\s+featuring|,|\.|\n|$)'
        match = re.search(pattern, lower_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # clean adjectives that are materials or descriptors
            words = candidate.split()
            filtered = [w for w in words if w not in ["industrial", "custom", "high-grade", "precision", "heavy-duty"]]
            if filtered:
                return " ".join(filtered).title()
            return candidate.title()

        # Fallback: look for product noun phrases
        noun_candidates = [
            "centrifugal pump", "water pump", "gear pump", "mechanical gear", "spur gear",
            "helical gear", "bevel gear", "ball bearing", "roller bearing", "flange coupling",
            "butterfly valve", "gate valve", "ball valve", "stepper motor", "ac motor",
            "hydraulic cylinder", "pneumatic actuator", "electrical enclosure", "junction box",
            "heat exchanger", "exhaust manifold", "drive shaft", "connecting rod", "nozzle"
        ]
        for nc in noun_candidates:
            if nc in lower_text:
                return nc.title()

        first_line = text.split("\n")[0].split(".")[0]
        words = first_line.split()
        return " ".join(words[:4]).strip().title() if words else "Industrial Component"

    def _detect_category(self, lower_text: str, product_name: str) -> str:
        combined = f"{lower_text} {product_name.lower()}"
        for keyword, cat in self.KNOWN_CATEGORIES.items():
            if keyword in combined:
                return cat
        return "Custom Industrial Component"

    def _extract_material(self, lower_text: str) -> Dict[str, Any]:
        detected_mat = "steel"
        for mat in self.KNOWN_MATERIALS:
            if mat in lower_text:
                detected_mat = mat
                break

        # Grade search (e.g. 316L, 304, 6061-T6, AISI 4140)
        grade_match = re.search(r'\b(304|316l?|4140|1018|6061(?:-t6)?|7075(?:-t6)?|ti-6al-4v|a36)\b', lower_text)
        grade = grade_match.group(1).upper() if grade_match else None

        # Treatment
        treatment = None
        for tr in ["hardened", "quenched", "tempered", "case-hardened", "annealed", "carburized"]:
            if tr in lower_text:
                treatment = tr.capitalize()
                break

        return {
            "base_material": detected_mat,
            "grade": grade,
            "treatment": treatment
        }

    def _extract_finish(self, lower_text: str) -> str:
        for finish in self.KNOWN_FINISHES:
            if finish in lower_text:
                return finish
        if "matte" in lower_text:
            return "matte"
        if "glossy" in lower_text:
            return "glossy"
        return "machined"

    def _extract_color(self, lower_text: str, base_material: str) -> str:
        for c in self.KNOWN_COLORS:
            if c in lower_text:
                return c
        for c in ["black", "silver", "grey", "gray", "yellow", "orange", "blue", "red", "white", "gold"]:
            if re.search(rf'\b{c}\b', lower_text):
                return c
        if "brass" in base_material or "bronze" in base_material:
            return "metallic bronze"
        return "metallic silver"

    def _extract_dimensions(self, text: str, lower_text: str) -> Dict[str, Any]:
        dims: Dict[str, Any] = {
            "unit": "mm",
            "outer_diameter": None,
            "inner_diameter": None,
            "central_bore": None,
            "length": None,
            "width": None,
            "height": None,
            "thickness": None,
            "teeth_count": None,
            "mounting_holes_count": None,
            "bolt_circle_diameter": None,
            "raw_dimension_strings": []
        }

        # Determine unit
        if "inch" in lower_text or 'in.' in lower_text or '"' in lower_text:
            dims["unit"] = "inch"

        # Find raw dimension matches e.g. "100 mm", "25mm", "100 x 50 x 20 mm", "M8", "36 teeth"
        raw_dim_matches = re.findall(
            r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|inch|in|teeth|holes|deg|diameter|dia|od|id|bore|thick|wide|high|long)\b',
            lower_text
        )
        dims["raw_dimension_strings"] = list(set(raw_dim_matches))

        # Outer diameter (support "100 mm outer diameter", "outer diameter of 100 mm", "45 mm diameter", etc.)
        od_match_postfix = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|in|inch)?\s*(?:outer\s+diameter|outer\s+dia|od|diameter\b)', lower_text)
        od_match_prefix = re.search(r'(?:outer\s+diameter|outer\s+dia|od|diameter\s+of)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if od_match_postfix:
            dims["outer_diameter"] = float(od_match_postfix.group(1))
        elif od_match_prefix:
            dims["outer_diameter"] = float(od_match_prefix.group(1))

        # Central bore / inner diameter (support "25 mm central bore", "bore of 25 mm", etc.)
        bore_match_postfix = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|cm|in|inch)?\s*(?:central\s+bore|inner\s+diameter|inner\s+dia|id|bore\b|shaft\s+hole)', lower_text)
        bore_match_prefix = re.search(r'(?:central\s+bore|inner\s+diameter|inner\s+dia|id|bore|shaft\s+hole)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if bore_match_postfix:
            dims["central_bore"] = float(bore_match_postfix.group(1))
            dims["inner_diameter"] = float(bore_match_postfix.group(1))
        elif bore_match_prefix:
            dims["central_bore"] = float(bore_match_prefix.group(1))
            dims["inner_diameter"] = float(bore_match_prefix.group(1))

        # Teeth count
        teeth_match = re.search(r'(\d+)\s*(?:teeth|cogs|tooth)', lower_text)
        if teeth_match:
            dims["teeth_count"] = int(teeth_match.group(1))

        # Mounting holes
        holes_match = re.search(r'(\d+|four|six|eight|two|three|twelve)\s*(?:mounting\s+holes|holes|threaded\s+holes)', lower_text)
        if holes_match:
            word_to_num = {"two": 2, "three": 3, "four": 4, "six": 6, "eight": 8, "twelve": 12}
            val = holes_match.group(1)
            dims["mounting_holes_count"] = word_to_num.get(val, int(val) if val.isdigit() else 4)

        # Length, Width, Height, Thickness
        len_match = re.search(r'(?:length|long)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if len_match:
            dims["length"] = float(len_match.group(1))

        wid_match = re.search(r'(?:width|wide)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if wid_match:
            dims["width"] = float(wid_match.group(1))

        ht_match = re.search(r'(?:height|high)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if ht_match:
            dims["height"] = float(ht_match.group(1))

        thk_match = re.search(r'(?:thickness|thick)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)', lower_text)
        if thk_match:
            dims["thickness"] = float(thk_match.group(1))

        # Generic Dimension triple e.g. "300 x 200 x 150 mm"
        box_dims = re.search(r'(\d+(?:\.\d+)?)\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)', lower_text)
        if box_dims:
            dims["length"] = float(box_dims.group(1))
            dims["width"] = float(box_dims.group(2))
            dims["height"] = float(box_dims.group(3))

        return dims

    def _extract_geometry(self, lower_text: str, product_name: str) -> Dict[str, Any]:
        shape = "cylindrical"
        symmetry = "rotational"
        aspect = "balanced"

        if any(w in lower_text for w in ["box", "enclosure", "rectangular", "cube", "cabinet", "prism"]):
            shape = "rectangular prism"
            symmetry = "bilateral"
        elif any(w in lower_text for w in ["torus", "ring", "bearing", "washer", "annulus"]):
            shape = "toroidal"
            symmetry = "rotational"
        elif any(w in lower_text for w in ["gear", "sprocket", "cog"]):
            shape = "toothed circular disk"
            symmetry = "radial"
        elif any(w in lower_text for w in ["pump", "turbine", "scroll", "volute"]):
            shape = "spiral volute with cylindrical housing"
            symmetry = "asymmetric"
        elif any(w in lower_text for w in ["flange", "disk", "plate"]):
            shape = "circular flange disk"
            symmetry = "radial"
        elif any(w in lower_text for w in ["shaft", "rod", "spindle"]):
            shape = "elongated cylinder"
            symmetry = "axisymmetric"
            aspect = "high length-to-diameter"
        elif any(w in lower_text for w in ["valve", "manifold"]):
            shape = "compound fluid block with ports"
            symmetry = "orthogonal"

        return {
            "shape_type": shape,
            "symmetry": symmetry,
            "aspect_ratio": aspect
        }

    def _extract_components(self, lower_text: str, product_name: str) -> List[str]:
        comp_keywords = [
            "casing", "housing", "rotor", "stator", "impeller", "volute",
            "shaft", "seal", "inner ring", "outer ring", "rolling elements", "cage",
            "teeth", "hub", "keyway", "faceplate", "door", "hinge", "latch",
            "terminal block", "cable gland", "mounting bracket", "heat sink",
            "cooling fins", "inlet", "outlet", "nozzle", "flange", "drain plug"
        ]
        found = [ck for ck in comp_keywords if ck in lower_text]
        if not found:
            # fallback component inference based on product
            if "pump" in product_name.lower():
                found = ["volute casing", "impeller", "suction port", "discharge port", "mounting foot"]
            elif "gear" in product_name.lower():
                found = ["gear teeth", "central hub", "bore", "rim"]
            elif "bearing" in product_name.lower():
                found = ["outer race", "inner race", "rolling balls", "separator cage"]
            elif "enclosure" in product_name.lower():
                found = ["sheet body", "hinged cover", "knockouts", "wall brackets"]
            else:
                found = ["main body", "mounting interface"]
        return list(dict.fromkeys(found))

    def _extract_features(self, lower_text: str, dims: Dict[str, Any]) -> List[str]:
        feats = []
        if dims.get("mounting_holes_count"):
            feats.append(f"{dims['mounting_holes_count']} mounting holes")
        elif "mounting hole" in lower_text or "holes" in lower_text:
            feats.append("mounting holes")

        if dims.get("teeth_count"):
            feats.append(f"{dims['teeth_count']} precision cut teeth")

        keywords = [
            ("circular inlet", "circular suction inlet"),
            ("side outlet", "tangential discharge outlet"),
            ("keyway", "machined keyway slot"),
            ("chamfer", "beveled edge chamfers"),
            ("cooling fin", "external cooling fins"),
            ("rib", "structural reinforcement ribs"),
            ("o-ring", "o-ring groove sealing"),
            ("thread", "precision screw threading"),
            ("gasket", "flange gasket groove")
        ]
        for term, feature_desc in keywords:
            if term in lower_text:
                feats.append(feature_desc)

        if not feats:
            feats.append("precision engineered surfaces")
        return list(dict.fromkeys(feats))

    def _extract_quantity(self, lower_text: str) -> Optional[int]:
        qty_match = re.search(r'\b(?:qty|quantity|count)\s*(?:is|:)?\s*(\d+)\b', lower_text)
        return int(qty_match.group(1)) if qty_match else 1

    def _extract_constraints(self, lower_text: str) -> List[str]:
        constraints = []
        c_keywords = [
            ("compact", "compact industrial envelope"),
            ("high pressure", "high pressure rating"),
            ("heavy duty", "heavy duty durability"),
            ("corrosion resistant", "corrosion resistant environment"),
            ("high speed", "high rotational speed rating"),
            ("hermetic", "hermetically sealed")
        ]
        for kw, desc in c_keywords:
            if kw in lower_text:
                constraints.append(desc)
        return constraints

    def _extract_appearance(self, lower_text: str) -> str:
        if "catalogue" in lower_text or "catalog" in lower_text:
            return "professional product catalogue appearance, crisp studio lighting"
        if "technical drawing" in lower_text or "blueprint" in lower_text:
            return "technical engineering schematic styling"
        if "photorealistic" in lower_text or "realistic" in lower_text:
            return "photorealistic industrial studio render, 8k resolution"
        return "clean industrial product photography, high-contrast studio lighting"

    def _extract_viewing_angle(self, lower_text: str) -> str:
        if "isometric" in lower_text or "3d" in lower_text:
            return "isometric 3D perspective view"
        if "front" in lower_text:
            return "direct front elevation view"
        if "top" in lower_text or "plan" in lower_text:
            return "top-down orthographic plan view"
        if "side" in lower_text:
            return "side profile view"
        if "exploded" in lower_text:
            return "exploded assembly view"
        if "cross-section" in lower_text or "cutaway" in lower_text:
            return "cutaway cross-sectional view"
        return "isometric 3D perspective view"

    def _check_missing_info(
        self,
        product_name: str,
        geom: Dict[str, Any],
        dims: Dict[str, Any],
        mat: Dict[str, Any]
    ) -> tuple[List[str], bool, List[str]]:
        missing = []
        questions = []

        shape = geom.get("shape_type", "")
        # Check dimensional completeness without inventing values
        if "cylindrical" in shape or "circular" in shape or "gear" in product_name.lower():
            if dims.get("outer_diameter") is None:
                missing.append("Outer diameter unspecified")
                questions.append("What is the desired outer diameter (e.g. in mm)?")
            if dims.get("thickness") is None and dims.get("length") is None and dims.get("height") is None:
                missing.append("Axial width or thickness unspecified")
                questions.append("What is the axial thickness or overall length?")

        if "gear" in product_name.lower() and dims.get("teeth_count") is None:
            missing.append("Gear tooth count (teeth_count) unspecified")
            questions.append("How many teeth should the gear feature?")

        if "pump" in product_name.lower():
            if dims.get("outer_diameter") is None and dims.get("length") is None:
                missing.append("Overall pump envelope dimensions unspecified")
                questions.append("What are the approximate outer dimensions or port sizes?")

        clarification_needed = len(missing) > 0
        return missing, clarification_needed, questions

    def _merge_user_hints(self, spec: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, Any]:
        """Override fields explicitly modified by user in the UI."""
        if "material" in hints and hints["material"]:
            spec["material"]["base_material"] = str(hints["material"])
        if "surface_finish" in hints and hints["surface_finish"]:
            spec["surface_finish"] = str(hints["surface_finish"])
        if "color" in hints and hints["color"]:
            spec["color"] = str(hints["color"])
        if "viewing_angle" in hints and hints["viewing_angle"]:
            spec["viewing_angle"] = str(hints["viewing_angle"])
        if "dimensions" in hints and isinstance(hints["dimensions"], dict):
            for k, v in hints["dimensions"].items():
                if v is not None and v != "":
                    spec["dimensions"][k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit() else v
        return spec

    def _create_empty_spec(self, reason: str) -> Dict[str, Any]:
        return {
            "product_name": "Unspecified Product",
            "category": "General Industrial",
            "primary_geometry": {"shape_type": "generic", "symmetry": "none", "aspect_ratio": "1:1"},
            "dimensions": {"unit": "mm", "raw_dimension_strings": []},
            "material": {"base_material": "steel", "grade": None, "treatment": None},
            "components": [],
            "features": [],
            "quantity": 1,
            "surface_finish": "machined",
            "color": "metallic silver",
            "design_constraints": [],
            "required_appearance": "clean industrial studio render",
            "viewing_angle": "isometric 3D perspective view",
            "missing_information": [reason],
            "clarification_needed": True,
            "clarification_questions": ["Please provide a description of the industrial component."],
            "raw_input": "",
            "parsing_source": "empty_fallback"
        }

    def _parse_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Call LLM API (OpenAI or compatible) for structured JSON extraction."""
        import requests
        prompt = (
            "You are an expert industrial mechanical engineering specification parser.\n"
            "Analyze the following user product request and output ONLY valid JSON adhering strictly to this schema:\n"
            f"{json.dumps(SPEC_SCHEMA, indent=2)}\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER invent dimensions that are not mentioned in the text. Put null for missing dimensions.\n"
            "2. Identify missing information explicitly in missing_information list.\n"
            "3. Extract precise engineering materials, treatments, geometry, and features.\n\n"
            f"User input:\n{text}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        endpoint = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise industrial engineering spec parser. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["parsing_source"] = "llm_api"
            parsed["raw_input"] = text
            return parsed
        else:
            logger.warning(f"LLM API returned status {resp.status_code}: {resp.text}")
            return None
