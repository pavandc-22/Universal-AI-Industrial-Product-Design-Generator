"""
Natural Language Design Modification Engine.
Interprets multi-turn user modification requests, modifies target specifications,
and strictly preserves all unchanged engineering parameters.
"""

import re
import copy
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("IndustrialDesignAI.modifier")

class NaturalLanguageDesignModifier:
    """Updates industrial specifications from free-form conversational requests."""

    def modify(self, current_spec: Dict[str, Any], modification_text: str) -> Tuple[Dict[str, Any], str]:
        """Applies requested changes while preserving all unchanged specs."""
        updated = copy.deepcopy(current_spec)
        lower_mod = modification_text.lower().strip()
        changes = []

        # 1. Material modification
        for mat in [
            "aluminum", "aluminium", "stainless steel", "hardened steel", "carbon steel",
            "brass", "bronze", "titanium", "cast iron", "copper", "polycarbonate", "peek"
        ]:
            if f"to {mat}" in lower_mod or f"in {mat}" in lower_mod or f"use {mat}" in lower_mod or lower_mod == mat:
                std_mat = "aluminum" if mat == "aluminium" else mat
                updated["material"]["base_material"] = std_mat
                changes.append(f"Updated base material to {std_mat.title()}")
                break

        # 2. Color modification
        for color in ["black", "yellow", "orange", "blue", "silver", "grey", "gray", "red", "gold", "white"]:
            if f"to {color}" in lower_mod or f"color {color}" in lower_mod or f"colour {color}" in lower_mod:
                color_name = f"matte {color}" if color in ["black", "grey"] else f"industrial {color}"
                updated["color"] = color_name
                changes.append(f"Updated coloration to {color_name.title()}")
                break

        # 3. Size / scale / compact
        if "larger" in lower_mod or "bigger" in lower_mod or "scale up" in lower_mod:
            dims = updated.get("dimensions", {})
            for key in ["outer_diameter", "length", "width", "height", "thickness"]:
                if dims.get(key) is not None:
                    dims[key] = round(dims[key] * 1.25, 1)
            changes.append("Scaled up dimensions by +25%")
        elif "smaller" in lower_mod or "scale down" in lower_mod:
            dims = updated.get("dimensions", {})
            for key in ["outer_diameter", "length", "width", "height", "thickness"]:
                if dims.get(key) is not None:
                    dims[key] = round(dims[key] * 0.8, 1)
            changes.append("Scaled down dimensions by -20%")
        elif "compact" in lower_mod:
            if "compact industrial envelope" not in updated.get("design_constraints", []):
                updated["design_constraints"].append("compact industrial envelope")
            changes.append("Enforced compact design constraint")

        # 4. Mounting holes / features addition
        hole_match = re.search(r'add\s+(\d+|two|three|four|six|eight)\s*(?:mounting\s+)?holes', lower_mod)
        if hole_match:
            word_map = {"two": 2, "three": 3, "four": 4, "six": 6, "eight": 8}
            val = hole_match.group(1)
            count = word_map.get(val, int(val) if val.isdigit() else 4)
            updated["dimensions"]["mounting_holes_count"] = count
            feat_desc = f"{count} mounting holes"
            if feat_desc not in updated["features"]:
                updated["features"].append(feat_desc)
            changes.append(f"Added {count} mounting holes")

        # 5. Shape modification
        if "cylindrical" in lower_mod:
            updated["primary_geometry"]["shape_type"] = "cylindrical"
            updated["primary_geometry"]["symmetry"] = "rotational"
            changes.append("Changed primary geometry to cylindrical")
        elif "rectangular" in lower_mod or "box" in lower_mod:
            updated["primary_geometry"]["shape_type"] = "rectangular prism"
            updated["primary_geometry"]["symmetry"] = "bilateral"
            changes.append("Changed primary geometry to rectangular prism")

        # 6. Viewing angle modification
        if "from the front" in lower_mod or "front view" in lower_mod:
            updated["viewing_angle"] = "direct front elevation view"
            changes.append("Adjusted viewing angle to front elevation")
        elif "top" in lower_mod or "top view" in lower_mod:
            updated["viewing_angle"] = "top-down orthographic plan view"
            changes.append("Adjusted viewing angle to top-down view")
        elif "isometric" in lower_mod or "3d" in lower_mod:
            updated["viewing_angle"] = "isometric 3D perspective view"
            changes.append("Adjusted viewing angle to isometric 3D")
        elif "cross-section" in lower_mod or "cutaway" in lower_mod:
            updated["viewing_angle"] = "cutaway cross-sectional view"
            changes.append("Adjusted viewing angle to cross-sectional cutaway")

        # 7. Remove component
        if "remove" in lower_mod or "delete" in lower_mod:
            for comp in list(updated.get("components", [])):
                if comp.lower() in lower_mod:
                    updated["components"].remove(comp)
                    changes.append(f"Removed component: {comp}")
            for feat in list(updated.get("features", [])):
                if any(w in lower_mod for w in feat.lower().split()):
                    updated["features"].remove(feat)
                    changes.append(f"Removed feature: {feat}")

        if not changes:
            changes.append(f"Regenerated design variant with request: '{modification_text}'")

        summary = "; ".join(changes)
        logger.info(f"Design modified: {summary}")
        return updated, summary
