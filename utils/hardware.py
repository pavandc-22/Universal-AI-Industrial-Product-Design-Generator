"""
Hardware detection and environment diagnostics.
Detects GPU, PyTorch, Diffusers, and sets operating mode.
"""
import os
import sys
import logging
from typing import Dict, Any

logger = logging.getLogger("IndustrialDesignAI.hardware")

def detect_hardware() -> Dict[str, Any]:
    info = {
        "has_torch": False,
        "torch_version": None,
        "has_cuda": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
        "has_diffusers": False,
        "diffusers_version": None,
        "has_transformers": False,
        "recommended_engine": "PARAMETRIC_CAD_RENDERER",
        "is_demonstration_mode": True,
        "status_message": "Parametric High-Fidelity CAD & Physics Engine Active (Demonstration Mode)"
    }

    try:
        import torch
        info["has_torch"] = True
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["has_cuda"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error inspecting PyTorch: {e}")

    try:
        import diffusers
        info["has_diffusers"] = True
        info["diffusers_version"] = diffusers.__version__
    except ImportError:
        pass

    try:
        import transformers
        info["has_transformers"] = True
    except ImportError:
        pass

    if info["has_cuda"] and info["has_diffusers"]:
        info["recommended_engine"] = "DIFFUSERS_CUDA"
        info["is_demonstration_mode"] = False
        info["status_message"] = f"CUDA Acceleration Active ({info['cuda_device_name']})"
    elif info["has_diffusers"]:
        info["recommended_engine"] = "DIFFUSERS_CPU"
        info["is_demonstration_mode"] = False
        info["status_message"] = "Diffusers CPU Mode Active (Slow Inference)"
    else:
        info["recommended_engine"] = "PARAMETRIC_CAD_RENDERER"
        info["is_demonstration_mode"] = True
        info["status_message"] = "Parametric CAD & Procedural Physics Shading Active (Demonstration Mode)"

    return info
