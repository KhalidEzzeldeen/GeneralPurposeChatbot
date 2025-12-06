"""
LLM Model Registry - Pre-configured models with performance characteristics.
"""

MODEL_REGISTRY = {
    "qwen2.5:7b-instruct": {
        "name": "Qwen2.5 7B Instruct",
        "size": "7B parameters",
        "speed": "Medium",
        "quality": "High",
        "description": "Balanced performance and quality. Good for general use.",
        "recommended": True
    },
    "qwen2.5:3b-instruct": {
        "name": "Qwen2.5 3B Instruct",
        "size": "3B parameters",
        "speed": "Fast",
        "quality": "High",
        "description": "Lighter version of Qwen2.5. 2-3x faster with similar quality. Recommended for faster responses.",
        "recommended": True
    },
    "llama3.2:3b": {
        "name": "Llama 3.2 3B",
        "size": "3B parameters",
        "speed": "Very Fast",
        "quality": "High",
        "description": "Very fast with excellent quality. Great for quick responses.",
        "recommended": True
    },
    "phi3:mini": {
        "name": "Phi-3 Mini",
        "size": "3.8B parameters",
        "speed": "Very Fast",
        "quality": "Good",
        "description": "Microsoft's efficient model. Very fast with good quality.",
        "recommended": True
    },
    "mistral:7b-instruct": {
        "name": "Mistral 7B Instruct",
        "size": "7B parameters",
        "speed": "Medium",
        "quality": "High",
        "description": "High quality with good speed. Popular choice.",
        "recommended": False
    },
    "gemma2:2b": {
        "name": "Gemma 2 2B",
        "size": "2B parameters",
        "speed": "Very Fast",
        "quality": "Good",
        "description": "Google's lightweight model. Fastest option with decent quality.",
        "recommended": True
    },
    "llama3.1:8b-instruct": {
        "name": "Llama 3.1 8B Instruct",
        "size": "8B parameters",
        "speed": "Medium",
        "quality": "Very High",
        "description": "Higher quality but slower. Best for complex tasks.",
        "recommended": False
    }
}

def get_recommended_models():
    """Get list of recommended models."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.get("recommended", False)}

def get_all_models():
    """Get all available models."""
    return MODEL_REGISTRY

def get_model_info(model_name):
    """Get information about a specific model."""
    return MODEL_REGISTRY.get(model_name, {
        "name": model_name,
        "size": "Unknown",
        "speed": "Unknown",
        "quality": "Unknown",
        "description": "Custom model",
        "recommended": False
    })

def get_fastest_models():
    """Get models sorted by speed (fastest first)."""
    speed_order = {"Very Fast": 1, "Fast": 2, "Medium": 3, "Slow": 4}
    return sorted(
        MODEL_REGISTRY.items(),
        key=lambda x: speed_order.get(x[1].get("speed", "Medium"), 3)
    )

