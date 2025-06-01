"""
Enhanced model loader for enhancer with safetensors support and JSON configs.
Provides efficient loading without state_dict filtering when using safetensors.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

import torch
from safetensors.torch import load_file

from .enhancer_inference import EnhancerInference
from .hparams import HParams

logger = logging.getLogger(__name__)


class JSONConfig:
    """Simple config class that works with JSON files instead of OmegaConf."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        # Set attributes for easy access
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, JSONConfig(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def load(cls, json_path: Union[str, Path]) -> 'JSONConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def get(self, key: str, default=None):
        """Get config value with default."""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return self._config


def load_enhancer_from_safetensors(model_dir: Union[str, Path], device: str = "cpu") -> EnhancerInference:
    """Load enhancer model from safetensors format.
    
    Args:
        model_dir: Directory containing model.safetensors and config.json
        device: Device to load the model on
        
    Returns:
        Loaded EnhancerInference model
    """
    model_path = Path(model_dir)
    
    # Load config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = JSONConfig.load(config_path)
    
    # Create HParams with default values, then update from config
    hp = HParams()
    
    # For frozen dataclasses, we need to use object.__setattr__
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        if hasattr(hp, key):
            try:
                object.__setattr__(hp, key, value)
            except Exception:
                logger.warning(f"Could not set {key}={value} on HParams")
    
    # Create model
    model = EnhancerInference(hp)
    
    # Load weights from safetensors
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    state_dict = load_file(weights_path, device=device)
    
    # No filtering needed - safetensors already contains only enhancer weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    logger.info(f"Loaded enhancer model from safetensors: {model_path}")
    return model


def load_enhancer_model(run_dir: Union[str, Path, None], device: str = "cpu") -> EnhancerInference:
    """Load enhancer model from either safetensors or DeepSpeed checkpoint.
    
    Args:
        run_dir: Path to model directory (safetensors) or checkpoint directory (DeepSpeed)
        device: Device to load the model on
        
    Returns:
        Loaded EnhancerInference model
    """
    if run_dir is None:
        return create_default_enhancer(device)
    
    run_dir = Path(run_dir)
    
    # Check if this is a safetensors model directory
    if (run_dir / "model.safetensors").exists() and (run_dir / "config.json").exists():
        logger.info("Loading from safetensors format")
        return load_enhancer_from_safetensors(run_dir, device)
    
    # Fall back to DeepSpeed checkpoint loading
    logger.info("Loading from DeepSpeed checkpoint format")
    return load_enhancer_from_deepspeed(run_dir, device)


def load_enhancer_from_deepspeed(run_dir: Path, device: str = "cpu") -> EnhancerInference:
    """Load enhancer model from DeepSpeed checkpoint (legacy format).
    
    Args:
        run_dir: Path to the model checkpoint directory
        device: Device to load the model on
        
    Returns:
        Loaded EnhancerInference model ready for inference
    """
    # Load hparams
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        logger.warning(f"hparams.yaml not found in {run_dir}, using defaults")
        hp = HParams()
    else:
        hp = HParams.load(run_dir)
    
    # Create model
    model = EnhancerInference(hp)
    
    # Load the state dict from DeepSpeed checkpoint
    ckpt_path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if not ckpt_path.exists():
        logger.warning(f"Model checkpoint not found at {ckpt_path}, returning default model")
        return create_default_enhancer(device)
    
    state_dict = torch.load(ckpt_path, map_location="cpu")["module"]
    
    # Filter out denoiser weights since EnhancerInference doesn't have a denoiser
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('denoiser.')}
    
    model.load_state_dict(filtered_state_dict)
    model.eval()
    model.to(device)
    
    logger.info(f"Loaded enhancer model from DeepSpeed checkpoint: {run_dir}")
    return model


def create_default_enhancer(device: str = "cpu") -> EnhancerInference:
    """Create a default enhancer model with default hyperparameters.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Default EnhancerInference model (not trained)
    """
    hp = HParams()
    model = EnhancerInference(hp)
    model.eval()
    model.to(device)
    logger.info("Created default enhancer model")
    return model
