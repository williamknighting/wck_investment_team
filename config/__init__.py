"""
Configuration module for AI Hedge Fund System
Provides centralized configuration management
"""
import yaml
from pathlib import Path
from typing import Dict, Any

from .agent_prompts import get_agent_prompt, get_config as get_prompt_config

def load_system_config() -> Dict[str, Any]:
    """Load system configuration from YAML file"""
    config_path = Path(__file__).parent / "system_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading system config: {e}")
        return {}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration section or full config"""
    config = load_system_config()
    
    if section:
        return config.get(section, {})
    return config

__all__ = ['get_agent_prompt', 'get_prompt_config', 'get_config', 'load_system_config']