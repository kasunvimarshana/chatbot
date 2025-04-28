import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from config import DATA_DIR

class BaseModule(ABC):
    """Abstract base class for all chatbot modules"""
    
    def __init__(self, config_file: str, module_name: str):
        self.module_name = module_name
        self.logger = logging.getLogger(f"chatbot.{module_name}")
        self.config = self._load_config(config_file)
        self._validate_config()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load module configuration from JSON file"""
        try:
            config_path = DATA_DIR / config_file
            with open(config_path) as f:
                self.logger.info(f"Loading config from {config_path}")
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise RuntimeError(f"Could not load {self.module_name} config")

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate module configuration"""
        pass

    @abstractmethod
    def can_handle(self, input_text: str) -> bool:
        """Check if module can handle the input"""
        pass

    @abstractmethod
    def process(self, input_text: str) -> Optional[str]:
        """Process input and return response"""
        pass