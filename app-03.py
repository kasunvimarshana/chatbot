import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable
import requests
from urllib.parse import quote
import time
from functools import lru_cache
import logging
from dataclasses import dataclass
from enum import Enum

# ======================
# ENUMS AND RESPONSE TYPES
# ======================

class ModelType(Enum):
    """Classification of model types"""
    PURE_AI = 1
    API_DEPENDENT = 2
    HYBRID = 3

@dataclass
class ModelResponse:
    """Standardized model response format"""
    success: bool
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

# ======================
# ABSTRACT BASE CLASSES
# ======================

class AIModel(ABC):
    """Base interface for all AI models"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_type = ModelType.PURE_AI

    @abstractmethod
    def predict(self, input_data: Any) -> ModelResponse:
        """Run prediction logic"""
        pass

    @abstractmethod
    def is_appropriate(self, input_data: Any) -> bool:
        """Determine if model should handle input"""
        pass

    def configure(self, **kwargs: Any) -> None:
        """Optional configuration"""
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

class APIDependentMixin(ABC):
    """Mixin for models calling external APIs"""

    def __init__(self):
        self.api_config = self._get_api_config()
        self._api_session: Optional[requests.Session] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def _get_api_config(self) -> Dict[str, Any]:
        """Return API configuration"""
        pass

    @property
    def api_session(self) -> requests.Session:
        if self._api_session is None:
            self._api_session = requests.Session()
            if 'headers' in self.api_config:
                self._api_session.headers.update(self.api_config['headers'])
        return self._api_session

    def _call_api(self, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Any:
        """Make an API call"""
        try:
            url = f"{self.api_config['base_url']}{endpoint}"
            response = self.api_session.request(
                method=self.api_config.get('method', 'GET'),
                url=url,
                params=params,
                json=data,
                timeout=self.api_config.get('timeout', 3)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API call failed: {str(e)}")
            raise

# ======================
# MODEL IMPLEMENTATIONS
# ======================

class SentimentAnalysisModel(AIModel):
    """Simple sentiment analysis model"""

    def __init__(self):
        super().__init__()
        self.positive_words = ['happy', 'great', 'awesome', 'love']
        self.negative_words = ['bad', 'terrible', 'awful', 'hate']

    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.split()) >= 2

    def predict(self, input_data: str) -> ModelResponse:
        text = input_data.lower()
        positive = sum(word in text for word in self.positive_words)
        negative = sum(word in text for word in self.negative_words)

        if positive > negative:
            sentiment = 'positive'
            score = positive / len(self.positive_words)
        elif negative > positive:
            sentiment = 'negative'
            score = negative / len(self.negative_words)
        else:
            sentiment = 'neutral'
            score = 0.5

        return ModelResponse(
            success=True,
            output={
                'sentiment': sentiment,
                'score': round(score, 2),
                'positive_matches': [w for w in self.positive_words if w in text],
                'negative_matches': [w for w in self.negative_words if w in text]
            },
            metadata={
                'model_type': str(self.model_type),
                'timestamp': time.time()
            }
        )

class WeatherModel(AIModel, APIDependentMixin):
    """Weather API model"""

    def __init__(self):
        AIModel.__init__(self)
        APIDependentMixin.__init__(self)
        self.model_type = ModelType.API_DEPENDENT
        self.location_service = LocationService()

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://api.open-meteo.com/v1/',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and 'weather' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        try:
            location = self._extract_location(input_data)
            coords = self.location_service.get_coordinates(location)

            data = self._call_api(
                endpoint='forecast',
                params={
                    'latitude': coords['lat'],
                    'longitude': coords['lon'],
                    'current_weather': True
                }
            )

            return self._process_response(data, location)
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error=str(e)
            )

    def _process_response(self, api_data: Dict[str, Any], location: str) -> ModelResponse:
        temp = api_data.get('current_weather', {}).get('temperature')
        return ModelResponse(
            success=True,
            output={
                'response': f"Weather in {location}: {temp}°C",
                'data': api_data
            },
            metadata={
                'model_type': str(self.model_type),
                'api_used': 'open-meteo',
                'timestamp': time.time()
            }
        )

    def _extract_location(self, text: str) -> str:
        return self.location_service.extract_location(text)

class LocationService:
    """Location extraction service"""

    def __init__(self):
        self.locations = {
            'london': {'lat': 51.5074, 'lon': -0.1278},
            'paris': {'lat': 48.8566, 'lon': 2.3522},
            'new york': {'lat': 40.7128, 'lon': -74.0060},
            'tokyo': {'lat': 35.6762, 'lon': 139.6503}
        }

    @lru_cache(maxsize=100)
    def get_coordinates(self, location: str) -> Dict[str, float]:
        return self.locations.get(location.lower(), self.locations['london'])

    def extract_location(self, text: str) -> str:
        for loc in self.locations:
            if loc in text.lower():
                return loc.capitalize()
        return 'London'

# ======================
# MODEL CONFIGURATION
# ======================

@dataclass
class ModelConfig:
    model_class: Type[AIModel]
    dependencies: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    preprocessors: Optional[List[Callable]] = None
    postprocessors: Optional[List[Callable]] = None

class ModelRegistry:
    """Registry for managing models"""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._initialized_models: Dict[str, AIModel] = {}
        self._dependency_graph: Dict[str, set] = {}

    def register_model(self, name: str, config: ModelConfig) -> None:
        if name in self._models:
            raise ValueError(f"Model {name} already registered")
        self._models[name] = config
        self._dependency_graph[name] = set(config.dependencies or [])

    def get_model(self, name: str) -> AIModel:
        if name not in self._initialized_models:
            self._initialize_model(name)
        return self._initialized_models[name]

    def _initialize_model(self, name: str) -> None:
        if name not in self._models:
            raise ValueError(f"Model {name} not registered")

        config = self._models[name]
        for dep in config.dependencies or []:
            if dep not in self._initialized_models:
                self._initialize_model(dep)

        model = config.model_class()
        if config.config and hasattr(model, 'configure'):
            model.configure(**config.config)
        self._initialized_models[name] = model

    def get_execution_order(self) -> List[str]:
        return self._topological_sort()

    def _topological_sort(self) -> List[str]:
        result = []
        visited = set()
        temp = set()

        def visit(node: str):
            if node in temp:
                raise ValueError("Cyclic dependency detected")
            if node not in visited:
                temp.add(node)
                for neighbor in self._dependency_graph[node]:
                    visit(neighbor)
                temp.remove(node)
                visited.add(node)
                result.append(node)

        for node in self._dependency_graph:
            visit(node)
        return result

# ======================
# ORCHESTRATOR
# ======================

class ModelOrchestrator:
    """Main orchestrator to manage predictions"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(self, input_data: Any) -> List[ModelResponse]:
        execution_order = self.registry.get_execution_order()
        responses: List[ModelResponse] = []

        for model_name in execution_order:
            model = self.registry.get_model(model_name)
            if model.is_appropriate(input_data):
                try:
                    response = model.predict(input_data)
                    responses.append(response)
                    self.logger.info(f"Model {model_name} handled input.")
                except Exception as e:
                    self.logger.error(f"Model {model_name} failed: {e}")
                    responses.append(ModelResponse(
                        success=False,
                        output={},
                        metadata={'model': model_name},
                        error=str(e)
                    ))
        return responses

# ======================
# USAGE EXAMPLE
# ======================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    registry = ModelRegistry()
    registry.register_model('sentiment', ModelConfig(model_class=SentimentAnalysisModel))
    registry.register_model('weather', ModelConfig(model_class=WeatherModel))

    orchestrator = ModelOrchestrator(registry)

    user_input = "Can you tell me the weather in Paris and also I feel happy today?"
    results = orchestrator.predict(user_input)

    for res in results:
        print(json.dumps(res.__dict__, indent=4))
