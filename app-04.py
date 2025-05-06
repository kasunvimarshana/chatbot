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

class CatFactModel(AIModel, APIDependentMixin):
    """Cat Facts API model"""

    def __init__(self):
        AIModel.__init__(self)
        APIDependentMixin.__init__(self)
        self.model_type = ModelType.API_DEPENDENT

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://cat-fact.herokuapp.com/',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return 'cat' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        try:
            data = self._call_api(endpoint='facts/random')
            return self._process_response(data)
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error=str(e)
            )

    def _process_response(self, api_data: Dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            success=True,
            output={
                'fact': api_data['text']
            },
            metadata={
                'model_type': str(self.model_type),
                'timestamp': time.time()
            }
        )

class JokeModel(AIModel, APIDependentMixin):
    """Joke API model"""

    def __init__(self):
        AIModel.__init__(self)
        APIDependentMixin.__init__(self)
        self.model_type = ModelType.API_DEPENDENT

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://official-joke-api.appspot.com/',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return 'joke' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        try:
            data = self._call_api(endpoint='random_joke')
            return self._process_response(data)
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error=str(e)
            )

    def _process_response(self, api_data: Dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            success=True,
            output={
                'setup': api_data['setup'],
                'punchline': api_data['punchline']
            },
            metadata={
                'model_type': str(self.model_type),
                'timestamp': time.time()
            }
        )

class QuoteModel(AIModel, APIDependentMixin):
    """Quote API model"""

    def __init__(self):
        AIModel.__init__(self)
        APIDependentMixin.__init__(self)
        self.model_type = ModelType.API_DEPENDENT

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://api.quotable.io/',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return 'quote' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        try:
            data = self._call_api(endpoint='random')
            return self._process_response(data)
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error=str(e)
            )

    def _process_response(self, api_data: Dict[str, Any]) -> ModelResponse:
        return ModelResponse(
            success=True,
            output={
                'quote': api_data['content'],
                'author': api_data['author']
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

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://api.open-meteo.com/v1/forecast',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return 'weather' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        try:
            location = self._extract_location(input_data)
            coords = self._get_coordinates(location)

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
        return 'London'  # A simplified location extractor

    def _get_coordinates(self, location: str) -> Dict[str, float]:
        location_map = {
            'london': {'lat': 51.5074, 'lon': -0.1278},
            'paris': {'lat': 48.8566, 'lon': 2.3522},
            'new york': {'lat': 40.7128, 'lon': -74.0060}
        }
        return location_map.get(location.lower(), location_map['london'])
    
class DictionaryModel(AIModel, APIDependentMixin):
    """Dictionary API model for retrieving word definitions"""

    def __init__(self):
        AIModel.__init__(self)
        APIDependentMixin.__init__(self)
        self.model_type = ModelType.API_DEPENDENT

    def _get_api_config(self) -> Dict[str, Any]:
        return {
            'base_url': 'https://api.dictionaryapi.dev/api/v2/entries/en/',
            'timeout': 5
        }

    def is_appropriate(self, input_data: Any) -> bool:
        return 'define' in input_data.lower() or 'meaning' in input_data.lower()

    def predict(self, input_data: str) -> ModelResponse:
        """Fetch word definition from the API"""
        try:
            # Extract the word from the input query
            word = self._extract_word(input_data)
            data = self._call_api(endpoint=word)
            return self._process_response(data)
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error=str(e)
            )

    def _extract_word(self, input_text: str) -> str:
        """Simple word extraction for this model (could be enhanced)"""
        words = input_text.split()
        return words[-1]  # Assume the last word is the one to be defined

    def _process_response(self, api_data: Dict[str, Any]) -> ModelResponse:
        """Process the API response into a standardized output format"""
        if isinstance(api_data, list) and api_data:
            word_data = api_data[0]
            definition = word_data['meanings'][0]['definitions'][0]['definition']
            return ModelResponse(
                success=True,
                output={
                    'word': word_data['word'],
                    'definition': definition
                },
                metadata={
                    'model_type': str(self.model_type),
                    'timestamp': time.time()
                }
            )
        else:
            return ModelResponse(
                success=False,
                output={},
                metadata={'model_type': str(self.model_type)},
                error="No definition found"
            )


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

    def __init__(self):
        self.registry = ModelRegistry()
        self.registry.register_model('cat_facts', ModelConfig(
            model_class=CatFactModel
        ))
        self.registry.register_model('jokes', ModelConfig(
            model_class=JokeModel
        ))
        self.registry.register_model('quotes', ModelConfig(
            model_class=QuoteModel
        ))
        self.registry.register_model('weather', ModelConfig(
            model_class=WeatherModel
        ))
        self.registry.register_model('dictionary', ModelConfig(
            model_class=DictionaryModel
        ))

    def process_input(self, input_data: str) -> ModelResponse:
        for model_name in self.registry.get_execution_order():
            model = self.registry.get_model(model_name)
            if model.is_appropriate(input_data):
                return model.predict(input_data)
        return ModelResponse(
            success=False,
            output={},
            metadata={},
            error="No suitable model found"
        )

# ======================
# USER INPUT HANDLING
# ======================

def get_user_input() -> str:
    """Prompt user for input with examples for all models"""
    print("Enter your query. Here are some examples for each model:")
    print("\n📘 CatFactModel:")
    print("  - 'Tell me a cat fact'")
    print("  - 'Do you know any facts about cats?'")
    print("\n😂 JokeModel:")
    print("  - 'Tell me a joke'")
    print("  - 'I need a laugh'")
    print("\n💬 QuoteModel:")
    print("  - 'Give me a quote'")
    print("  - 'Inspire me with a quote'")
    print("\n🌦️ WeatherModel:")
    print("  - 'What is the weather in London?'")
    print("  - 'Tell me the current weather in Paris'")
    print("\n📚 DictionaryModel:")
    print("  - 'Define serendipity'")
    print("  - 'What is the meaning of ephemeral?'")
    return input("\nYour query: ")

def main():
    """Main entry point for user interaction"""
    orchestrator = ModelOrchestrator()
    user_input = get_user_input()
    response = orchestrator.process_input(user_input)
    if response.success:
        print(f"Response: {json.dumps(response.output, indent=2)}")
    else:
        print(f"Error: {response.error}")

# ======================
# RUN EXAMPLE
# ======================

if __name__ == "__main__":
    main()
