from datetime import datetime
import pytz
import re
import random
from typing import Dict, Optional
from .base_module import BaseModule
from config import DATA_DIR

class TimeModule(BaseModule):
    """Customizable time handling module"""
    
    def __init__(self, config_file: str = "time_config.json"):
        super().__init__(config_file, "time_module")
        self.timezone = self.config.get("default_timezone", "UTC")
        self._init_timezone_aliases()
        self._init_response_templates()

    def _validate_config(self) -> None:
        """Validate time module configuration"""
        required_keys = ["time_formats", "timezone_aliases", "response_templates"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _init_timezone_aliases(self) -> None:
        """Initialize timezone aliases"""
        self.timezone_aliases = self.config["timezone_aliases"]
        try:
            pytz.timezone(self.timezone)  # Validate default timezone
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.warning(f"Invalid default timezone: {self.timezone}")
            self.timezone = "UTC"

    def _init_response_templates(self) -> None:
        """Initialize response templates"""
        self.response_templates = self.config["response_templates"].get("time", [])
        if not self.response_templates:
            self.response_templates = ["The current time in {tz} is {time}"]

    def can_handle(self, input_text: str) -> bool:
        """Check if input is a time query"""
        patterns = self.config.get("time_patterns", [])
        return any(re.search(pattern, input_text.lower()) for pattern in patterns)

    def set_timezone(self, timezone: str) -> bool:
        """Set the active timezone"""
        normalized_tz = timezone.lower()
        if normalized_tz in self.timezone_aliases:
            timezone = self.timezone_aliases[normalized_tz]
        
        try:
            pytz.timezone(timezone)
            self.timezone = timezone
            self.logger.info(f"Timezone set to {timezone}")
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.warning(f"Unknown timezone: {timezone}")
            return False

    def get_current_time(self, format_key: str = "12h") -> str:
        """Get formatted current time"""
        time_format = self.config["time_formats"].get(format_key, "%I:%M %p")
        tz = pytz.timezone(self.timezone)
        return datetime.now(tz).strftime(time_format)

    def extract_timezone(self, text: str) -> Optional[str]:
        """Extract timezone from text if specified"""
        # Improved regex to more accurately find timezone mentions
        tz_match = re.search(
            r'(?:in|at)\s+((?:(?:the\s+)?\w+(?:\s+\w+)*)(?=\s|$|\.|,)', 
            text.lower()
        )
        
        if tz_match:
            potential_tz = tz_match.group(1).strip()
            # Filter out false positives like "the", "is the", etc.
            if (len(potential_tz) > 2 and 
                not re.match(r'^(?:the|is|a|an|at|in)$', potential_tz)):
                return potential_tz
        return None

    def process(self, input_text: str) -> Optional[str]:
        """Process time request and generate response"""
        if not self.can_handle(input_text):
            return None

        # Check if a specific timezone was requested
        requested_tz = self.extract_timezone(input_text)
        
        if requested_tz:
            # Try to set the requested timezone
            if not self.set_timezone(requested_tz):
                error_responses = self.config["response_templates"].get("timezone_errors", [])
                return random.choice(error_responses).format(tz=requested_tz) if error_responses else None
        else:
            # No specific timezone requested - use default
            requested_tz = self.timezone

        current_time = self.get_current_time()
        return random.choice(self.response_templates).format(
            tz=requested_tz,
            time=current_time
        )
