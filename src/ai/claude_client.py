"""Claude AI client for pattern generation."""

import json
from typing import Any, Optional

import anthropic

from ..core.config import Config
from ..core.exceptions import ConfigurationError
from ..core.logger import get_logger

logger = get_logger(__name__)


class ClaudeClient:
    """
    Client for interacting with Claude AI API.

    Handles API communication, rate limiting, and response parsing.
    """

    def __init__(self, config: Config):
        """
        Initialize Claude client.

        Args:
            config: Configuration object with AI settings
        """
        self.config = config

        if not config.ai.api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY not configured. "
                "Set it via environment variable or config file."
            )

        self.client = anthropic.Anthropic(api_key=config.ai.api_key)
        self.model = config.ai.model
        self.max_tokens = config.ai.max_tokens
        self.temperature = config.ai.temperature

        logger.info(f"Claude client initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from Claude.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)

        Returns:
            Generated response text
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.temperature

        try:
            response = self.client.messages.create(**kwargs)

            # Extract text from response
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            logger.debug(f"Claude response: {text[:200]}...")
            return text

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Generate a JSON response from Claude.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override temperature (optional)

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON
        """
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # Extract JSON from response
        json_str = self._extract_json(response)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from Claude: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may contain markdown code blocks.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string
        """
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find JSON object/array directly
        # Find first { or [
        obj_start = text.find("{")
        arr_start = text.find("[")

        if obj_start == -1 and arr_start == -1:
            return text.strip()

        if obj_start == -1:
            start = arr_start
            end_char = "]"
        elif arr_start == -1:
            start = obj_start
            end_char = "}"
        else:
            if obj_start < arr_start:
                start = obj_start
                end_char = "}"
            else:
                start = arr_start
                end_char = "]"

        # Find matching end
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{" or char == "[":
                depth += 1
            elif char == "}" or char == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

        # Return from start to end
        return text[start:].strip()

    def analyze_market(
        self,
        symbol: str,
        price_data: str,
        indicators: str,
        timeframe: str = "1min",
    ) -> dict[str, Any]:
        """
        Get market analysis from Claude.

        Args:
            symbol: Ticker symbol
            price_data: Formatted price data
            indicators: Formatted indicator data
            timeframe: Data timeframe

        Returns:
            Market analysis dictionary
        """
        from .prompts import PromptTemplates

        prompt = PromptTemplates.MARKET_ANALYSIS.format(
            symbol=symbol,
            timeframe=timeframe,
            num_bars=len(price_data.split("\n")),
            price_data=price_data,
            indicators=indicators,
        )

        return self.generate_json(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.3,
        )

    def explain_pattern(self, pattern: dict[str, Any]) -> str:
        """
        Get explanation of a pattern from Claude.

        Args:
            pattern: Pattern dictionary

        Returns:
            Pattern explanation
        """
        from .prompts import PromptTemplates

        prompt = PromptTemplates.EXPLAIN_PATTERN.format(
            pattern_json=json.dumps(pattern, indent=2)
        )

        return self.generate(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.5,
        )
