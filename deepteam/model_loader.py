from __future__ import annotations

from typing import Any, Dict, Union, Optional

from deepeval.models import (
    GPTModel,
    AzureOpenAIModel,
    OllamaModel,
    LocalModel,
    GeminiModel,
    AnthropicModel,
    AmazonBedrockModel,
    DeepEvalBaseLLM,
)
from deepeval.metrics.utils import initialize_model


def load_model(spec: Union[str, Dict[str, Any], None]) -> DeepEvalBaseLLM:
    """Construct a DeepEval model instance from a YAML spec or string."""
    if spec is None or isinstance(spec, str):
        # use deepeval helper which respects global config
        model, _ = initialize_model(spec)
        return model

    provider = str(spec.get("provider", "openai")).lower()
    model_name = spec.get("model") or spec.get("model_name")
    temperature = spec.get("temperature", 0)

    if provider == "openai":
        return GPTModel(model=model_name, temperature=temperature)

    if provider == "azure":
        return AzureOpenAIModel(
            model_name=model_name,
            deployment_name=spec.get("deployment_name"),
            azure_openai_api_key=spec.get("api_key"),
            openai_api_version=spec.get("openai_api_version"),
            azure_endpoint=spec.get("endpoint") or spec.get("azure_endpoint"),
            temperature=temperature,
        )

    if provider == "ollama":
        return OllamaModel(
            model_name=model_name,
            base_url=spec.get("base_url", "http://localhost:11434"),
            temperature=temperature,
        )

    if provider == "local":
        return LocalModel(
            model_name=model_name,
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key"),
            temperature=temperature,
        )

    if provider == "gemini":
        if spec.get("project") and spec.get("location"):
            return GeminiModel(
                model_name=model_name,
                project=spec.get("project"),
                location=spec.get("location"),
                temperature=temperature,
            )
        return GeminiModel(
            model_name=model_name,
            api_key=spec.get("api_key"),
            temperature=temperature,
        )

    if provider == "anthropic":
        return AnthropicModel(model=model_name, temperature=temperature)

    if provider == "bedrock":
        return AmazonBedrockModel(
            model_id=model_name,
            region_name=spec.get("region_name"),
            aws_access_key_id=spec.get("aws_access_key_id"),
            aws_secret_access_key=spec.get("aws_secret_access_key"),
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider: {provider}")
