import yaml
import typer

from . import config
from ..model_loader import load_model

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    Bias,
    Toxicity,
    Misinformation,
    IllegalActivity,
    PromptLeakage,
    PIILeakage,
    UnauthorizedAccess,
    ExcessiveAgency,
    Robustness,
    IntellectualProperty,
    Competition,
    GraphicContent,
    PersonalSafety,
    CustomVulnerability,
)
from deepteam.attacks.single_turn import (
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
)
from deepteam.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
)

app = typer.Typer(name="deepteam")

VULN_CLASSES = [
    Bias,
    Toxicity,
    Misinformation,
    IllegalActivity,
    PromptLeakage,
    PIILeakage,
    UnauthorizedAccess,
    ExcessiveAgency,
    Robustness,
    IntellectualProperty,
    Competition,
    GraphicContent,
    PersonalSafety,
]
VULN_MAP = {cls().get_name(): cls for cls in VULN_CLASSES}

ATTACK_CLASSES = [
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
]
ATTACK_MAP = {cls().get_name(): cls for cls in ATTACK_CLASSES}


def _build_vulnerability(cfg: dict):
    name = cfg.get("name")
    if not name:
        raise ValueError("Vulnerability entry missing 'name'")
    if name == "CustomVulnerability":
        return CustomVulnerability(
            name=cfg.get("custom_name", "Custom"),
            types=cfg.get("types"),
            custom_prompt=cfg.get("prompt"),
        )
    cls = VULN_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown vulnerability: {name}")
    return cls(types=cfg.get("types"))


def _build_attack(cfg: dict):
    name = cfg.get("name")
    if not name:
        raise ValueError("Attack entry missing 'name'")
    cls = ATTACK_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown attack: {name}")
    kwargs = {}
    if "weight" in cfg:
        kwargs["weight"] = cfg["weight"]
    if "type" in cfg:
        kwargs["type"] = cfg["type"]
    if "persona" in cfg:
        kwargs["persona"] = cfg["persona"]
    if "category" in cfg:
        kwargs["category"] = cfg["category"]
    if "turns" in cfg:
        kwargs["turns"] = cfg["turns"]
    if "enable_refinement" in cfg:
        kwargs["enable_refinement"] = cfg["enable_refinement"]
    return cls(**kwargs)


def _load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


async def _echo_callback(input: str) -> str:
    return input


@app.command()
def run(config: str):
    """Run a red teaming execution based on a YAML configuration"""
    cfg = _load_config(config)
    config.apply_env()

    target = cfg.get("target", {})

    red_teamer = RedTeamer(
        simulator_model=target.get("simulator_model", "gpt-3.5-turbo-0125"),
        evaluation_model=target.get("evaluation_model", "gpt-4o"),

        target_purpose=target.get("purpose", ""),
        async_mode=cfg.get("options", {}).get("run_async", True),
        max_concurrent=cfg.get("options", {}).get("max_concurrent", 10),
    )

    vulnerabilities_cfg = cfg.get("default_vulnerabilities", [])
    vulnerabilities_cfg += cfg.get("custom_vulnerabilities", [])
    vulnerabilities = [_build_vulnerability(v) for v in vulnerabilities_cfg]

    attacks = [_build_attack(a) for a in cfg.get("attacks", [])]

    risk = red_teamer.red_team(
        model_callback=_echo_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=cfg.get("options", {}).get(
            "attacks_per_vulnerability_type", 1
        ),
        ignore_errors=cfg.get("options", {}).get("ignore_errors", False),
    )

    red_teamer._print_risk_assessment()
    return risk


@app.command()
def login(api_key: str = typer.Argument(..., help="OpenAI API Key")):
    """Store API key for later runs."""
    config.set_key("OPENAI_API_KEY", api_key)
    typer.echo("API key saved.")


@app.command()
def logout():
    """Remove stored API key."""
    config.remove_key("OPENAI_API_KEY")
    typer.echo("Logged out.")


@app.command("set-azure-openai")
def set_azure_openai(
    openai_api_key: str = typer.Option(..., "--openai-api-key"),
    openai_endpoint: str = typer.Option(..., "--openai-endpoint"),
    openai_api_version: str = typer.Option(..., "--openai-api-version"),
    openai_model_name: str = typer.Option(..., "--openai-model-name"),
    deployment_name: str = typer.Option(..., "--deployment-name"),
    model_version: str = typer.Option(None, "--model-version"),
):
    """Configure Azure OpenAI credentials."""
    config.set_key("AZURE_OPENAI_API_KEY", openai_api_key)
    config.set_key("AZURE_OPENAI_ENDPOINT", openai_endpoint)
    config.set_key("OPENAI_API_VERSION", openai_api_version)
    config.set_key("AZURE_MODEL_NAME", openai_model_name)
    config.set_key("AZURE_DEPLOYMENT_NAME", deployment_name)
    if model_version:
        config.set_key("AZURE_MODEL_VERSION", model_version)
    config.set_key("USE_AZURE_OPENAI", "YES")
    config.set_key("USE_LOCAL_MODEL", "NO")
    typer.echo("Azure OpenAI configured.")


@app.command("unset-azure-openai")
def unset_azure_openai():
    """Remove Azure OpenAI configuration."""
    for key in [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "AZURE_MODEL_NAME",
        "AZURE_DEPLOYMENT_NAME",
        "AZURE_MODEL_VERSION",
        "USE_AZURE_OPENAI",
    ]:
        config.remove_key(key)
    typer.echo("Azure OpenAI unset.")


@app.command("set-local-model")
def set_local_model(
    model_name: str = typer.Argument(...),
    base_url: str = typer.Option(..., "--base-url"),
    api_key: str = typer.Option(None, "--api-key"),
):
    """Configure a local model endpoint."""
    config.set_key("LOCAL_MODEL_NAME", model_name)
    config.set_key("LOCAL_MODEL_BASE_URL", base_url)
    if api_key:
        config.set_key("LOCAL_MODEL_API_KEY", api_key)
    typer.echo("Local model configured.")


@app.command("unset-local-model")
def unset_local_model():
    """Remove local model configuration."""
    config.remove_key("LOCAL_MODEL_NAME")
    config.remove_key("LOCAL_MODEL_BASE_URL")
    config.remove_key("LOCAL_MODEL_API_KEY")
    typer.echo("Local model unset.")

@app.command("set-ollama")
def set_ollama(
    model_name: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Use a local Ollama model."""
    config.set_key("LOCAL_MODEL_NAME", model_name)
    config.set_key("LOCAL_MODEL_BASE_URL", base_url)
    config.set_key("LOCAL_MODEL_API_KEY", "ollama")
    config.set_key("USE_LOCAL_MODEL", "YES")
    config.set_key("USE_AZURE_OPENAI", "NO")
    typer.echo("Ollama model configured.")


@app.command("unset-ollama")
def unset_ollama():
    """Stop using local Ollama model."""
    for key in [
        "LOCAL_MODEL_NAME",
        "LOCAL_MODEL_BASE_URL",
        "LOCAL_MODEL_API_KEY",
        "USE_LOCAL_MODEL",
    ]:
        config.remove_key(key)
    typer.echo("Ollama model unset.")


@app.command("set-gemini")
def set_gemini(
    model_name: str = typer.Option(None, "--model-name"),
    google_api_key: str = typer.Option(None, "--google-api-key"),
    project_id: str = typer.Option(None, "--project-id"),
    location: str = typer.Option(None, "--location"),
):
    """Configure Gemini models via API key or Vertex AI."""
    if not google_api_key and not (project_id and location):
        typer.echo(
            "Provide --google-api-key or both --project-id and --location.", err=True
        )
        raise typer.Exit(code=1)
    config.set_key("USE_GEMINI_MODEL", "YES")
    if model_name:
        config.set_key("GEMINI_MODEL_NAME", model_name)
    if google_api_key:
        config.set_key("GOOGLE_API_KEY", google_api_key)
    else:
        config.set_key("GOOGLE_GENAI_USE_VERTEXAI", "YES")
        config.set_key("GOOGLE_CLOUD_PROJECT", project_id)
        config.set_key("GOOGLE_CLOUD_LOCATION", location)
    typer.echo("Gemini configured.")


@app.command("unset-gemini")
def unset_gemini():
    """Remove Gemini configuration."""
    for key in [
        "USE_GEMINI_MODEL",
        "GEMINI_MODEL_NAME",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]:
        config.remove_key(key)
    typer.echo("Gemini unset.")

if __name__ == "__main__":
    app()
