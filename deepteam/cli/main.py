import yaml
import typer

from . import config

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

if __name__ == "__main__":
    app()
