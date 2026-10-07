"""Start DENSO A2 with optional per-developer OpenAI-compatible settings."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse


PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
PLACEHOLDER_MARKERS = ("YOUR_", "YOUR_REACHABLE_PROVIDER")


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without printing or persisting values."""
    parsed: dict[str, str] = {}
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(
                f"Invalid .env syntax at line {number}: expected KEY=VALUE"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid .env key name at line {number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        parsed[key] = value
    return parsed


def build_child_environment(
    inherited: dict[str, str] | None = None,
    env_file: Path = ENV_FILE,
) -> tuple[dict[str, str], bool]:
    """Return child environment and whether LLM capture is enabled.

    An explicit local .env overrides inherited values. If the compatible
    provider is disabled, OpenAI client variables are removed from the child
    so an inherited private endpoint cannot be selected accidentally.
    """
    child_env = dict(os.environ if inherited is None else inherited)
    if env_file.is_file():
        child_env.update(read_env_file(env_file))

    raw_enabled = child_env.get("OPENAI_COMPAT_ENABLED", "false").strip().lower()
    if raw_enabled in TRUE_VALUES:
        enabled = True
    elif raw_enabled in FALSE_VALUES:
        enabled = False
    else:
        raise ValueError(
            "OPENAI_COMPAT_ENABLED must be true/false, yes/no, on/off, or 1/0."
        )

    if not enabled:
        child_env.pop("OPENAI_BASE_URL", None)
        child_env.pop("OPENAI_API_KEY", None)
        return child_env, False

    mappings = {
        "OPENAI_COMPAT_BASE_URL": "OPENAI_BASE_URL",
        "OPENAI_COMPAT_API_KEY": "OPENAI_API_KEY",
    }
    for source, destination in mappings.items():
        if child_env.get(source):
            child_env[destination] = child_env[source]

    required = (
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "OPENAI_COMPAT_API_KEY",
    )
    missing = [name for name in required if not child_env.get(name, "").strip()]
    if missing:
        raise ValueError("Missing configuration variable(s): " + ", ".join(missing))

    placeholders = [
        name
        for name in required
        if any(marker in child_env[name] for marker in PLACEHOLDER_MARKERS)
    ]
    if placeholders:
        raise ValueError(
            "Placeholder configuration variable(s): " + ", ".join(placeholders)
        )

    parsed_url = urlparse(child_env["OPENAI_COMPAT_BASE_URL"])
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("OPENAI_COMPAT_BASE_URL must be a valid HTTP(S) URL.")
    return child_env, True


def main() -> int:
    try:
        child_env, enabled = build_child_environment()
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    if enabled:
        print("Starting DENSO A2 with the configured OpenAI-compatible provider.")
    else:
        print(
            "Starting DENSO A2 without LLM capture. Browsing, retrieval, and "
            "reports remain available."
        )
    print("Credentials are not displayed. Browser URL is provided by Streamlit.")
    try:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(PROJECT_DIR / "app.py"),
                *sys.argv[1:],
            ],
            cwd=PROJECT_DIR,
            env=child_env,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
