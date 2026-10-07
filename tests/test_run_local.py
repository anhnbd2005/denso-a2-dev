"""Focused tests for the portable local launcher configuration contract."""

from run_local import build_child_environment, read_env_file


class EnvFixture:
    def __init__(self, content: str | None):
        self.content = content

    def is_file(self) -> bool:
        return self.content is not None

    def read_text(self, **_kwargs) -> str:
        assert self.content is not None
        return self.content


def run_launcher_tests() -> None:
    child, enabled = build_child_environment(
        {
            "OPENAI_BASE_URL": "http://owner-private.invalid/v1",
            "OPENAI_API_KEY": "inherited-secret",
        },
        EnvFixture(None),
    )
    assert not enabled
    assert "OPENAI_BASE_URL" not in child
    assert "OPENAI_API_KEY" not in child

    configured = EnvFixture(
        "\n".join(
            [
                "OPENAI_COMPAT_ENABLED=true",
                "OPENAI_COMPAT_BASE_URL=https://team-provider.example/v1",
                "OPENAI_COMPAT_MODEL=team-model",
                "OPENAI_COMPAT_API_KEY=team-secret",
            ]
        )
    )
    child, enabled = build_child_environment(
        {
            "OPENAI_BASE_URL": "http://owner-private.invalid/v1",
            "OPENAI_API_KEY": "inherited-secret",
        },
        configured,
    )
    assert enabled
    assert child["OPENAI_BASE_URL"] == "https://team-provider.example/v1"
    assert child["OPENAI_API_KEY"] == "team-secret"
    assert child["OPENAI_COMPAT_MODEL"] == "team-model"

    complete_values = {
        "OPENAI_COMPAT_BASE_URL": "https://team-provider.example/v1",
        "OPENAI_COMPAT_MODEL": "team-model",
        "OPENAI_COMPAT_API_KEY": "team-secret",
    }
    for missing_name in complete_values:
        lines = ["OPENAI_COMPAT_ENABLED=true"] + [
            f"{name}={value}"
            for name, value in complete_values.items()
            if name != missing_name
        ]
        try:
            build_child_environment({}, EnvFixture("\n".join(lines)))
        except ValueError as exc:
            assert missing_name in str(exc)
        else:
            raise AssertionError(f"Missing {missing_name} was accepted")

    disabled = EnvFixture(
        "OPENAI_COMPAT_ENABLED=false\nOPENAI_COMPAT_API_KEY=local-secret\n"
    )
    child, enabled = build_child_environment({}, disabled)
    assert not enabled
    assert "OPENAI_API_KEY" not in child

    malformed_url = EnvFixture(
        "OPENAI_COMPAT_ENABLED=true\n"
        "OPENAI_COMPAT_BASE_URL=not-a-url\n"
        "OPENAI_COMPAT_MODEL=model\n"
        "OPENAI_COMPAT_API_KEY=secret\n"
    )
    try:
        build_child_environment({}, malformed_url)
    except ValueError as exc:
        assert "valid HTTP(S) URL" in str(exc)
    else:
        raise AssertionError("Malformed endpoint was accepted")

    try:
        build_child_environment(
            {}, EnvFixture("OPENAI_COMPAT_ENABLED=perhaps\n")
        )
    except ValueError as exc:
        assert "must be true/false" in str(exc)
    else:
        raise AssertionError("Invalid enable flag was accepted")

    try:
        read_env_file(EnvFixture("INVALID LINE"))
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("Malformed .env line was accepted")

    print("[PASS] PORTABLE RUN_LOCAL CONFIGURATION")


if __name__ == "__main__":
    run_launcher_tests()
