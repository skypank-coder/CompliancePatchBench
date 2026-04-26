import json

from environment.patch_env import CompliancePatchEnv
from project.agent import AgentConfig, ComplianceAgent


def _simple_task():
    return {
        "task_id": "test_debug_fix",
        "description": "Disable debug mode",
        "framework": ["GDPR"],
        "difficulty": "easy",
        "adversarial": False,
        "codebase": {
            "settings.py": "import os\n\nDEBUG = True\n"
        },
        "violations": [
            {
                "file": "settings.py",
                "rule_id": "GDPR-ART32",
                "severity": "high",
                "line_start": 3,
                "line_end": 3,
            }
        ],
        "max_steps": 4,
        "file_reads_remaining": 1,
    }


def test_choose_action_retries_then_returns_valid_json():
    calls = {"n": 0}

    def llm(_messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return "not json"
        return '{"action_type":"read_file","path":"settings.py"}'

    agent = ComplianceAgent(llm=llm, config=AgentConfig(max_retries=2))
    obs = {
        "available_files": ["settings.py"],
        "violations": _simple_task()["violations"],
        "file_reads_remaining": 1,
        "ci_results": [],
    }
    messages = [{"role": "system", "content": agent.system_prompt}]

    action, _raw, used_fallback, retries, _logprob, extra = agent._choose_action(messages, obs)

    assert action == {"action_type": "read_file", "path": "settings.py"}
    assert used_fallback is False
    assert retries == 2
    assert extra["parse_failures"] == 2


def test_choose_action_fallback_is_safe_valid_json():
    agent = ComplianceAgent(
        llm=lambda _messages: "still not json",
        config=AgentConfig(max_retries=2),
    )
    obs = {
        "available_files": ["settings.py"],
        "violations": _simple_task()["violations"],
        "file_reads_remaining": 1,
        "ci_results": [],
    }
    messages = [{"role": "system", "content": agent.system_prompt}]

    action, _raw, used_fallback, retries, _logprob, extra = agent._choose_action(messages, obs)

    assert used_fallback is True
    assert retries == 2
    assert extra["invalid_output_rate"] == 1.0
    assert action == {"action_type": "read_file", "path": "settings.py"}


def test_invalid_json_never_reaches_env_and_does_not_get_format_penalty():
    responses = iter([
        "bad output",
        json.dumps({"action_type": "read_file", "path": "settings.py"}),
        json.dumps({"action_type": "write_patch", "file": "settings.py", "line_start": 3, "line_end": 3, "new_code": "DEBUG = False"}),
        json.dumps({"action_type": "run_ci"}),
    ])

    agent = ComplianceAgent(llm=lambda _messages: next(responses), config=AgentConfig(max_retries=2, max_steps=4))
    env = CompliancePatchEnv()
    result = agent.run(env, _simple_task())

    assert result.error is None
    assert result.steps[0].parsed_action == {"action_type": "read_file", "path": "settings.py"}
    assert result.steps[0].reward >= -0.02
    assert result.final_score >= 1.5
