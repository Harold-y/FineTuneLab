"""Local and OpenAI-compatible RLAIF judges."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, cast

from finetunelab.config import JudgeConfig
from finetunelab.devices import DeviceRuntime, activate_runtime, resolve_runtime
from finetunelab.errors import FineTuneLabError


def parse_judge_response(text: str, candidate_count: int) -> dict[str, Any]:
    """Parse a JSON decision, tolerating a surrounding Markdown code fence."""

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise FineTuneLabError("Judge response did not contain a JSON object")
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise FineTuneLabError(f"Judge returned invalid JSON: {exc}") from exc
    winner = value.get("winner")
    scores = value.get("scores")
    if not isinstance(winner, int) or not 0 <= winner < candidate_count:
        raise FineTuneLabError("Judge winner is outside the candidate range")
    if not isinstance(scores, list) or len(scores) != candidate_count:
        raise FineTuneLabError("Judge must return one score per candidate")
    value["scores"] = [float(score) for score in scores]
    value.setdefault("reason", "")
    return cast(dict[str, Any], value)


def _judge_messages(
    config: JudgeConfig, prompt: str, candidates: list[str]
) -> list[dict[str, str]]:
    body = "\n\n".join(f"Candidate {index}:\n{text}" for index, text in enumerate(candidates))
    return [
        {"role": "system", "content": config.prompt_template},
        {"role": "user", "content": f"User request:\n{prompt}\n\n{body}"},
    ]


@dataclass
class OpenAICompatibleJudge:
    config: JudgeConfig

    def compare(self, prompt: str, candidates: list[str]) -> dict[str, Any]:
        token = os.getenv(self.config.api_key_env)
        if not token:
            raise FineTuneLabError(
                f"Environment variable {self.config.api_key_env} is required by the API judge"
            )
        if not self.config.base_url:
            raise FineTuneLabError("An OpenAI-compatible judge requires judge.base_url")
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": _judge_messages(self.config, prompt, candidates),
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
        ).encode()
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                raw = response.read().decode()
        except OSError as exc:
            raise FineTuneLabError(f"Judge request failed: {exc}") from exc
        content = json.loads(raw)["choices"][0]["message"]["content"]
        result = parse_judge_response(content, len(candidates))
        result["raw_output"] = content
        return result


@dataclass
class LocalJudge:
    config: JudgeConfig
    runtime: DeviceRuntime | None = None

    def __post_init__(self) -> None:
        from transformers import pipeline

        runtime = self.runtime or resolve_runtime()
        activate_runtime(runtime)
        self._pipeline = pipeline(
            "text-generation",
            model=self.config.model,
            device=runtime.device,
            dtype=runtime.torch_dtype,
        )

    def compare(self, prompt: str, candidates: list[str]) -> dict[str, Any]:
        messages = _judge_messages(self.config, prompt, candidates)
        output = self._pipeline(messages, max_new_tokens=256, do_sample=False)
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            generated = generated[-1]["content"]
        result = parse_judge_response(str(generated), len(candidates))
        result["raw_output"] = generated
        return result


def build_judge(
    config: JudgeConfig, *, runtime: DeviceRuntime | None = None
) -> LocalJudge | OpenAICompatibleJudge:
    if config.backend == "local":
        return LocalJudge(config, runtime)
    return OpenAICompatibleJudge(config)
