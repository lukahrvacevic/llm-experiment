from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import requests


@dataclass
class GenerationResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    generation_seconds: float
    peak_vram_mb: float | None
    raw_response: dict[str, Any]


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600.0,
        keep_alive: str = "30m",
        num_ctx: int | None = None,
        raw: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.raw = raw

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        for prediction_id in range(num_return_sequences):
            request_seed = None if seed is None else seed + prediction_id
            results.append(
                self._generate_once(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    seed=request_seed,
                )
            )
        return results

    def _generate_once(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": self.raw,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature if do_sample else 0.0,
                "top_p": top_p,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx

        start = perf_counter()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        generation_seconds = perf_counter() - start
        body = response.json()
        input_tokens = body.get("prompt_eval_count")
        output_tokens = body.get("eval_count")

        return GenerationResult(
            text=f"{prompt}{body.get('response', '')}",
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            generation_seconds=generation_seconds,
            peak_vram_mb=self._get_loaded_vram_mb(),
            raw_response=body,
        )

    def count_prompt_tokens(self, prompt: str) -> int | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": self.raw,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": 1,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        }
        if self.num_ctx is not None:
            payload["options"]["num_ctx"] = self.num_ctx

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        prompt_eval_count = body.get("prompt_eval_count")
        return int(prompt_eval_count) if prompt_eval_count is not None else None

    def _get_loaded_vram_mb(self) -> float | None:
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=10.0)
            response.raise_for_status()
            body = response.json()
        except requests.RequestException:
            return None

        models = body.get("models", [])
        for model_info in models:
            if model_info.get("name") != self.model:
                continue
            size_vram = model_info.get("size_vram")
            if isinstance(size_vram, int):
                return size_vram / (1024 * 1024)
        return None
