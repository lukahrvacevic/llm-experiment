from __future__ import annotations

import subprocess
from time import perf_counter
from typing import Callable

import requests

from repoexec_baseline.llm_client import GenerationResult


class VLLMClient:
    """Synchronous client for a vLLM OpenAI-compatible completion server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        timeout_seconds: float = 1800.0,
        parallel_requests: int = 1,
        gpu_index: int = 0,
    ) -> None:
        if parallel_requests != 1:
            raise ValueError("VLLMClient only supports parallel_requests=1 in the serial experiment")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.parallel_requests = parallel_requests
        self.gpu_index = gpu_index

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int | None = None,
        on_prediction_complete: Callable[[int, GenerationResult], None] | None = None,
    ) -> list[GenerationResult]:
        if num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")

        results: list[GenerationResult] = []
        for prediction_id in range(num_return_sequences):
            request_seed = None if seed is None else seed + prediction_id
            result = self._generate_once(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                seed=request_seed,
            )
            results.append(result)
            if on_prediction_complete is not None:
                on_prediction_complete(prediction_id, result)
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
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "n": 1,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p if do_sample else 1.0,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed

        started = perf_counter()
        response = requests.post(
            f"{self.base_url}/completions",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        elapsed = perf_counter() - started
        body = response.json()

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("vLLM response did not contain a completion choice")
        generated_text = str(choices[0].get("text", ""))
        usage = body.get("usage") or {}

        return GenerationResult(
            text=f"{prompt}{generated_text}",
            input_tokens=self._optional_int(usage.get("prompt_tokens")),
            output_tokens=self._optional_int(usage.get("completion_tokens")),
            generation_seconds=elapsed,
            peak_vram_mb=self._gpu_memory_snapshot_mb(),
            raw_response=body,
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None

    def _gpu_memory_snapshot_mb(self) -> float | None:
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            first_line = process.stdout.strip().splitlines()[0]
            return float(first_line.strip())
        except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
            return None

