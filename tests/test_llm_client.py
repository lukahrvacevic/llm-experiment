from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from repoexec_baseline.generate import ordered_parallel_map
from repoexec_baseline.llm_client import GenerationResult, LLMClient


class LLMClientConcurrencyTest(unittest.TestCase):
    def run_generation(self, parallel_requests: int) -> tuple[list[int], int]:
        client = LLMClient(model="test-model", parallel_requests=parallel_requests)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def generate_once(**kwargs: object) -> GenerationResult:
            nonlocal active, max_active
            seed = int(kwargs["seed"])
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03 * (4 - seed))
            with lock:
                active -= 1
            return GenerationResult(
                text=str(seed),
                input_tokens=1,
                output_tokens=1,
                generation_seconds=0.01,
                peak_vram_mb=None,
                raw_response={},
            )

        with patch.object(client, "_generate_once", side_effect=generate_once):
            results = client.generate(prompt="prompt", max_new_tokens=1, num_return_sequences=3, seed=1)
        return [int(result.text) for result in results], max_active

    def test_serial_mode_is_default_compatible(self) -> None:
        seeds, max_active = self.run_generation(parallel_requests=1)
        self.assertEqual(seeds, [1, 2, 3])
        self.assertEqual(max_active, 1)

    def test_parallel_mode_preserves_prediction_order(self) -> None:
        seeds, max_active = self.run_generation(parallel_requests=3)
        self.assertEqual(seeds, [1, 2, 3])
        self.assertEqual(max_active, 3)

    def test_completion_callback_reports_every_prediction(self) -> None:
        client = LLMClient(model="test-model", parallel_requests=2)
        completed: list[int] = []

        def generate_once(**kwargs: object) -> GenerationResult:
            seed = int(kwargs["seed"])
            return GenerationResult(str(seed), 1, 1, 0.01, None, {})

        with patch.object(client, "_generate_once", side_effect=generate_once):
            results = client.generate(
                prompt="prompt",
                max_new_tokens=1,
                num_return_sequences=3,
                seed=10,
                on_prediction_complete=lambda prediction_id, _: completed.append(prediction_id),
            )
        self.assertEqual([result.text for result in results], ["10", "11", "12"])
        self.assertEqual(sorted(completed), [0, 1, 2])


class TaskConcurrencyTest(unittest.TestCase):
    def test_parallel_tasks_preserve_input_order(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def process(value: int) -> int:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03 * (4 - value))
            with lock:
                active -= 1
            return value

        results = list(ordered_parallel_map(process, [1, 2, 3], max_workers=2))
        self.assertEqual(results, [1, 2, 3])
        self.assertEqual(max_active, 2)


if __name__ == "__main__":
    unittest.main()
