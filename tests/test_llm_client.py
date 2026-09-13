from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
