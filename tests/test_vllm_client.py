from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from repoexec_baseline.vllm_client import VLLMClient


class VLLMClientTest(unittest.TestCase):
    def test_rejects_parallel_requests(self) -> None:
        with self.assertRaisesRegex(ValueError, "parallel_requests=1"):
            VLLMClient(model="test", parallel_requests=2)

    @patch.object(VLLMClient, "_gpu_memory_snapshot_mb", return_value=3210.0)
    @patch("repoexec_baseline.vllm_client.requests.post")
    def test_generates_candidates_serially(self, post: Mock, _: Mock) -> None:
        responses = []
        for prediction_id in range(3):
            response = Mock()
            response.json.return_value = {
                "choices": [{"text": f"result-{prediction_id}"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
            responses.append(response)
        post.side_effect = responses

        completed: list[int] = []
        client = VLLMClient(model="test-model")
        results = client.generate(
            prompt="prompt:",
            max_new_tokens=32,
            num_return_sequences=3,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            seed=42,
            on_prediction_complete=lambda prediction_id, _: completed.append(prediction_id),
        )

        self.assertEqual([result.text for result in results], ["prompt:result-0", "prompt:result-1", "prompt:result-2"])
        self.assertEqual(completed, [0, 1, 2])
        self.assertEqual([call.kwargs["json"]["seed"] for call in post.call_args_list], [42, 43, 44])
        self.assertTrue(all(call.kwargs["json"]["n"] == 1 for call in post.call_args_list))


if __name__ == "__main__":
    unittest.main()
