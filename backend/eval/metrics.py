"""
Evaluation Metrics Calculator Stub (Section 12.2 & Section 15)
"""


class EvalMetricsCalculator:
    def compute_recall(self, retrieved_ids: list, ground_truth_ids: list) -> float:
        raise NotImplementedError

    def compute_faithfulness(self, answer: str, context_chunks: list) -> float:
        raise NotImplementedError
