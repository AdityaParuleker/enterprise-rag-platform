"""
Evaluation Failure Taxonomy Classifier Stub (Section 12.3 & Section 15)
Categories: RETRIEVAL_FAILURE | RERANKING_FAILURE | CONTEXT_FAILURE | GENERATION_FAILURE | CITATION_FAILURE | AUTHORIZATION_FAILURE
"""


class FailureAnalyzer:
    def classify_failure(self, eval_result_id: str) -> str:
        raise NotImplementedError
