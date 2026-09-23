"""
CLI Entrypoint for RAG Evaluation Harness (Phase 9)
Usage: python scripts/run_eval.py --config full_pipeline --dataset standard_rag_benchmark
"""

import argparse
import asyncio
import json
import sys
from backend.app.eval.eval_dataset import generate_standard_benchmark_dataset
from backend.app.eval.harness import EvaluationHarness


async def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG Evaluation Harness")
    parser.add_argument("--config", type=str, default="full_pipeline", choices=["vector_only", "hybrid", "hybrid_rerank", "full_pipeline"])
    parser.add_argument("--dataset", type=str, default="standard_rag_benchmark")
    parser.add_argument("--tenant-id", type=str, default="ee45761a-f471-4f41-97ff-697d927e0745")
    parser.add_argument("--user-id", type=str, default="ca3b120e-6936-4bb3-bee9-550f6b1dce80")
    args = parser.parse_args()

    print(f"===========================================================")
    print(f" Starting Evaluation Run: Config = '{args.config}'")
    print(f" Dataset = '{args.dataset}', Tenant ID = '{args.tenant_id}'")
    print(f"===========================================================")

    dataset = generate_standard_benchmark_dataset(tenant_id=args.tenant_id)
    print(f"Loaded dataset '{dataset.name}' (v{dataset.version})")
    print(f"Canonical Dataset Hash (SHA-256): {dataset.dataset_hash}")
    print(f"Total Questions: {len(dataset.questions)}")

    harness = EvaluationHarness()
    summary = await harness.run_evaluation_dataset(
        dataset=dataset,
        config_name=args.config,
        user_id=args.user_id,
        tenant_id=args.tenant_id
    )

    print("\n================ EVALUATION RESULTS SUMMARY ================")
    print(f"Run ID:            {summary['run_id']}")
    print(f"Status:            {summary['status']}")
    print(f"Questions Tested:  {summary['question_count']}")
    print(f"Average Recall@3:  {summary['avg_recall'] * 100:.2f}%")
    print(f"Average Precision: {summary['avg_precision'] * 100:.2f}%")
    print(f"Average MRR:       {summary['avg_mrr']:.4f}" if summary['avg_mrr'] is not None else "Average MRR:       N/A")
    print(f"Faithfulness:      {summary['avg_faithfulness'] * 100:.2f}%")
    print(f"Token F1 Score:    {summary['avg_token_f1'] * 100:.2f}%")
    print("============================================================\n")

    # Failure Taxonomy Breakdown
    taxonomy_counts = {}
    for r in summary["results"]:
        cat = r["failure_category"]
        taxonomy_counts[cat] = taxonomy_counts.get(cat, 0) + 1

    print("Failure Taxonomy Breakdown:")
    for cat, count in sorted(taxonomy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat:<32}: {count}")

    print("\nEvaluation Harness Run Completed Successfully.")


if __name__ == "__main__":
    asyncio.run(main())
