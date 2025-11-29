"""
Quick LLM Comparison Test for RAG Pipeline.
Tests different LLMs on a small sample from validation set.

Usage:
    python scripts/test_llm_comparison.py --llm tinyllama
    python scripts/test_llm_comparison.py --llm mistral
    python scripts/test_llm_comparison.py --llm phi3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import List, Dict, Any

from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever
from src.generator import RAGGenerator, MODEL_CONFIGS


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import re
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = ' '.join(s.split())
    return s


def check_answer(prediction: str, gold_answers: List[str]) -> bool:
    """Check if prediction matches any gold answer."""
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        if normalize_answer(gold) in pred_norm or pred_norm in normalize_answer(gold):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Test LLM on validation samples")
    parser.add_argument("--llm", type=str, default="tinyllama",
                        choices=["tinyllama", "mistral", "phi3"],
                        help="LLM to test")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to test")
    parser.add_argument("--sparse_index", type=str, default="bm25_index.pkl",
                        help="Path to BM25 index")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Testing LLM: {args.llm}")
    print(f"Model ID: {MODEL_CONFIGS.get(args.llm, {}).get('model_id', args.llm)}")
    print("=" * 60)

    # Load retriever
    print("\nLoading BM25 retriever...")
    retriever = SparseRetriever()
    
    if not os.path.exists(args.sparse_index):
        print(f"ERROR: Index not found at {args.sparse_index}")
        print("Run: python -m src.pipeline --mode sparse --force_rebuild")
        return
    
    retriever.load_index(args.sparse_index)
    print(f"Loaded index with {len(retriever.corpus)} documents")

    # Load generator
    print(f"\nLoading LLM: {args.llm}...")
    generator = RAGGenerator(model_name=args.llm)
    print(f"Loaded: {generator.model_id}")

    # Load validation data
    print("\nLoading validation data...")
    data = load_trivia_qa()
    validation = data["validation"].select(range(args.num_samples))

    # Run test
    print(f"\nTesting on {args.num_samples} questions...\n")
    print("-" * 60)

    results: List[Dict[str, Any]] = []
    correct = 0

    for i, example in enumerate(validation):
        question = example["question"]
        gold_answers = example["answer"]["aliases"]

        # Retrieve context
        docs = retriever.retrieve(question, top_k=3)
        context = "\n\n".join([f"Title: {d['title']}\n{d['text']}" for d in docs])

        # Generate answer
        prediction = generator.generate_answer(question, context)

        # Check correctness
        is_correct = check_answer(prediction, gold_answers)
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{i+1:2d}] {status} Q: {question[:60]}...")
        print(f"       Pred: {prediction}")
        print(f"       Gold: {gold_answers[0] if gold_answers else 'N/A'}")
        print()

        results.append({
            "question": question,
            "prediction": prediction,
            "gold": gold_answers,
            "correct": is_correct
        })

    # Summary
    accuracy = correct / len(results) * 100
    print("-" * 60)
    print(f"\n📊 RESULTS for {args.llm.upper()}")
    print(f"   Accuracy: {correct}/{len(results)} = {accuracy:.1f}%")
    print("-" * 60)

    # Save results
    output_file = f"output/test_{args.llm}_{args.num_samples}samples.json"
    os.makedirs("output", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "llm": args.llm,
            "model_id": generator.model_id,
            "num_samples": args.num_samples,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

