#!/usr/bin/env python3
"""
LLM Comparison Script for Google Colab.
Compares TinyLlama vs Qwen2 on 100 samples.

Usage in Colab:
    !git clone https://github.com/Wojz12/RAG_LLM_project.git
    %cd RAG_LLM_project
    !python run_colab_comparison.py
"""

import json
import gc
import logging
from datetime import datetime
from typing import Dict, Any, Callable

import torch
from tqdm import tqdm

from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever
from src.re_ranker import Reranker
from src.generator import RAGGenerator
from src.evaluation import evaluate_predictions
from src.utils import prepare_corpus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
EVAL_SIZE = 100
CORPUS_SIZE = 10000
BM25_TOP_K = 10
RERANK_TOP_K = 3


def cleanup_memory():
    """Clean up GPU/CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    generator_fn: Callable[[str, str], str],
    eval_data,
    retriever: SparseRetriever,
    reranker: Reranker,
    model_name: str
) -> Dict[str, Any]:
    """Evaluate a single model on the RAG pipeline."""
    
    predictions = {}
    references = {}
    
    for example in tqdm(eval_data, desc=f"Evaluating {model_name}"):
        q = example["question"]
        q_id = example["question_id"]
        
        # Retrieve & rerank
        docs = retriever.retrieve(q, top_k=BM25_TOP_K)
        texts = [d["text"] for d in docs]
        contexts = reranker.rerank(q, texts, top_k=RERANK_TOP_K)
        context_str = "\n\n".join(contexts)
        
        # Generate answer
        answer = generator_fn(q, context_str)
        
        predictions[q_id] = answer
        references[q_id] = example["answer"]["aliases"]
    
    metrics = evaluate_predictions(predictions, references)
    return metrics


def main():
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info(f"Device: {device} ({gpu_name})")
    
    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    logger.info("Loading TriviaQA dataset...")
    data = load_trivia_qa()
    
    train_subset = data["train"].select(range(CORPUS_SIZE))
    eval_data = data["test"].select(range(EVAL_SIZE))
    
    logger.info(f"Corpus samples: {CORPUS_SIZE}")
    logger.info(f"Eval samples: {EVAL_SIZE}")
    
    # --------------------------------------------------------
    # 2. Build corpus and index
    # --------------------------------------------------------
    logger.info("Building corpus...")
    corpus = prepare_corpus(train_subset)
    logger.info(f"Corpus chunks: {len(corpus)}")
    
    logger.info("Building BM25 index...")
    sparse = SparseRetriever()
    sparse.build_index(corpus)
    
    logger.info("Loading reranker...")
    reranker = Reranker()
    
    results = []
    
    # --------------------------------------------------------
    # 3. Evaluate TinyLlama
    # --------------------------------------------------------
    cleanup_memory()
    logger.info("\n" + "="*50)
    logger.info("MODEL 1: TinyLlama-1.1B-Chat")
    logger.info("="*50)
    
    generator_tiny = RAGGenerator(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    metrics_tiny = evaluate_model(
        generator_fn=lambda q, ctx: generator_tiny.generate_answer(q, ctx),
        eval_data=eval_data,
        retriever=sparse,
        reranker=reranker,
        model_name="TinyLlama"
    )
    
    results.append({
        "name": "TinyLlama-1.1B-Chat",
        "params": "1.1B",
        "em": round(metrics_tiny["exact_match"], 2),
        "f1": round(metrics_tiny["f1"], 2),
        "winner": False
    })
    
    logger.info(f"TinyLlama: EM={metrics_tiny['exact_match']:.2f}%, F1={metrics_tiny['f1']:.2f}%")
    
    del generator_tiny
    cleanup_memory()
    
    # --------------------------------------------------------
    # 4. Evaluate Qwen2
    # --------------------------------------------------------
    logger.info("\n" + "="*50)
    logger.info("MODEL 2: Qwen2-1.5B-Instruct")
    logger.info("="*50)
    
    generator_qwen = RAGGenerator(model_name="Qwen/Qwen2-1.5B-Instruct")
    metrics_qwen = evaluate_model(
        generator_fn=lambda q, ctx: generator_qwen.generate_answer(q, ctx),
        eval_data=eval_data,
        retriever=sparse,
        reranker=reranker,
        model_name="Qwen2"
    )
    
    results.append({
        "name": "Qwen2-1.5B-Instruct",
        "params": "1.5B",
        "em": round(metrics_qwen["exact_match"], 2),
        "f1": round(metrics_qwen["f1"], 2),
        "winner": False
    })
    
    logger.info(f"Qwen2: EM={metrics_qwen['exact_match']:.2f}%, F1={metrics_qwen['f1']:.2f}%")
    
    del generator_qwen
    cleanup_memory()
    
    # --------------------------------------------------------
    # 5. Determine winner & save
    # --------------------------------------------------------
    best_idx = max(range(len(results)), key=lambda i: results[i]["em"])
    results[best_idx]["winner"] = True
    
    output = {
        "experiment": "TinyLlama vs Qwen2 Comparison (Colab)",
        "retriever": "BM25 (10k docs)",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "eval_samples": EVAL_SIZE,
        "device": device,
        "gpu": gpu_name,
        "timestamp": datetime.now().isoformat(),
        "models": results,
        "conclusion": f"{results[best_idx]['name']} achieves the best performance."
    }
    
    output_file = "llm_comparison_colab.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # --------------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\n{'Model':<25} {'Params':<10} {'EM':<12} {'F1':<12} {'Winner'}")
    print("-"*60)
    for m in results:
        winner = "🏆" if m["winner"] else ""
        print(f"{m['name']:<25} {m['params']:<10} {m['em']:<12.2f} {m['f1']:<12.2f} {winner}")
    print("-"*60)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

