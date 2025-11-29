"""Hybrid RAG Pipeline: Dense + BM25 + Reranking + LLM Generation."""

import argparse
import logging
import os
import json
import pickle
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from rank_bm25 import BM25Okapi

from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever, DenseRetriever
from src.utils import prepare_corpus
from src.re_ranker import Reranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for grounding prompt
GROUNDING_FALLBACK = "I don't know from the given documents."

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG Pipeline for TriviaQA")
    parser.add_argument("--mode", type=str, choices=["sparse", "generate"], default="sparse", 
                        help="Mode: 'sparse' for BM25-only, 'generate' for full Hybrid RAG")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--sparse_index_path", type=str, default="bm25_index.pkl", 
                        help="Path to rank_bm25 sparse index (.pkl)")
    parser.add_argument("--dense_index_path", type=str, default="rag_index", 
                        help="Path prefix for FAISS dense index (.faiss/.meta.pkl)")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuilding the index")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save predictions")
    parser.add_argument("--sample_size", type=int, help="Number of examples for debugging")
    parser.add_argument("--predict_split", type=str, choices=["train", "validation", "test"], 
                        help="Run predictions on a specific split")
    parser.add_argument("--llm", type=str, default="tinyllama", 
                        choices=["tinyllama", "mistral", "phi3"],
                        help="LLM to use: tinyllama (1.1B baseline), mistral (7B best), phi3 (3.8B balanced)")
    parser.add_argument("--model_name", type=str, default=None, 
                        help="Custom HuggingFace model ID (overrides --llm)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Sparse Retriever (rank_bm25 - for sparse mode)
    sparse_retriever = SparseRetriever()
    reranker = Reranker()
    
    # Dense retriever and BM25 are initialized lazily for generate mode
    dense_retriever: Optional[DenseRetriever] = None
    bm25: Optional[BM25Okapi] = None
    bm25_corpus: List[Dict[str, Any]] = []
    
    # Logic for Loading/Building Sparse Index (rank_bm25)
    if os.path.exists(args.sparse_index_path) and not args.force_rebuild:
        try:
            sparse_retriever.load_index(args.sparse_index_path)
        except Exception as e:
            logger.error(f"Failed to load sparse index: {e}. Rebuilding...")
            args.force_rebuild = True
            
    if args.force_rebuild or not os.path.exists(args.sparse_index_path):
        logger.info("Sparse index not found or rebuild forced. Loading dataset...")
        data = load_trivia_qa()
        train_data = data["train"]
        if args.sample_size:
            logger.warning(f"Using a sample of {args.sample_size} examples for debugging.")
            train_data = train_data.select(range(args.sample_size))
        
        corpus = prepare_corpus(train_data)
        sparse_retriever.build_index(corpus)
        sparse_retriever.save_index(args.sparse_index_path)

    if args.mode == "sparse":
        logger.info("Running Sparse Retrieval Baseline (rank_bm25)...")
        
        # 1. Single Query Mode
        if args.query:
            logger.info(f"Query: {args.query}")
            results = sparse_retriever.retrieve(args.query, top_k=5)
            for i, res in enumerate(results):
                print(f"\nRank {i+1} (Score: {res['score']:.4f}):")
                print(f"Title: {res['title']}")
                print(f"Text: {res['text'][:200]}...")
                
        # 2. Batch Prediction Mode (for Evaluation)
        if args.predict_split:
            logger.info(f"Running batch predictions on {args.predict_split} split...")
            data = load_trivia_qa()
            dataset = data[args.predict_split]
            
            if args.sample_size:
                logger.warning(f"Predicting on a sample of {args.sample_size} examples.")
                dataset = dataset.select(range(args.sample_size))
            
            predictions: List[Dict[str, Any]] = []
            
            for example in tqdm(dataset, desc="Predicting"):
                question = example["question"]
                q_id = example["question_id"]
                
                # Retrieve top document
                results = sparse_retriever.retrieve(question, top_k=1)
                
                if results:
                    # In Phase 1 (Sparse Only), prediction is the retrieved text
                    predicted_answer = results[0]["text"]
                else:
                    predicted_answer = ""
                
                predictions.append({
                    "id": q_id,
                    "prediction": predicted_answer,
                    "answers": example["answer"]["aliases"]
                })
            
            output_file = os.path.join(args.output_dir, "preds_sparse.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to {output_file}")

    elif args.mode == "generate":
        logger.info("Running Hybrid RAG Generation Mode (Dense + BM25 + Reranking)...")
        from src.generator import RAGGenerator
        
        # Initialize Dense Retriever
        dense_retriever = DenseRetriever()
        dense_index_file = args.dense_index_path + ".faiss"
        if os.path.exists(dense_index_file):
            dense_retriever.load_index(args.dense_index_path)
        else:
            logger.error(f"Dense index not found: {dense_index_file}")
            logger.info("Run 'python src/main.py --force_rebuild' first to build the FAISS index.")
            return
        
        # Load BM25 index (rank_bm25)
        print("Loading BM25 index...")
        try:
            with open(args.sparse_index_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            bm25 = bm25_data["bm25"]
            bm25_corpus = bm25_data["corpus"]
            
            print(f"BM25 index loaded with {len(bm25_corpus)} documents.")
        except FileNotFoundError:
            logger.warning(f"BM25 index not found: {args.sparse_index_path}")
            logger.warning("Hybrid retrieval will use only Dense retriever.")
            bm25 = None
            bm25_corpus = []
        
        # Initialize Generator with selected LLM
        if args.model_name:
            # Custom model ID provided
            generator = RAGGenerator(model_name=args.model_name)
        else:
            # Use preset model (tinyllama, mistral, phi3)
            generator = RAGGenerator(model_name=args.llm)
        logger.info(f"Using LLM: {generator.model_id}")
        
        # 1. Single Query Mode
        if args.query:
            logger.info(f"Query: {args.query}")
            # Hybrid Retrieval: Dense + BM25 + Reranking
            dense_docs = dense_retriever.retrieve(args.query, top_k=20)
            dense_texts = [doc['text'] for doc in dense_docs]
            
            # BM25 retrieval (rank_bm25)
            bm25_top_texts: List[str] = []
            if bm25 is not None:
                tokenized_query = args.query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                top_bm25_ids = bm25_scores.argsort()[-20:][::-1]
                bm25_top_texts = [bm25_corpus[i]["text"] for i in top_bm25_ids if i < len(bm25_corpus)]
            
            # Combine and deduplicate
            combined = list(set(dense_texts + bm25_top_texts))
            
            # Rerank
            contexts = reranker.rerank(args.query, combined, top_k=5)
            context_str = "\n\n".join(contexts)
            
            print(f"\nContext (Top-5 reranked):\n{context_str[:500]}...\n")
            
            # Generate with hard-grounding
            answer = generator.generate_answer(args.query, context_str)
            print(f"\nGenerated Answer: {answer}")
            
        # 2. Batch Prediction Mode
        if args.predict_split:
            logger.info(f"Running batch generation on {args.predict_split} split...")
            data = load_trivia_qa()
            dataset = data[args.predict_split]
            
            if args.sample_size:
                logger.warning(f"Predicting on a sample of {args.sample_size} examples.")
                dataset = dataset.select(range(args.sample_size))
            
            predictions: List[Dict[str, Any]] = []
            
            for example in tqdm(dataset, desc="Generating"):
                question = example["question"]
                q_id = example["question_id"]
                
                # Hybrid Retrieval: Dense + BM25 + Reranking
                dense_docs = dense_retriever.retrieve(question, top_k=20)
                dense_texts = [doc['text'] for doc in dense_docs]
                
                # BM25 retrieval (rank_bm25)
                bm25_top_texts_batch: List[str] = []
                if bm25 is not None:
                    tokenized_query = question.lower().split()
                    bm25_scores = bm25.get_scores(tokenized_query)
                    top_bm25_ids = bm25_scores.argsort()[-20:][::-1]
                    bm25_top_texts_batch = [bm25_corpus[i]["text"] for i in top_bm25_ids if i < len(bm25_corpus)]
                
                # Combine and deduplicate
                combined = list(set(dense_texts + bm25_top_texts_batch))
                
                # Rerank
                contexts = reranker.rerank(question, combined, top_k=5)
                context_str = "\n\n".join(contexts)
                
                # Generate with hard-grounding
                predicted_answer = generator.generate_answer(question, context_str)
                
                predictions.append({
                    "id": q_id,
                    "prediction": predicted_answer,
                    "answers": example["answer"]["aliases"]
                })
            
            output_file = os.path.join(args.output_dir, "preds_rag.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
