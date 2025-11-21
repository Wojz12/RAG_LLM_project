import argparse
import logging
import os
import json
from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever
from src.utils import prepare_corpus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for TriviaQA")
    parser.add_argument("--mode", type=str, choices=["sparse", "generate"], default="sparse", help="Mode of operation")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--index_path", type=str, default="bm25_index.pkl", help="Path to save/load BM25 index")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuilding the index even if it exists")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save predictions")
    parser.add_argument("--sample_size", type=int, help="Number of examples to use for testing (debugging)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "sparse":
        logger.info("Running Sparse Retrieval Baseline...")
        
        retriever = SparseRetriever()
        
        # Check if index exists
        if os.path.exists(args.index_path) and not args.force_rebuild:
            retriever.load_index(args.index_path)
        else:
            logger.info("Index not found or rebuild forced. Loading dataset...")
            data = load_trivia_qa()
            train_data = data["train"]
            
            # Apply sampling if requested
            if args.sample_size:
                logger.warning(f"Using a sample of {args.sample_size} examples for debugging.")
                train_data = train_data.select(range(args.sample_size))
            
            # Prepare corpus from training data
            corpus = prepare_corpus(train_data)
            
            # Build index
            retriever.build_index(corpus)
            retriever.save_index(args.index_path)
            
        # Test Query
        if args.query:
            logger.info(f"Query: {args.query}")
            results = retriever.retrieve(args.query, top_k=3)
            for i, res in enumerate(results):
                print(f"\nRank {i+1} (Score: {res['score']:.4f}):")
                print(f"Title: {res['title']}")
                print(f"Text: {res['text'][:200]}...")
        else:
            logger.info("No query provided. Use --query to test a single question.")
            
if __name__ == "__main__":
    main()
