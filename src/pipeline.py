import argparse
import logging
import os
import json
from tqdm import tqdm
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
    parser.add_argument("--predict_split", type=str, choices=["train", "validation", "test"], help="Run predictions on a specific split")
    
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
            
        # 1. Single Query Mode
        if args.query:
            logger.info(f"Query: {args.query}")
            results = retriever.retrieve(args.query, top_k=3)
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
            
            predictions = []
            
            for example in tqdm(dataset, desc="Predicting"):
                question = example["question"]
                q_id = example["question_id"]
                
                # Retrieve top document
                results = retriever.retrieve(question, top_k=1)
                
                if results:
                    # In Phase 1 (Sparse Only), the "prediction" is the retrieved text 
                    # (or we could say it's just a retrieval task, but the eval script expects an answer).
                    # Since we don't have a generator yet, we can't generate an answer.
                    # BUT, for retrieval evaluation (Recall@K), we usually check if answer is in the retrieved text.
                    # However, our evaluation.py calculates EM/F1 against the text. 
                    # If we just return the passage, EM will be near 0.
                    # This implies Phase 1 might be about evaluating "Retrieval Recall" or 
                    # we are expected to output the passage as the 'answer' for now?
                    # The README says: "Evaluates using Exact Match (EM) and F1".
                    # If we return a 256-token passage, F1 might be okay, but EM will be 0.
                    
                    # Let's assume for now the prediction is the top retrieved passage text.
                    predicted_answer = results[0]["text"]
                else:
                    predicted_answer = ""
                
                # We save in the format expected by evaluation.py
                predictions.append({
                    "id": q_id,
                    "prediction": predicted_answer,
                    "answers": example["answer"]["aliases"]
                })
            
            output_file = os.path.join(args.output_dir, "preds.json")
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to {output_file}")

        if not args.query and not args.predict_split:
            logger.info("No query or predict_split provided. Use --query or --predict_split.")
            
if __name__ == "__main__":
    main()
