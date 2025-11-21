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
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model name for generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Retriever (Common to both modes)
    retriever = SparseRetriever()
    
    # Logic for Loading/Building Index
    if os.path.exists(args.index_path) and not args.force_rebuild:
        try:
            retriever.load_index(args.index_path)
        except Exception as e:
            logger.error(f"Failed to load index: {e}. Rebuilding...")
            args.force_rebuild = True
            
    if args.force_rebuild or not os.path.exists(args.index_path):
        logger.info("Index not found or rebuild forced. Loading dataset...")
        data = load_trivia_qa()
        train_data = data["train"]
        if args.sample_size:
            logger.warning(f"Using a sample of {args.sample_size} examples for debugging.")
            train_data = train_data.select(range(args.sample_size))
        
        corpus = prepare_corpus(train_data)
        retriever.build_index(corpus)
        retriever.save_index(args.index_path)

    if args.mode == "sparse":
        logger.info("Running Sparse Retrieval Baseline...")
        
        # 1. Single Query Mode
        if args.query:
            logger.info(f"Query: {args.query}")
            results = retriever.retrieve(args.query, top_k=5)
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
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to {output_file}")

    elif args.mode == "generate":
        logger.info("Running RAG Generation Mode...")
        from src.generator import RAGGenerator
        
        # Initialize Generator
        generator = RAGGenerator(model_name=args.model_name)
        
        # 1. Single Query Mode
        if args.query:
            logger.info(f"Query: {args.query}")
            # Retrieve
            # Increase top_k to 5 to get more context potential
            retrieved_docs = retriever.retrieve(args.query, top_k=5)
            
            # Combine context
            # Adding Title helps the model know what the text is about
            context_pieces = [f"Document: {doc['title']}\nContent: {doc['text']}" for doc in retrieved_docs]
            context = "\n\n".join(context_pieces)
            
            print(f"\nContext (Top-5):\n{context[:500]}...\n")
            
            # Generate
            answer = generator.generate_answer(args.query, context)
            print(f"\nGenerated Answer: {answer}")
            
        # 2. Batch Prediction Mode
        if args.predict_split:
            logger.info(f"Running batch generation on {args.predict_split} split...")
            data = load_trivia_qa()
            dataset = data[args.predict_split]
            
            if args.sample_size:
                 logger.warning(f"Predicting on a sample of {args.sample_size} examples.")
                 dataset = dataset.select(range(args.sample_size))
            
            predictions = []
            
            for example in tqdm(dataset, desc="Generating"):
                question = example["question"]
                q_id = example["question_id"]
                
                # Retrieve
                results = retriever.retrieve(question, top_k=5)
                context_pieces = [f"Document: {doc['title']}\nContent: {doc['text']}" for doc in results]
                context = "\n\n".join(context_pieces)
                
                # Generate
                predicted_answer = generator.generate_answer(question, context)
                
                predictions.append({
                    "id": q_id,
                    "prediction": predicted_answer,
                    "answers": example["answer"]["aliases"]
                })
            
            output_file = os.path.join(args.output_dir, "preds_rag.json")
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
