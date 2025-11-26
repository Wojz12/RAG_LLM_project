import argparse
import logging
import os
import json
from tqdm import tqdm
import torch

# Ensure clean imports to avoid TF conflicts
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.data_loader import load_trivia_qa
from src.retriever import SparseRetriever, DenseRetriever
from src.utils import prepare_corpus
from src.generator import RAGGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline (Dense/Sparse) for TriviaQA")
    
    # Arguments
    parser.add_argument("--retriever_type", type=str, choices=["sparse", "dense"], default="dense", help="Type of retriever to use")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--index_path", type=str, default="rag_index", help="Base path for index (will add extensions)")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuilding the index")
    parser.add_argument("--sample_size", type=int, help="Number of examples to use for indexing")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="LLM model name")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Embedding model for Dense Retriever")
    
    args = parser.parse_args()
    
    # 1. Initialize Retriever
    if args.retriever_type == "dense":
        logger.info(f"Initializing Dense Retriever with {args.embedding_model}...")
        retriever = DenseRetriever(model_name=args.embedding_model)
        # Dense retriever saves to path.faiss and path.meta.pkl
        # We use args.index_path as the prefix
    else:
        logger.info("Initializing Sparse Retriever (BM25)...")
        retriever = SparseRetriever()
        # Sparse uses .pkl extension usually, we'll handle it inside retriever.save_index if needed
        # or just pass the raw path if the class handles it. 
        # My SparseRetriever expects a full filename for pickle.
        if not args.index_path.endswith(".pkl"):
             args.index_path += ".pkl"

    # 2. Load or Build Index
    index_exists = False
    if args.retriever_type == "dense":
        if os.path.exists(args.index_path + ".faiss"):
            index_exists = True
    else:
        if os.path.exists(args.index_path):
            index_exists = True
            
    if index_exists and not args.force_rebuild:
        try:
            retriever.load_index(args.index_path)
        except Exception as e:
            logger.error(f"Failed to load index: {e}. Rebuilding...")
            args.force_rebuild = True
            
    if args.force_rebuild or not index_exists:
        logger.info("Building index...")
        data = load_trivia_qa()
        train_data = data["train"]
        
        if args.sample_size:
            logger.warning(f"Using sample of {args.sample_size} docs...")
            train_data = train_data.select(range(args.sample_size))
            
        corpus = prepare_corpus(train_data)
        
        # Save chunks for BM25
        import json
        with open("processed_chunks.json", "w") as f:
            json.dump(corpus, f)
        
        retriever.build_index(corpus)
        retriever.save_index(args.index_path)

    # 3. Initialize Generator
    logger.info(f"Initializing Generator ({args.model_name})...")
    generator = RAGGenerator(model_name=args.model_name)

    # 4. Execute RAG (Single Query)
    if args.query:
        logger.info(f"Query: {args.query}")
        
        # Retrieve
        results = retriever.retrieve(args.query, top_k=3)
        
        print("\nRetrieved Contexts:")
        context_pieces = []
        for i, doc in enumerate(results):
            print(f"[{i+1}] {doc['title']} (Score: {doc['score']:.4f})")
            # print(f"    {doc['text'][:100]}...")
            context_pieces.append(f"Title: {doc['title']}\nText: {doc['text']}")
            
        full_context = "\n\n".join(context_pieces)
        
        # Generate
        print("\nGenerating Answer...")
        answer = generator.generate_answer(args.query, full_context)
        
        print("-" * 50)
        print(f"Question: {args.query}")
        print(f"Answer:   {answer}")
        print("-" * 50)
    else:
        logger.info("No query provided. Use --query 'Your question' to test.")

if __name__ == "__main__":
    main()

