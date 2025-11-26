"""Build BM25 index using Pyserini/Lucene from processed chunks."""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Union
from pyserini.index.lucene import LuceneIndexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_bm25_index(
    docs: List[Union[str, Dict[str, Any]]], 
    index_path: str = "bm25_index"
) -> None:
    """Build a Lucene/BM25 index from documents.
    
    Args:
        docs: List of documents. Can be either:
              - List[str]: Raw text strings
              - List[Dict]: Dicts with 'text' field (from prepare_corpus)
        index_path: Directory path to save the Lucene index.
    """
    os.makedirs(index_path, exist_ok=True)
    indexer = LuceneIndexer(index_path)
    
    logger.info(f"Building BM25 index with {len(docs)} documents...")
    
    for i, doc in enumerate(docs):
        # Handle both formats: raw string or dict with 'text' field
        if isinstance(doc, dict):
            text = doc.get("text", "")
            doc_id = doc.get("id", str(i))
        else:
            text = doc
            doc_id = str(i)
        
        if text:
            indexer.add_doc(doc_id, text)
    
    indexer.commit()
    logger.info(f"BM25 index saved to {index_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 index from processed chunks")
    parser.add_argument("--chunks_path", type=str, default="processed_chunks.json",
                        help="Path to processed_chunks.json")
    parser.add_argument("--index_path", type=str, default="bm25_index",
                        help="Output directory for Lucene index")
    args = parser.parse_args()
    
    if not os.path.exists(args.chunks_path):
        logger.error(f"Chunks file not found: {args.chunks_path}")
        logger.info("Run 'python src/main.py --force_rebuild' first to generate chunks.")
        exit(1)
    
    with open(args.chunks_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    build_bm25_index(docs, args.index_path)

