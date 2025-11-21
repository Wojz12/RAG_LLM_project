import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """
    Splits text into overlapping chunks of words (simple whitespace tokenization).
    
    Args:
        text (str): The full document text.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between chunks.
        
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
        
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Move start forward, but respect overlap
        # If we are at the end, break
        if end >= len(words):
            break
            
        start += (chunk_size - overlap)
        
    return chunks

def prepare_corpus(dataset, doc_field: str = "entity_pages") -> List[Dict[str, Any]]:
    """
    Extracts and chunks documents from the TriviaQA dataset to form the retrieval corpus.
    
    TriviaQA 'rc' structure for entity pages:
    example['entity_pages'] is a list of dicts: [{'wiki_context': '...', 'title': '...'}]
    
    Args:
        dataset: The HuggingFace dataset (usually the train split).
        doc_field: Field name containing documents.
        
    Returns:
        List[Dict[str, Any]]: Flat list of chunks. 
        Format: [{"id": "doc_id_chunk_id", "text": "...", "title": "...", "source_id": "..."}]
    """
    logger.info("Processing dataset to create retrieval corpus (chunking)...")
    
    corpus = []
    
    # We iterate through the dataset. 
    # NOTE: TriviaQA RC examples can share documents, but for simplicity in this baseline,
    # we process each example's context. 
    # A more advanced version might deduplicate documents based on URL/Title.
    
    doc_id_counter = 0
    
    for example in tqdm(dataset, desc="Chunking documents"):
        # TriviaQA has 'entity_pages' which contains the Wikipedia content
        pages = example.get(doc_field, [])
        
        for page in pages:
            title = page.get("title", "")
            # wiki_context contains the text
            text = page.get("wiki_context", "")
            
            if not text:
                continue
            
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                corpus.append({
                    "id": f"{doc_id_counter}_{i}",
                    "text": chunk,
                    "title": title,
                    "source_id": example["question_id"] # Tracking where this came from
                })
            
            doc_id_counter += 1
            
    logger.info(f"Created corpus with {len(corpus)} chunks from {doc_id_counter} original docs.")
    return corpus

