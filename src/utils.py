import logging
from typing import List, Dict, Any
from tqdm import tqdm

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
        # However, the structure can sometimes be complex or fields might be missing.
        # For 'rc.wikipedia', entity_pages is a Sequence of Structs.
        # In datasets, this usually comes out as a dict of lists:
        # {'doc_source': ['...'], 'filename': ['...'], 'title': ['...'], 'wiki_context': ['...']}
        
        pages_struct = example.get(doc_field, {})
        
        # If pages_struct is empty or None
        if not pages_struct:
            continue

        # Depending on datasets version, it might be a list of dicts OR a dict of lists.
        # The error "AttributeError: 'str' object has no attribute 'get'" suggests we might be iterating over keys of a dict
        # or accessing it wrongly. 
        
        # In standard TriviaQA RC via HF datasets:
        # entity_pages = {'title': ['Title A', 'Title B'], 'wiki_context': ['Text A', 'Text B'], ...}
        
        titles = pages_struct.get("title", [])
        contexts = pages_struct.get("wiki_context", [])
        
        # Ensure we iterate correctly. 
        # If it's a list of dicts (less common in recent HF datasets for this struct), we'd do differently.
        # But the error suggests we treated a struct (dict) as a list of items?
        # Actually, if we did `pages = example.get(doc_field, [])` and pages is a dict, iterating it gives keys (strs).
        # Then `page.get()` fails on the string key.
        
        if isinstance(pages_struct, dict):
             # It is a dict of lists
             count = len(titles)
             for i in range(count):
                 title = titles[i]
                 text = contexts[i]
                 
                 if not text:
                     continue
                
                 chunks = chunk_text(text)
                 for j, chunk in enumerate(chunks):
                    corpus.append({
                        "id": f"{doc_id_counter}_{j}",
                        "text": chunk,
                        "title": title,
                        "source_id": example["question_id"]
                    })
                 doc_id_counter += 1
        
        elif isinstance(pages_struct, list):
            # It is a list of dicts (older HF version or different config)
            for page in pages_struct:
                title = page.get("title", "")
                text = page.get("wiki_context", "")
                
                if not text:
                    continue
                
                chunks = chunk_text(text)
                for j, chunk in enumerate(chunks):
                    corpus.append({
                        "id": f"{doc_id_counter}_{j}",
                        "text": chunk,
                        "title": title,
                        "source_id": example["question_id"]
                    })
                doc_id_counter += 1
            
    logger.info(f"Created corpus with {len(corpus)} chunks from {doc_id_counter} original docs.")
    return corpus
