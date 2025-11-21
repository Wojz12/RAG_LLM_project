import argparse
import json
import logging
import string
import re
from collections import Counter
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Computes the maximum score over a list of ground truths.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_predictions(predictions: Dict[str, str], references: Dict[str, List[str]]):
    """
    Computes EM and F1 scores.
    
    Args:
        predictions: Dict mapping question_id to predicted answer string.
        references: Dict mapping question_id to list of valid answer strings.
    """
    f1 = 0
    exact_match = 0
    total = 0
    
    for q_id, prediction in predictions.items():
        if q_id not in references:
            logger.warning(f"Question ID {q_id} found in predictions but not in references.")
            continue
            
        ground_truths = references[q_id]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        total += 1
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return {'exact_match': exact_match, 'f1': f1, 'total': total}

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--references", type=str, help="Path to references JSON (optional if embedded in preds)")
    
    args = parser.parse_args()
    
    logger.info(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, "r") as f:
        data = json.load(f)
    
    # Handle two formats:
    # 1. Simple dict: {id: prediction} (Need external refs)
    # 2. List of dicts: [{id: ..., prediction: ..., answers: [...]}]
    
    predictions = {}
    references = {}
    
    if isinstance(data, list):
        for item in data:
            q_id = item.get("id") or item.get("question_id")
            predictions[q_id] = item.get("prediction", "")
            # TriviaQA provides 'answer' dict with 'aliases' and 'normalized_aliases'
            # We usually use aliases list for evaluation
            if "answers" in item:
                 references[q_id] = item["answers"]
    elif isinstance(data, dict):
         # Assume simple format, references must be provided separately or embedded differently
         predictions = data
         if args.references:
             with open(args.references, "r") as f:
                 ref_data = json.load(f)
                 # Load refs logic here...
                 pass
         else:
             logger.error("If predictions are a simple dict, --references must be provided.")
             return

    logger.info(f"Evaluating {len(predictions)} examples...")
    metrics = evaluate_predictions(predictions, references)
    
    print("\nResults:")
    print(f"Exact Match: {metrics['exact_match']:.2f}")
    print(f"F1 Score:    {metrics['f1']:.2f}")
    print(f"Total:       {metrics['total']}")

if __name__ == "__main__":
    main()

