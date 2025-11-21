import logging
from typing import Dict, Any
from datasets import load_dataset, Dataset, DatasetDict

logger = logging.getLogger(__name__)

def load_trivia_qa(split_name: str = "rc.wikipedia") -> DatasetDict:
    """
    Loads the TriviaQA dataset and applies the custom split rules for the project.

    Rules:
        1. Validation Set: First 7,900 examples of the original 'train' set.
        2. Training Set: The remaining examples of the original 'train' set.
        3. Test Set: The original 'validation' set.

    Args:
        split_name (str): The configuration name for TriviaQA (default: "rc.wikipedia").

    Returns:
        DatasetDict: A dictionary containing 'train', 'validation', and 'test' splits.
    """
    logger.info(f"Loading TriviaQA dataset config: {split_name}...")
    
    # Load the full dataset
    # 'verification_mode' is set to 'no_checks' to avoid potential checksum issues with some HF datasets,
    # though usually not strictly necessary for TriviaQA, it's safer for automation.
    raw_dataset = load_dataset("trivia_qa", split_name)
    
    original_train = raw_dataset["train"]
    original_validation = raw_dataset["validation"]
    
    # Enforce the 7,900 split rule
    split_index = 7900
    
    logger.info(f"Splitting original train set at index {split_index}...")
    
    # Create new validation set (first 7,900 of train)
    new_validation = original_train.select(range(split_index))
    
    # Create new train set (rest of train)
    new_train = original_train.select(range(split_index, len(original_train)))
    
    # The original validation becomes the test set
    new_test = original_validation
    
    final_dataset = DatasetDict({
        "train": new_train,
        "validation": new_validation,
        "test": new_test
    })
    
    logger.info(f"Data splitting complete.")
    logger.info(f"Train size: {len(final_dataset['train'])}")
    logger.info(f"Validation size: {len(final_dataset['validation'])}")
    logger.info(f"Test size: {len(final_dataset['test'])}")
    
    return final_dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_trivia_qa()
    print("Sample Train Example:", data["train"][0]["question"])

