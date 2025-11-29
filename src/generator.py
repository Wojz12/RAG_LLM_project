"""RAG Generator module with multi-model support for better QA performance."""

import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

# Unified grounding constant - use this exact phrase for "I don't know" responses
GROUNDING_FALLBACK = "I don't know from the given documents."

# Supported models with their configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "use_4bit": False,  # Small enough without quantization
        "max_context": 1500,
        "template": "simple",
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "use_4bit": True,  # Required for Colab T4
        "max_context": 2000,
        "template": "mistral",
    },
    "phi3": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "use_4bit": True,  # Recommended for Colab
        "max_context": 2500,
        "template": "phi3",
    },
}


class RAGGenerator:
    """
    RAG Generator supporting multiple LLMs with model-specific prompt templates.
    Optimized for Colab T4 GPU with 4-bit quantization for larger models.
    
    Supported models:
        - tinyllama: TinyLlama-1.1B (baseline, fast)
        - mistral: Mistral-7B-Instruct-v0.3 (best quality)
        - phi3: Phi-3-mini-4k-instruct (good balance)
    """
    
    def __init__(
        self, 
        model_name: str = "tinyllama",
        custom_model_id: Optional[str] = None,
        force_4bit: Optional[bool] = None
    ):
        """
        Initialize the generator with a model.
        
        Args:
            model_name: Short name from MODEL_CONFIGS ('tinyllama', 'mistral', 'phi3')
                       or full HuggingFace model ID if custom_model_id is not set.
            custom_model_id: Override model ID (useful for testing variants).
            force_4bit: Override 4-bit quantization setting.
        """
        # Resolve model configuration
        if model_name in MODEL_CONFIGS:
            self.config = MODEL_CONFIGS[model_name].copy()
        else:
            # Treat model_name as a full HuggingFace ID
            self.config = {
                "model_id": model_name,
                "use_4bit": True,
                "max_context": 1500,
                "template": "simple",
            }
        
        if custom_model_id:
            self.config["model_id"] = custom_model_id
        if force_4bit is not None:
            self.config["use_4bit"] = force_4bit
            
        self.model_id = self.config["model_id"]
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the model with appropriate quantization settings."""
        logger.info(f"Loading LLM: {self.model_id} on {self.device}...")
        logger.info(f"4-bit quantization: {self.config['use_4bit']}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True  # Required for Phi-3
            )
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for larger models
            if self.config["use_4bit"] and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Standard loading (fp16 on GPU, fp32 on CPU)
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                    
            logger.info(f"LLM loaded successfully: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

    def _build_prompt(self, query: str, context: str) -> str:
        """Build model-specific prompt."""
        template = self.config["template"]
        max_ctx = self.config["max_context"]
        
        # Truncate context
        context = context[:max_ctx] if len(context) > max_ctx else context
        
        # Base instruction for all models
        instruction = (
            "You are a precise QA assistant. Answer the question using ONLY the provided context. "
            "Give a short, factual answer (1-5 words). If the answer is not in the context, say 'I don't know'."
        )
        
        qa_content = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        if template == "mistral":
            # Mistral [INST]...[/INST] format
            return f"<s>[INST] {instruction}\n\n{qa_content} [/INST]"
        
        elif template == "phi3":
            # Phi-3 format
            return f"<|user|>\n{instruction}\n\n{qa_content}<|end|>\n<|assistant|>"
        
        else:
            # Simple/TinyLlama format - few-shot style
            return f"""{instruction}

Context: {context}

Q: Who wrote Romeo and Juliet?
A: William Shakespeare

Q: What is the capital of Japan?
A: Tokyo

Q: {query}
A:"""

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 30) -> str:
        """Generate an answer given the query and retrieved context.
        
        Args:
            query: The user's question.
            context: Retrieved and reranked context passages.
            max_new_tokens: Maximum tokens to generate.
            
        Returns:
            Generated answer string.
        """
        prompt = self._build_prompt(query, context)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=3500
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated part
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up the answer
        answer = self._clean_answer(answer)
        
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize the generated answer."""
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "A:", "Answer:", "The answer is:", "The answer is", 
            "It is", "It's", "The ", "I think it's", "I believe it's",
            "Based on the context,", "According to the context,"
        ]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Take only first line, stop at newline or next Q:
        answer = answer.split('\n')[0].split('Q:')[0].strip()
        
        # Remove trailing incomplete sentences
        if '.' in answer:
            parts = answer.split('.')
            if len(parts[-1].split()) < 3:  # Last part is incomplete
                answer = '.'.join(parts[:-1]).strip()
            else:
                answer = parts[0].strip()
        
        # Handle comma-separated lists (take first item if it's substantial)
        if ',' in answer and len(answer.split(',')[0]) > 3:
            first_part = answer.split(',')[0].strip()
            if len(first_part) > 2:
                answer = first_part
        
        return answer


# Quick test function
def test_generator(model_name: str = "tinyllama"):
    """Quick test of the generator."""
    print(f"\n{'='*50}")
    print(f"Testing RAGGenerator with model: {model_name}")
    print(f"{'='*50}\n")
    
    gen = RAGGenerator(model_name=model_name)
    
    context = """
    William Shakespeare was an English playwright and poet who lived from 1564 to 1616.
    He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth.
    Shakespeare is often called the greatest writer in the English language.
    """
    
    query = "Who wrote Romeo and Juliet?"
    answer = gen.generate_answer(query, context)
    
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"\nExpected: William Shakespeare")
    
    return gen


if __name__ == "__main__":
    test_generator("tinyllama")
