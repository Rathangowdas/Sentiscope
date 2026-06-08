import logging
import torch
from typing import List, Dict, Union, Optional
from transformers import BertTokenizer
from utils import MODEL_NAME, MAX_LENGTH

logger = logging.getLogger("SentiScope")


class SentimentTokenizer:
    """
    Wrapper around the HuggingFace BERT Tokenizer to provide standardized
    padding, truncation, and return tensors for SentiScope datasets.
    """

    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        self.max_length = max_length
        try:
            logger.info(f"Initializing BERT tokenizer '{model_name}'...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace BERT tokenizer for model '{model_name}': {str(e)}"
            )

    def tokenize(
        self, texts: List[str], labels: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a list of string reviews and returns input_ids, attention_mask,
        and optionally labels as PyTorch tensors.
        """
        if not isinstance(texts, list):
            texts = [texts]

        # Basic error check for empty lists
        if len(texts) == 0:
            raise ValueError("Tokenization failed: Empty text list provided.")

        # Ensure all texts are strings
        cleaned_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning(f"Item at index {i} is not a string ({type(text)}). Casting to string.")
                cleaned_texts.append(str(text))
            else:
                cleaned_texts.append(text)

        try:
            # Tokenize using huggingface tokenizer
            encoded = self.tokenizer(
                cleaned_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Construct output dictionary
            output_dict = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }

            # Map labels if provided
            if labels is not None:
                if len(labels) != len(texts):
                    raise ValueError(
                        f"Mismatch between number of texts ({len(texts)}) and labels ({len(labels)})."
                    )
                output_dict["labels"] = torch.tensor(labels, dtype=torch.long)

            return output_dict

        except Exception as e:
            logger.error(f"Error during tokenization process: {str(e)}")
            raise RuntimeError(f"Tokenization failed: {str(e)}")

    def save_pretrained(self, save_path: str) -> None:
        """
        Saves the tokenizer vocabulary files to a directory so it can be reloaded.
        """
        try:
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Tokenizer saved successfully to {save_path}")
        except Exception as e:
            raise IOError(f"Failed to save tokenizer to '{save_path}': {str(e)}")
            
    @classmethod
    def load_pretrained(cls, load_path: str, max_length: int = MAX_LENGTH) -> "SentimentTokenizer":
        """
        Loads a saved tokenizer from a directory.
        """
        try:
            logger.info(f"Loading custom tokenizer from {load_path}...")
            instance = cls.__new__(cls)
            instance.max_length = max_length
            instance.tokenizer = BertTokenizer.from_pretrained(load_path)
            return instance
        except Exception as e:
            raise IOError(f"Failed to load tokenizer from directory '{load_path}': {str(e)}")
