import os
import shutil
import logging
from typing import Tuple
from transformers import BertForSequenceClassification, BertTokenizer
from utils import MODEL_NAME

logger = logging.getLogger("SentiScope")


class ModelManager:
    """
    Handles downloading, loading, saving, and managing BERT models
    for sequence classification.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def load_pretrained_model(self, num_labels: int = 3) -> BertForSequenceClassification:
        """
        Loads the pre-trained BERT classification model from Hugging Face.
        """
        try:
            logger.info(f"Loading pre-trained BERT sequence classifier '{self.model_name}' for {num_labels} classes...")
            model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load pre-trained model '{self.model_name}': {str(e)}")
            raise RuntimeError(
                f"Could not load pre-trained BERT model. Please check your internet connection "
                f"or model name. Details: {str(e)}"
            )

    def save_model(
        self, 
        model: BertForSequenceClassification, 
        tokenizer_wrapper, 
        save_path: str
    ) -> None:
        """
        Saves the fine-tuned BERT model and its tokenizer to a target directory.
        Creates directories if they do not exist.
        """
        try:
            logger.info(f"Saving fine-tuned SentiScope model to '{save_path}'...")
            
            # Create directory tree if missing
            os.makedirs(save_path, exist_ok=True)

            # Save the PyTorch model config and weights using HuggingFace save_pretrained
            model.save_pretrained(save_path)
            
            # Save the tokenizer files
            tokenizer_wrapper.save_pretrained(save_path)
            
            logger.info(f"Model and tokenizer successfully saved to '{save_path}'")
        except Exception as e:
            logger.error(f"Failed to save model to path '{save_path}': {str(e)}")
            raise IOError(f"Could not save fine-tuned model checkpoint: {str(e)}")

    def load_fine_tuned_model(
        self, 
        model_path: str
    ) -> Tuple[BertForSequenceClassification, BertTokenizer]:
        """
        Loads a fine-tuned BERT classification model and its corresponding tokenizer 
        from a local checkpoint directory.
        Handles missing files and directories gracefully.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory '{model_path}' does not exist. "
                "Ensure you have trained the model first or provided a valid path."
            )

        # Check for typical files indicating a saved model
        config_file = os.path.join(model_path, "config.json")
        weights_file = os.path.join(model_path, "model.safetensors")
        weights_bin_file = os.path.join(model_path, "pytorch_model.bin")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Invalid model directory: '{config_file}' is missing."
            )
            
        if not os.path.exists(weights_file) and not os.path.exists(weights_bin_file):
            raise FileNotFoundError(
                f"Invalid model directory: Model weights (pytorch_model.bin or model.safetensors) "
                f"were not found in '{model_path}'."
            )

        try:
            logger.info(f"Loading SentiScope fine-tuned model from local path '{model_path}'...")
            model = BertForSequenceClassification.from_pretrained(model_path)
            
            logger.info(f"Loading SentiScope tokenizer from local path '{model_path}'...")
            tokenizer = BertTokenizer.from_pretrained(model_path)
            
            logger.info("Local SentiScope checkpoint loaded successfully.")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading SentiScope model/tokenizer from '{model_path}': {str(e)}")
            raise RuntimeError(
                f"Failed to parse and load the local model checkpoint from '{model_path}': {str(e)}"
            )
