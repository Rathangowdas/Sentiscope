import os
import logging
import torch
import pandas as pd
from typing import List, Tuple, Dict, Union
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from utils import get_device, INV_LABEL_MAP, Colors, color_text
from data_processor import DataProcessor
from tokenizer import SentimentTokenizer

logger = logging.getLogger("SentiScope")


class SentimentPredictor:
    """
    Handles sentiment inference workflows for SentiScope.
    Supports single review classification and high-performance batched CSV inference.
    """

    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        device: torch.device = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else get_device()
        
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Predicts the sentiment and confidence score of a single string.
        """
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Input review text cannot be empty or null.")

        # Clean the text using the standard processor
        cleaned_text = DataProcessor.clean_text(text)
        if cleaned_text == "":
            logger.warning("Text resulted in an empty string after preprocessing cleaning.")
            cleaned_text = text  # fallback to raw text if cleaning wipes it out

        try:
            # Tokenize single text
            inputs = self.tokenizer(
                cleaned_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                
                confidence, predicted_class_id = torch.max(probabilities, dim=-1)
                
                class_id = int(predicted_class_id.item())
                score = float(confidence.item())
                
                predicted_label = INV_LABEL_MAP[class_id]
                return predicted_label, score

        except Exception as e:
            logger.error(f"Inference error on input text: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def display_pretty_prediction(self, text: str, label: str, confidence: float) -> None:
        """
        Prints premium pretty output for single text sentiment classification.
        """
        # Determine colored label
        if label == "positive":
            colored_label = color_text(label.upper(), Colors.GREEN)
        elif label == "negative":
            colored_label = color_text(label.upper(), Colors.FAIL)
        else:
            colored_label = color_text(label.upper(), Colors.WARNING)

        # Print formatted block
        border = "=" * 60
        print(border)
        print(color_text(" SentiScope Sentiment Prediction Result ", Colors.HEADER + Colors.BOLD))
        print(border)
        print(f"Review Text:  {text}")
        print(f"Sentiment:    {colored_label}")
        print(f"Confidence:   {color_text(f'{confidence:.2%}', Colors.BLUE + Colors.BOLD)}")
        print(border)

    def predict_batch(
        self, 
        data_path: str, 
        output_path: str = None, 
        batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Reads a CSV dataset from data_path, processes it in batches, 
        predicts sentiments, and exports results to output_path.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Batch prediction input file not found: '{data_path}'")

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {str(e)}")

        # Validate column requirements
        if "text" not in df.columns:
            raise ValueError(
                f"Batch prediction CSV must contain a 'text' column. Found columns: {list(df.columns)}"
            )

        # Handle null/nan rows
        df["text"] = df["text"].fillna("")
        
        texts = df["text"].tolist()
        predictions = []
        confidences = []

        logger.info(f"Starting batch prediction on {len(texts)} item(s) (batch size: {batch_size})...")

        # Process in batches for optimization
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch Inference"):
            batch_texts = texts[i : i + batch_size]
            
            # Preprocess elements in batch
            cleaned_batch = [DataProcessor.clean_text(t) for t in batch_texts]
            # Replace empty cleaned values with a dummy dot string to avoid tokenizer errors
            cleaned_batch = [t if t != "" else "." for t in cleaned_batch]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    cleaned_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=-1)
                    
                    batch_confidences, batch_class_ids = torch.max(probabilities, dim=-1)
                    
                    for conf, cid in zip(batch_confidences, batch_class_ids):
                        predictions.append(INV_LABEL_MAP[int(cid.item())])
                        confidences.append(float(conf.item()))
            except Exception as e:
                logger.error(f"Failed during batch prediction slice [{i}:{i+batch_size}]: {str(e)}")
                # Fill slice with neutral fallbacks to prevent crash
                predictions.extend(["neutral"] * len(batch_texts))
                confidences.extend([0.33] * len(batch_texts))

        # Augment dataframe
        df["predicted_label"] = predictions
        df["confidence"] = confidences

        # Export if output_path is provided
        if output_path:
            try:
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                df.to_csv(output_path, index=False)
                logger.info(f"Batch prediction results successfully exported to '{output_path}'")
            except Exception as e:
                logger.error(f"Failed to write prediction output CSV: {str(e)}")
                raise IOError(f"Could not save batch predictions: {str(e)}")

        return df
