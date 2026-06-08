import argparse
import sys
from typing import Dict, Any


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the SentiScope project.
    Validates required constraints and formats.
    """
    parser = argparse.ArgumentParser(
        description="SentiScope - Modular Sentiment Analysis System using PyTorch and BERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Mode Selection ---
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Pipeline execution mode: 'train' to fine-tune a model or 'predict' to run inference."
    )

    # --- Data Path Arguments ---
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the dataset CSV file (Required for training or batch prediction)."
    )

    # --- Model Saving & Loading Paths ---
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/sentiscope_bert",
        help="Directory path to save the trained model or load the model for prediction."
    )

    # --- Predictor Specific ---
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Single review text string to classify in 'predict' mode."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/predictions.csv",
        help="Output CSV path for batch predictions."
    )

    # --- Trainer Specific ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train the model."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience epochs for early stopping before training terminates."
    )

    args = parser.parse_args()

    # --- Custom Logical Constraint Validations ---
    if args.mode == "train":
        if args.data_path is None:
            parser.error("--data_path is required when --mode is 'train'.")
        if args.epochs <= 0:
            parser.error("--epochs must be a positive integer greater than 0.")
        if args.batch_size <= 0:
            parser.error("--batch_size must be a positive integer greater than 0.")
        if args.patience <= 0:
            parser.error("--patience must be a positive integer greater than 0.")

    elif args.mode == "predict":
        # Predict mode requires either a single text or a CSV path
        if args.input_text is None and args.data_path is None:
            parser.error("Prediction requires either --input_text for single prediction or --data_path for batch prediction.")
        if args.input_text is not None and args.data_path is not None:
            parser.error("Specify --input_text OR --data_path for prediction, not both.")

    return args
