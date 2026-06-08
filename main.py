import sys
import os
import logging
import traceback
import torch
from torch.utils.data import DataLoader
from cli import parse_arguments
from utils import setup_logger, set_seed, get_device, Colors, color_text
from data_processor import DataProcessor
from tokenizer import SentimentTokenizer
from dataset import SentimentDataset
from model_manager import ModelManager
from trainer import SentimentTrainer
from predictor import SentimentPredictor

# Setup logger globally for main execution
logger = setup_logger()


def train_workflow(args) -> None:
    """
    Executes the SentiScope training pipeline:
    1. Loads dataset
    2. Cleans and preprocesses data
    3. Splits data
    4. Tokenizes inputs
    5. Wraps in PyTorch Datasets & DataLoaders
    6. Initializes BERT model
    7. Runs training & evaluation
    8. Saves the best model
    """
    logger.info("Initializing SentiScope Model Training Workflow...")

    # Set reproducibility seed
    set_seed(42)

    # Initialize components
    processor = DataProcessor()
    tokenizer_wrapper = SentimentTokenizer()
    model_manager = ModelManager()

    # 1. Load Data
    df = processor.load_data(args.data_path)

    # 2. Preprocess & Encode Labels
    df_preprocessed = processor.preprocess(df)

    # 3. Stratified/Fallback Split
    df_train, df_val, df_test = processor.split_data(df_preprocessed)

    # 4. Tokenization
    logger.info("Tokenizing training set split...")
    train_features = tokenizer_wrapper.tokenize(
        df_train["clean_text"].tolist(), 
        df_train["label_id"].tolist()
    )
    
    logger.info("Tokenizing validation set split...")
    val_features = tokenizer_wrapper.tokenize(
        df_val["clean_text"].tolist(), 
        df_val["label_id"].tolist()
    )
    
    logger.info("Tokenizing testing set split...")
    test_features = tokenizer_wrapper.tokenize(
        df_test["clean_text"].tolist(), 
        df_test["label_id"].tolist()
    )

    # 5. Create PyTorch Datasets
    train_dataset = SentimentDataset(train_features)
    val_dataset = SentimentDataset(val_features)
    test_dataset = SentimentDataset(test_features)

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 7. Download Pretrained BERT Model (3 output classes: positive, negative, neutral)
    model = model_manager.load_pretrained_model(num_labels=3)

    # 8. Initialize Trainer and Start Training
    device = get_device()
    trainer = SentimentTrainer(
        model=model,
        device=device,
        patience=args.patience,
        outputs_dir="outputs",
        report_dir="report"
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs
    )

    # 9. Save Best Fine-Tuned Model & Tokenizer
    model_manager.save_model(
        model=trainer.model,
        tokenizer_wrapper=tokenizer_wrapper,
        save_path=args.model_path
    )
    
    logger.info(color_text("SentiScope training workflow completed successfully!", Colors.GREEN + Colors.BOLD))


def predict_workflow(args) -> None:
    """
    Executes the SentiScope prediction pipeline:
    1. Loads local fine-tuned model and tokenizer
    2. Runs single or batch predictions
    """
    logger.info("Initializing SentiScope Inference Workflow...")

    model_manager = ModelManager()

    # 1. Load fine-tuned model & tokenizer
    try:
        model, tokenizer = model_manager.load_fine_tuned_model(args.model_path)
    except FileNotFoundError as fnf:
        raise fnf
    except Exception as e:
        raise RuntimeError(f"Failed to load fine-tuned model checkpoint: {str(e)}")

    # 2. Initialize Predictor
    device = get_device()
    predictor = SentimentPredictor(model=model, tokenizer=tokenizer, device=device)

    # 3. Perform Inference
    if args.input_text is not None:
        # Single review prediction
        label, confidence = predictor.predict_single(args.input_text)
        predictor.display_pretty_prediction(args.input_text, label, confidence)
    elif args.data_path is not None:
        # Batch prediction
        predictor.predict_batch(
            data_path=args.data_path,
            output_path=args.output_path,
            batch_size=args.batch_size
        )
        logger.info(color_text("Batch prediction process completed successfully!", Colors.GREEN + Colors.BOLD))


def main() -> None:
    """
    Entry point for SentiScope pipeline.
    Validates execution mode and routes to train or predict workflows.
    Implements a top-level error handling system to protect CLI users from raw tracebacks.
    """
    try:
        # Parse and validate CLI arguments
        args = parse_arguments()

        if args.mode == "train":
            train_workflow(args)
        elif args.mode == "predict":
            predict_workflow(args)

    except torch.cuda.OutOfMemoryError as oom:
        err_msg = (
            "CUDA OUT OF MEMORY ERROR: SentiScope ran out of VRAM memory during operations.\n"
            "Try reducing the '--batch_size' parameter (e.g. --batch_size 2 or 4) or run on CPU."
        )
        logger.error(err_msg)
        logger.error(traceback.format_exc())
        print(f"\n{color_text(err_msg, Colors.FAIL + Colors.BOLD)}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (Ctrl+C). Exiting.")
        print(f"\n{color_text('Execution cancelled by user.', Colors.WARNING)}", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        # Handle all unexpected exceptions gracefully without raw tracebacks
        logger.error("A critical execution error occurred during pipeline execution:")
        logger.error(traceback.format_exc())

        user_friendly_msg = (
            f"CRITICAL ERROR: {str(e)}\n\n"
            "For full technical details, please check the logs file located at: 'logs/training.log'"
        )
        print(f"\n{color_text(user_friendly_msg, Colors.FAIL + Colors.BOLD)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
