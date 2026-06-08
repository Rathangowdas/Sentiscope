import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from utils import get_device, INV_LABEL_MAP

logger = logging.getLogger("SentiScope")


class SentimentTrainer:
    """
    Handles fine-tuning of the BERT model, epoch-wise evaluation,
    early stopping, metric logging, and graph plotting.
    """

    def __init__(
        self,
        model: BertForSequenceClassification,
        device: torch.device = None,
        lr: float = 2e-5,
        max_grad_norm: float = 1.0,
        patience: int = 3,
        outputs_dir: str = "outputs",
        report_dir: str = "report"
    ):
        self.model = model
        self.device = device if device is not None else get_device()
        self.model.to(self.device)
        
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        
        self.outputs_dir = outputs_dir
        self.report_dir = report_dir
        
        # Create output and report directories
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 3,
        warmup_ratio: float = 0.1
    ) -> Dict[str, List[float]]:
        """
        Trains the BERT model on train_loader, validates it on val_loader at each epoch,
        saves the best model checkpoint based on validation loss, and evaluates on test_loader
        after training completes.
        """
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(total_steps * warmup_ratio)

        # Setup Optimizer and Scheduler as requested
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        logger.info(f"Total training steps: {total_steps} | Warmup steps: {num_warmup_steps}")

        # Metrics lists for plotting
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        epochs_no_improve = 0
        best_model_weights = None

        # Determine if validation set is too small to provide a reliable early stopping signal
        use_train_loss_checkpoint = len(val_loader.dataset) < 5
        if use_train_loss_checkpoint:
            logger.warning(
                f"Validation dataset size is very small ({len(val_loader.dataset)} samples). "
                "Bypassing validation loss-based early stopping; training loss will be used for model checkpointing."
            )

        for epoch in range(1, epochs + 1):
            # --- TRAINING PHASE ---
            self.model.train()
            train_loss = 0.0
            
            # Progress bar for training
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
            for batch in train_bar:
                optimizer.zero_grad()
                
                # Move tensors to active device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimize
                optimizer.step()
                scheduler.step()

                train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # --- VALIDATION PHASE ---
            val_loss, val_metrics, _, _ = self.evaluate(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_weighted"])

            logger.info(
                f"Epoch {epoch} Results: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}"
            )

            # --- CHECKPOINTING & EARLY STOPPING ---
            if use_train_loss_checkpoint:
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    epochs_no_improve = 0
                    logger.info(f"Training loss improved to {avg_train_loss:.4f}. Recording best model checkpoint.")
                    # Keep copy of best weights in RAM to save at the end of training
                    best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    logger.info(f"Validation loss improved to {val_loss:.4f}. Recording best model checkpoint.")
                    # Keep copy of best weights in RAM to save at the end of training
                    best_model_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    logger.info(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{self.patience}")

            if epochs_no_improve >= self.patience:
                logger.warning(f"Early stopping triggered! Training stopped after epoch {epoch}.")
                break

        # Load best weights back into model before final test evaluation
        if best_model_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_weights.items()})

        # --- FINAL TEST EVALUATION ---
        logger.info("Training completed. Evaluating best model on test set...")
        test_loss, test_metrics, y_true, y_pred = self.evaluate(test_loader, desc="Test Set Evaluation")
        
        logger.info(
            f"Test Set Results: "
            f"Loss: {test_loss:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | "
            f"F1 (Weighted): {test_metrics['f1_weighted']:.4f}"
        )

        # Plot curves and confusion matrix
        self.plot_training_curves(history)
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Save metrics to JSON
        metrics_summary = {
            "training_history": history,
            "best_validation_loss": best_val_loss,
            "test_evaluation": {
                "loss": test_loss,
                "accuracy": test_metrics["accuracy"],
                "precision_weighted": test_metrics["precision_weighted"],
                "recall_weighted": test_metrics["recall_weighted"],
                "f1_weighted": test_metrics["f1_weighted"]
            }
        }
        self.save_metrics_json(metrics_summary)

        # Generate markdown evaluation report
        self.generate_evaluation_report(test_metrics, y_true, y_pred)

        return history

    def evaluate(
        self, data_loader: DataLoader, desc: str = "Evaluating"
    ) -> Tuple[float, Dict[str, float], List[int], List[int]]:
        """
        Evaluates the model on the provided data loader.
        Returns:
            - avg_loss: Average cross entropy loss
            - metrics: Dictionary of classification metrics (Accuracy, Precision, Recall, F1)
            - all_labels: List of actual labels (IDs)
            - all_predictions: List of predicted labels (IDs)
        """
        self.model.eval()
        eval_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            eval_bar = tqdm(data_loader, desc=desc, unit="batch", leave=False)
            for batch in eval_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                eval_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_predictions.extend(preds.cpu().numpy().tolist())

        avg_loss = eval_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        
        # Calculate metrics using scikit-learn
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # We calculate weighted metric averages to accommodate multi-class settings
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1)
        }

        return avg_loss, metrics, all_labels, all_predictions

    def plot_training_curves(self, history: Dict[str, List[float]]) -> None:
        """
        Generates and saves the loss and validation accuracy curves.
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        # Plot 1: Loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
        plt.plot(epochs, history["val_loss"], "r-s", label="Validation Loss")
        plt.title("SentiScope Training & Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(epochs)
        plt.legend()
        loss_path = os.path.join(self.outputs_dir, "training_loss.png")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=150)
        plt.close()
        logger.info(f"Loss curves plot saved to '{loss_path}'")

        # Plot 2: Validation Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["val_accuracy"], "g-^", label="Val Accuracy")
        plt.plot(epochs, history["val_f1"], "m-x", label="Val F1 (Weighted)")
        plt.title("SentiScope Validation Performance")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(epochs)
        plt.legend()
        acc_path = os.path.join(self.outputs_dir, "validation_accuracy.png")
        plt.tight_layout()
        plt.savefig(acc_path, dpi=150)
        plt.close()
        logger.info(f"Accuracy performance plot saved to '{acc_path}'")

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        """
        Generates and saves the Confusion Matrix heatmap.
        """
        # Ensure confusion matrix labels are ordered matching INV_LABEL_MAP key list (0, 1, 2)
        classes = [INV_LABEL_MAP[i] for i in range(len(INV_LABEL_MAP))]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=False
        )
        plt.title("Confusion Matrix Heatmap")
        plt.ylabel("True Sentiment")
        plt.xlabel("Predicted Sentiment")
        
        cm_path = os.path.join(self.outputs_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        logger.info(f"Confusion Matrix plot saved to '{cm_path}'")

    def save_metrics_json(self, metrics: dict) -> None:
        """
        Saves training history and evaluations into a local JSON file.
        """
        json_path = os.path.join(self.outputs_dir, "training_metrics.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to JSON file '{json_path}'")
        except Exception as e:
            logger.error(f"Failed to write metrics to JSON: {str(e)}")

    def generate_evaluation_report(
        self, test_metrics: Dict[str, float], y_true: List[int], y_pred: List[int]
    ) -> None:
        """
        Generates a comprehensive Markdown evaluation report based on the test set.
        """
        report_path = os.path.join(self.report_dir, "evaluation_report.md")
        
        # Labels list
        labels = [0, 1, 2]
        target_names = [INV_LABEL_MAP[i] for i in labels]
        
        # Generate classification report
        cls_report = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        markdown_content = f"""# SentiScope Model Evaluation Report

This report outlines the performance of the fine-tuned BERT model (`bert-base-uncased`) on the SentiScope Amazon Product Reviews test set.

## Overall Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | {test_metrics['accuracy']:.4f} |
| **Precision (Weighted)** | {test_metrics['precision_weighted']:.4f} |
| **Recall (Weighted)** | {test_metrics['recall_weighted']:.4f} |
| **F1-Score (Weighted)** | {test_metrics['f1_weighted']:.4f} |

---

## Detailed Classification Report

```text
{cls_report}
```

---

## Confusion Matrix

| True \\ Predicted | Negative (0) | Neutral (1) | Positive (2) |
| :--- | :---: | :---: | :---: |
| **Negative (0)** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |
| **Neutral (1)** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |
| **Positive (2)** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |

---

## Performance Visualizations
All charts can be found in the `outputs/` directory:
- **Confusion Matrix Heatmap**: `outputs/confusion_matrix.png`
- **Training Loss Curves**: `outputs/training_loss.png`
- **Validation Accuracy Curves**: `outputs/validation_accuracy.png`

*Report compiled automatically by SentiScope Trainer.*
"""
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Evaluation report compiled successfully to '{report_path}'")
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {str(e)}")
