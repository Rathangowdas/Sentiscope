import torch
from torch.utils.data import Dataset
from typing import Dict


class SentimentDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for SentiScope.
    Holds the tokenized inputs (input_ids, attention_mask) and target label IDs
    for feeding into a BERT sequence classifier.
    """

    def __init__(self, tokenized_data: Dict[str, torch.Tensor]):
        """
        Args:
            tokenized_data (dict): Dictionary returned by SentimentTokenizer.tokenize,
                                  containing keys 'input_ids', 'attention_mask', 
                                  and optionally 'labels'.
        """
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        
        # Labels are optional (e.g. inference time dataset doesn't have them)
        self.labels = tokenized_data.get("labels", None)

    def __len__(self) -> int:
        """
        Returns the total number of items in this dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the PyTorch tensor dictionary for a given index.
        """
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
            
        return item
