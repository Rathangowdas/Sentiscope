import re
import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split
from utils import LABEL_MAP

logger = logging.getLogger("SentiScope")


class DataProcessor:
    """
    Handles CSV loading, text preprocessing, validation, label encoding, 
    and splitting of the Amazon Product Review dataset.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans input text by:
        - Removing HTML tags
        - Lowercasing
        - Keeping alphanumeric characters and essential punctuation
        - Normalizing whitespaces
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML tags using regex
        text = re.sub(r"<[^>]+>", " ", text)
        
        # Lowercase
        text = text.lower()
        
        # Keep letters, numbers, spaces, and basic punctuation (, . ! ? ' " -)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"-]", "", text)
        
        # Normalize multiple spaces and linebreaks
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Loads the CSV dataset and validates columns.
        """
        try:
            logger.info(f"Loading dataset from {data_path}...")
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at path: '{data_path}'")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {str(e)}")

        # Validate required columns
        required_cols = ["text", "label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Dataset missing required column: '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )

        # Drop rows with null values in required columns
        initial_len = len(df)
        df = df.dropna(subset=required_cols)
        dropped_nulls = initial_len - len(df)
        if dropped_nulls > 0:
            logger.warning(f"Dropped {dropped_nulls} row(s) with null text or labels.")

        # Handle empty string values after cleaning
        df["text"] = df["text"].astype(str)
        
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning, filters out invalid reviews or labels, 
        and encodes string labels into numeric IDs.
        """
        df = df.copy()

        # Clean text column
        df["clean_text"] = df["text"].apply(self.clean_text)

        # Drop rows where cleaned text is empty
        empty_clean = df["clean_text"] == ""
        if empty_clean.any():
            dropped_empty = empty_clean.sum()
            df = df[~empty_clean]
            logger.warning(f"Dropped {dropped_empty} row(s) with empty review texts after cleaning.")

        # Encode labels using LABEL_MAP
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        
        # Verify that all labels are valid
        valid_mask = df["label"].isin(LABEL_MAP.keys())
        invalid_labels = df.loc[~valid_mask, "label"].unique()
        
        if len(invalid_labels) > 0:
            logger.warning(f"Found invalid labels in dataset: {list(invalid_labels)}. Removing these rows.")
            df = df[valid_mask]

        df["label_id"] = df["label"].map(LABEL_MAP)
        
        if len(df) == 0:
            raise ValueError("Preprocessing resulted in an empty dataset. Check your CSV texts and labels.")

        logger.info(f"Dataset preprocessed successfully. Total records: {len(df)}")
        return df

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits preprocessed DataFrame into train, val, and test subsets.
        Uses stratification to maintain class distribution, with a safe fallback
        for extremely small datasets where stratification is mathematically impossible.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "Split ratios must sum to 1.0"
        
        y = df["label_id"].values
        
        # Determine if we can use stratified splitting safely
        # Stratification requires at least 2 instances per class
        class_counts = df["label_id"].value_counts()
        can_stratify = len(class_counts) > 1 and (class_counts >= 2).all()
        
        stratify_y = y if can_stratify else None
        if not can_stratify:
            logger.warning(
                "Some classes have less than 2 instances. "
                "Stratified splitting is disabled; falling back to standard split."
            )

        # First split: Train vs Temp (Val + Test)
        temp_size = val_size + test_size
        try:
            df_train, df_temp = train_test_split(
                df,
                train_size=train_size,
                random_state=42,
                stratify=stratify_y
            )
            
            # Second split: Val vs Test from Temp
            val_relative_size = val_size / temp_size
            
            # Recalculate stratification for the second split
            y_temp = df_temp["label_id"].values
            temp_counts = df_temp["label_id"].value_counts()
            can_stratify_temp = len(temp_counts) > 1 and (temp_counts >= 2).all()
            stratify_temp_y = y_temp if can_stratify_temp else None

            df_val, df_test = train_test_split(
                df_temp,
                train_size=val_relative_size,
                random_state=42,
                stratify=stratify_temp_y
            )
        except Exception as e:
            logger.warning(
                f"Split failed with standard split configuration ({str(e)}). "
                "Performing fallback split without stratification."
            )
            # Safe absolute manual slicing fallback
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n = len(df)
            n_train = max(1, int(n * train_size))
            n_val = max(1, int(n * val_size))
            
            df_train = df.iloc[:n_train]
            df_val = df.iloc[n_train:n_train + n_val]
            df_test = df.iloc[n_train + n_val:]
            
            # Ensure splits are non-empty
            if len(df_val) == 0 and len(df) >= 2:
                df_val = df.iloc[-1:]
            if len(df_test) == 0 and len(df) >= 3:
                df_test = df.iloc[-2:-1]

        logger.info(
            f"Dataset split completed - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
        )
        return df_train, df_val, df_test
