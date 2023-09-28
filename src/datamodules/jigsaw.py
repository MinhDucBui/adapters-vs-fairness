from typing import Optional
import os.path
from src.utils.utils import get_logger
from src.datamodules.base import BaseDataModule
from datasets import Dataset
import pandas as pd


log = get_logger(__name__)

SEED = 42


class JigsawDataModule(BaseDataModule):
    def __init__(
            self,
            train_data_path,
            test_data_path,
            testing=False,
            split_seed=42,
            shuffle=True,
            *args,
            **kwargs,
    ):

        # TODO: Different Tokenizer for student/teacher
        super().__init__(*args, **kwargs)

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.testing = testing
        self.split_seed = split_seed
        self.shuffle = shuffle

    @property
    def num_labels(self) -> int:
        return 2

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            log.info("Check Data...")
            dataset = self.load_dataset(self.train_data_path)
            dataset = dataset.train_test_split(test_size=0.2,
                                               shuffle=self.shuffle,
                                               seed=self.split_seed)

            self.data_train = dataset["train"]
            self.data_val = dataset["test"]
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            dataset = self.load_dataset(self.test_data_path)
            self.data_test = dataset

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to
        be done only from a single process in distributed settings, e.g.,
        download, tokenize, etc...

        """

        # Get Training Data from cc100 for MLM
        log.info("Check Data...")

        # Check if the dataset files exist
        if os.path.exists(self.train_data_path) and os.path.exists(self.test_data_path):
            log.info(f"The dataset files exists.")
        else:
            # Give out an error if the dataset files do not exist.
            # Print the path of the dataset files
            log.error("The dataset files do not exist.")
            if not os.path.exists(self.train_data_path):
                FileNotFoundError(
                    f"Train data file not found at: {self.train_data_path}")
            if not os.path.exists(self.test_data_path):
                FileNotFoundError(
                    f"Test data file not found at: {self.test_data_path}")

    # TODO: Move to Collator

    def load_dataset(self, path):
        log.warning("Load CSV File...")
        df = pd.read_csv(path)
        if self.testing:
            df = df.iloc[:100]
        log.warning("Create coarsed grained annotation...")
        df = self.create_coarsed_grained_annotation(df)
        df = df.rename(columns={"comment_text": "text", "target": "labels"})
        # Define a threshold for binary conversion
        threshold = 0.5
        # Apply the threshold to convert the "labels" column to binary
        df["labels"] = (df["labels"] >= threshold).astype(float)
        # Create a Hugging Face dataset from the modified DataFrame
        dataset = Dataset.from_pandas(df)

        # https://github.com/huggingface/datasets/issues/2583
        # language_dataset = language_dataset.with_format("torch")
        # dataset = dataset.with_format("torch")
        log.warning("Tokenize Dataset...")
        # TODO: This Part should happen in Collator?
        dataset = dataset.map(self.tokenization)
        dataset = dataset.remove_columns(["text"])
        return dataset

    def tokenization(self, example):
        return self.tokenizer(example["text"], truncation=True, padding=False, return_token_type_ids=True)

    def create_coarsed_grained_annotation(self, df, drop_irrelevant_columns=True):
        MAPPING = {"religion": ["atheist", "buddhist", "muslim", "christian", "hindu", "jewish", "other_religion"],
                   "race": ["white", "asian", "black", "latino", "other_race_or_ethnicity"],
                   "gender_and_sexual_orientation": ["bisexual", "female", "male", "heterosexual",
                                                     "homosexual_gay_or_lesbian", "transgender", "other_gender",
                                                     "other_sexual_orientation"]}

        result = pd.DataFrame({key: (df[MAPPING[key]] >= 0.5).any(
            axis=1).astype(int) for key in MAPPING})
        df_combined = pd.concat([df, result], axis=1)

        # Fill Nan values in comment_text with empty string
        df_combined["comment_text"] = df_combined["comment_text"].fillna("")

        # For Test Set
        if "toxicity" in df_combined:
            df_combined = df_combined.rename(columns={"toxicity": "target"})

        if drop_irrelevant_columns:
            df_combined = df_combined[[
                "comment_text", "target", "religion", "race", "gender_and_sexual_orientation"]]

        return df_combined
