import json
import pandas as pd
import os.path
from src.utils.utils import get_logger
from src.datamodules.base import BaseDataModule
from datasets import Dataset
import pandas as pd
from collections import Counter
import os
import pickle


log = get_logger(__name__)

SEED = 42

P_MAPPING_INDEX_CLASS = {
    0: 'professor',
    1: 'physician',
    2: 'attorney',
    3: 'photographer',
    4: 'journalist',
    5: 'nurse',
    6: 'psychologist',
    7: 'teacher',
    8: 'dentist',
    9: 'surgeon',
    10: 'architect',
    11: 'painter',
    12: 'model',
    13: 'poet',
    14: 'filmmaker',
    15: 'software_engineer',
    16: 'accountant',
    17: 'composer',
    18: 'dietitian',
    19: 'comedian',
    20: 'chiropractor',
    21: 'pastor',
    22: 'paralegal',
    23: 'yoga_teacher',
    24: 'dj',
    25: 'interior_designer',
    26: 'personal_trainer',
    27: 'rapper'
}
P_MAPPING_CLASS_INDEX = {v: k for k, v in P_MAPPING_INDEX_CLASS.items()}


G_MAPPIN_INDEX_CLASS = {0: 'f', 1: 'm'}
G_MAPPING_CLASS_INDEX = {v: k for k, v in G_MAPPIN_INDEX_CLASS.items()}



class BiasBiosDataModule(BaseDataModule):
    def __init__(
            self,
            dataset_path,
            scrup_gender=False,
            post_id_division_path=None,
            train_data_path=None,
            test_data_path=None,
            *args,
            **kwargs,
    ):

        # TODO: Different Tokenizer for student/teacher
        super().__init__(*args, **kwargs)
        
        self.scrup_gender = scrup_gender

        self.train_dataset_path = os.path.join(dataset_path, "train.pickle")
        self.dev_dataset_path = os.path.join(dataset_path, "dev.pickle")
        self.test_dataset_path = os.path.join(dataset_path, "test.pickle")

    @property
    def num_labels(self) -> int:
        return 2

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.data_train, self.data_val, self.data_test = self.load_dataset(stage)

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to
        be done only from a single process in distributed settings, e.g.,
        download, tokenize, etc...

        """

        # Get Training Data from cc100 for MLM
        log.info("Check Data...")

        # Check if the dataset files exist
        if os.path.exists(self.train_dataset_path) \
        and os.path.exists(self.dev_dataset_path) \
        and os.path.exists(self.test_dataset_path):
            log.info(f"The dataset files exists.")
        else:
            # Give out an error if the dataset files do not exist.
            # Print the path of the dataset files
            log.error("The dataset files do not exist.")

    def load_dataset(self, stage):
        log.info("Load Pickle File...")
        # Open the datasets
        with open(self.train_dataset_path, 'rb') as file:
            train_data = pickle.load(file)
        with open(self.dev_dataset_path, 'rb') as file:
            dev_data = pickle.load(file)
        with open(self.test_dataset_path, 'rb') as file:
            test_data = pickle.load(file)
        
        # Use pandas
        train_data = pd.DataFrame(train_data)
        dev_data = pd.DataFrame(dev_data)
        test_data = pd.DataFrame(test_data)
        
        if self.scrup_gender:
            train_data["text"] = train_data["text_without_gender"]
            dev_data["text"] = dev_data["text_without_gender"]
            test_data["text"] = test_data["text_without_gender"]
        else:
            train_data["text"] = train_data["hard_text_untokenized"]
            dev_data["text"] = dev_data["hard_text_untokenized"]
            test_data["text"] = test_data["hard_text_untokenized"]
        
        # Print some examples
        for i in range(5):
            print("---------------------EXAMPLE {}-------------------".format(i))
            print("Hard Text: " + train_data["text"].iloc[i])
            print("Profession: " + train_data["p"].iloc[i])
            print("Label: " + train_data["g"].iloc[i])
            
        # Labels (Profession)
        train_data['labels'] = train_data['p'].map(P_MAPPING_CLASS_INDEX)
        dev_data['labels'] = dev_data['p'].map(P_MAPPING_CLASS_INDEX)
        test_data['labels'] = test_data['p'].map(P_MAPPING_CLASS_INDEX)
        
        # Gender
        train_data['g'] = train_data['g'].map(G_MAPPING_CLASS_INDEX).astype(float)
        dev_data['g'] = dev_data['g'].map(G_MAPPING_CLASS_INDEX).astype(float)
        test_data['g'] = test_data['g'].map(G_MAPPING_CLASS_INDEX).astype(float)
        
        # TESTING PURPOSES
        #train_data = train_data.iloc[:1000]
        #dev_data = dev_data.iloc[:1000]
        #test_data = test_data.iloc[:1000]

        log.info("Tokenize Dataset...")
        if stage == "fit":
            data_train = self.create_dataset(train_data)
            data_val = self.create_dataset(dev_data)
            data_test = None
        elif stage == "test":
            data_train = None
            data_val = None
            data_test = self.create_dataset(test_data)
        return data_train, data_val, data_test

    def create_dataset(self, df):
        df = df.drop(["hard_text", "hard_text_untokenized", 
                      "text_without_gender", "start", 'p'], axis=1)
        # Shuffle
        df = df.sample(frac = 1, random_state=SEED)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.tokenization)
        dataset = dataset.remove_columns(["__index_level_0__", "text"])
        return dataset

    def tokenization(self, example):
        return self.tokenizer(example["text"], truncation=True,
                              padding=False, return_token_type_ids=True)
