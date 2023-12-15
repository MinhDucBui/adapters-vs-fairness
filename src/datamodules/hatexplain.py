import json
import pandas as pd
import os.path
from src.utils.utils import get_logger
from src.datamodules.base import BaseDataModule
from datasets import Dataset
import pandas as pd
from collections import Counter


log = get_logger(__name__)

SEED = 42


class HateXplainDataModule(BaseDataModule):
    def __init__(
            self,
            dataset_path,
            post_id_division_path,
            train_data_path=None,
            test_data_path=None,
            *args,
            **kwargs,
    ):

        # TODO: Different Tokenizer for student/teacher
        super().__init__(*args, **kwargs)

        self.dataset_path = dataset_path
        self.post_id_division_path = post_id_division_path

        self.group_mapping = {"race": ["African", "Arab", "Asian", "Caucasian", "Hispanic"],
                        "religion": ["Islam", "Buddhism", "Jewish", "Hindu", "Christian"],
                        "gender_and_sexual_orientation": ["Men", "Women", "Homosexual"]}

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
        if os.path.exists(self.dataset_path) and os.path.exists(self.post_id_division_path):
            log.info(f"The dataset files exists.")
        else:
            # Give out an error if the dataset files do not exist.
            # Print the path of the dataset files
            log.error("The dataset files do not exist.")
            if not os.path.exists(self.dataset_path):
                FileNotFoundError(
                    f"Data file not found at: {self.dataset_path}")
            if not os.path.exists(self.post_id_division_path):
                FileNotFoundError(
                    f"Division file not found at: {self.post_id_division_path}")

    def load_dataset(self, stage):
        log.info("Load CSV File...")
        df = get_annotated_data(self.dataset_path)
        log.info("Create coarsed grained annotation...")
        df["community"] = df.apply(
            lambda x: generate_target_information(x), axis=1)
        for key in self.group_mapping:
            df[key] = df.apply(lambda x: 1 if any(item in x['community']
                               for item in self.group_mapping[key]) else 0,
                               axis=1)

        df = df[["post_id", "text", 'race', 'religion', 'gender_and_sexual_orientation', "labels"]]
        df["text"] = df["text"].apply(lambda x: " ".join(x))

        # Print some examples
        for i in range(10):
            print(df["text"].iloc[i], df["labels"].iloc[i])

        # Split Dataset
        # The post_id_divisions file stores the train, val, test split ids.
        # We select only the test ids.
        with open(self.post_id_division_path, 'r') as fp:
            post_id_dict = json.load(fp)
        log.info("Tokenize Dataset...")
        if stage == "fit":
            data_train = self.create_dataset(df, post_id_dict, subset="train")
            data_val = self.create_dataset(df, post_id_dict, subset="val")
            data_test = None
        elif stage == "test":
            data_train = None
            data_val = None
            data_test = self.create_dataset(df, post_id_dict, subset="test")
        return data_train, data_val, data_test

    def create_dataset(self, df, post_id_dict, subset="train"):
        dataset_subset = df[df['post_id'].isin(post_id_dict[subset])]
        dataset_subset = dataset_subset.drop("post_id", axis=1)
        # Shuffle
        dataset_subset = dataset_subset.sample(frac = 1, random_state=SEED)
        dataset_subset = Dataset.from_pandas(dataset_subset)
        dataset_subset = dataset_subset.map(self.tokenization)
        dataset_subset = dataset_subset.remove_columns(["text"])
        return dataset_subset

    def tokenization(self, example):
        return self.tokenizer(example["text"], truncation=True,
                              padding=False, return_token_type_ids=True)


def get_annotated_data(data_path):
    # temp_read = pd.read_pickle(params['data_file'])
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    dict_data = []
    for key in data:
        temp = {}
        temp['post_id'] = key
        temp['text'] = data[key]['post_tokens']
        final_label = []
        for i in range(1, 4):
            temp['annotatorid' +
                 str(i)] = data[key]['annotators'][i-1]['annotator_id']
            temp['target'+str(i)] = data[key]['annotators'][i-1]['target']
            temp['label'+str(i)] = data[key]['annotators'][i-1]['label']
            final_label.append(temp['label'+str(i)])

        final_label_id = max(final_label, key=final_label.count)
        temp['rationales'] = data[key]['rationales']

        if (final_label.count(final_label_id) == 1):
            temp['labels'] = 'undecided'
        else:
            if (final_label_id in ['hatespeech', 'offensive']):
                final_label_id = 1.0
            else:
                final_label_id = 0.0
            temp['labels'] = final_label_id

        dict_data.append(temp)
    temp_read = pd.DataFrame(dict_data)
    return temp_read


def generate_target_information(x):

    # All the target communities tagged for this post
    all_targets = x['target1']+x['target2']+x['target3']
    community_dict = dict(Counter(all_targets))

    sample_targets = []
    for key in community_dict:
        #if community_dict[key] > 1:
        sample_targets.append(key)

    return sample_targets
