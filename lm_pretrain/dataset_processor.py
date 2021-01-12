import os
import datasets
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer
from typing import List, Optional, Dict
from ltp import LTP
import torch
import json
from lm_pretrain.config import DATA_ROBERTA_CACHE, ROBERTA_LOADING_SCRIPT


# class CustomDatasets(datasets.GeneratorBasedBuilder):
#     """TODO(cmrc2018): Short description of my dataset."""
#
#     # TODO(cmrc2018): Set up version.
#     VERSION = datasets.Version("0.1.0")
#
#     def _info(self):
#         # TODO(cmrc2018): Specifies the datasets.DatasetInfo object
#         return datasets.DatasetInfo(
#             # This is the description that will appear on the datasets page.
#             description="Custom datasets",
#             # datasets.features.FeatureConnectors
#             features=datasets.Features(
#                 {
#                     "id": datasets.Value("int32"),
#                     "text": datasets.Value("string")
#                     # These are the features of your dataset like images, labels ...
#                 }
#             ),
#             # If there's a common (input, target) tuple from the features,
#             # specify them here. They'll be used if as_supervised=True in
#             # builder.as_dataset.
#             supervised_keys=None,
#             # Homepage of the dataset for documentation
#             citation="",
#         )
#
#     def _split_generators(self, dl_manager):
#         """Returns SplitGenerators."""
#         # TODO(cmrc2018): Downloads the data and defines the splits
#         # dl_manager is a datasets.download.DownloadManager that can be used to
#         # download and extract URLs
#         urls_to_download = {
#             "train": self.config.data_files["train"],
#             "validation": self.config.data_files["validation"],
#         }
#         downloaded_files = dl_manager.download_and_extract(urls_to_download)
#
#         return [
#             datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
#             datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
#         ]
#
#     def _generate_examples(self, filepath):
#         """Yields examples."""
#         # TODO(cmrc2018): Yields (key, example) tuples from the dataset
#         with open(filepath, encoding="utf-8") as f:
#             data = json.load(f)
#             for example in data:
#                 id_ = example["id"]
#                 yield id_, {
#                     "id": example["id"],
#                     "text": example["text"],
#                 }


class Cmrc2018Datasets(datasets.GeneratorBasedBuilder):
    """TODO(cmrc2018): Short description of my dataset."""

    # TODO(cmrc2018): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(cmrc2018): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description="Custom datasets",
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(cmrc2018): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = {
            "train": self.config.data_files["train"],
            "validation": self.config.data_files["validation"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(cmrc2018): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    id_ = example["id"]
                    yield id_, {
                        "id": id_,
                        "text": context,
                    }


class AddZhReferences:
    def __init__(
            self,
            data_args,
            ltp_tokenizer: Optional[LTP] = None,
            bert_tokenizer: Optional[BertTokenizer] = None
    ):
        self.data_args = data_args
        self.ltp_tokenizer = ltp_tokenizer
        self.bert_tokenizer = bert_tokenizer
        pass

    def to_list(self, dataset: Optional[Dataset]) -> List[str]:
        """Convert dataset to list"""
        return [data["text"].strip() for data in dataset]

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def is_chinese(self, word: str):
        # word like '180' or '身高' or '神'
        for char in word:
            char = ord(char)
            if not self._is_chinese_char(char):
                return 0
        return 1

    def get_chinese_word(self, tokens: List[str]):
        word_set = set()

        for token in tokens:
            chinese_word = len(token) > 1 and self.is_chinese(token)
            if chinese_word:
                word_set.add(token)
        word_list = list(word_set)
        return word_list

    def add_sub_symbol(self, bert_tokens: List[str], chinese_word_set: set()):
        if not chinese_word_set:
            return bert_tokens
        max_word_len = max([len(w) for w in chinese_word_set])

        bert_word = bert_tokens
        start, end = 0, len(bert_word)
        while start < end:
            single_word = True
            if self.is_chinese(bert_word[start]):
                l = min(end - start, max_word_len)
                for i in range(l, 1, -1):
                    whole_word = "".join(bert_word[start: start + i])
                    if whole_word in chinese_word_set:
                        for j in range(start + 1, start + i):
                            bert_word[j] = "##" + bert_word[j]
                        start = start + i
                        single_word = False
                        break
            if single_word:
                start += 1
        return bert_word

    def prepare_chinese_references(self, lines: List[str]) -> List[List[int]]:
        ltp_res = []
        # 100句语句同时分词去重，且仅保留中文，存在512截断
        for i in range(0, len(lines), 100):
            res = self.ltp_tokenizer.seg(lines[i: i + 100])[0]
            res = [self.get_chinese_word(r) for r in res]
            ltp_res.extend(res)
        assert len(ltp_res) == len(lines)

        bert_res = []
        for i in range(0, len(lines), 100):
            res = self.bert_tokenizer(lines[i: i + 100],
                                      add_special_tokens=self.data_args.add_special_tokens,
                                      padding=self.data_args.padding,
                                      truncation=self.data_args.truncation,
                                      max_length=self.data_args.max_seq_length)
            bert_res.extend(res["input_ids"])
        assert len(bert_res) == len(lines)

        ref_ids = []
        for input_ids, chinese_word in zip(bert_res, ltp_res):
            input_tokens = []
            for id in input_ids:
                token = self.bert_tokenizer._convert_id_to_token(id)
                input_tokens.append(token)
            input_tokens = self.add_sub_symbol(input_tokens, chinese_word)
            ref_id, ref = [], []
            # We only save pos of chinese subwords start with ##, which mean is part of a whole word.
            for i, token in enumerate(input_tokens):
                if token[:2] == "##":
                    clean_token = token[2:]
                    # save chinese tokens' pos
                    if len(clean_token) == 1 and self._is_chinese_char(ord(clean_token)):
                        ref_id.append(i)
                        ref.append(token)
            ref_ids.append(ref_id)

        assert len(ref_ids) == len(bert_res)
        return ref_ids

    def load_chinese_references(self, dataset: DatasetDict, ref_type: str = "train"):
        data_zh_ref = os.path.join(DATA_ROBERTA_CACHE, ref_type)
        if os.path.exists(data_zh_ref):
            print("Loading dataset from cache.")
            dataset_dict = torch.load(data_zh_ref)
        else:
            dataset_dict = {c: dataset[ref_type][c] for c in dataset[ref_type].column_names}
            data_list = [data["text"] for data in dataset[ref_type]]
            dataset_dict["chinese_ref"] = self.prepare_chinese_references(data_list)
            torch.save(dataset_dict, data_zh_ref)
        dataset[ref_type] = Dataset.from_dict(dataset_dict)
        return dataset

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        dataset = self.load_chinese_references(dataset, "train")
        dataset = self.load_chinese_references(dataset, "validation")
        return dataset


class LmModelDatasetProcessor:
    def __init__(self, data_args, ltp_tokenizer, bert_tokenizer):
        self.data_args = data_args
        self.ltp_tokenizer = ltp_tokenizer
        self.bert_tokenizer = bert_tokenizer
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files
        self.datasets = datasets.load_dataset(path=ROBERTA_LOADING_SCRIPT,
                                              data_files={"train": self.data_args.train_file,
                                                          "validation": self.data_args.validation_file},
                                              cache_dir=None)
        self.datasets = self._data_duplication(self.datasets)
        # Instantiate callable object add_zh_references for the chinese references(WWM)
        self.add_zh_references = AddZhReferences(data_args=self.data_args,
                                                 ltp_tokenizer=self.ltp_tokenizer,
                                                 bert_tokenizer=self.bert_tokenizer)
        pass

    def _data_duplication(self, dataset: DatasetDict) -> DatasetDict:
        dataset_train_origin_size, dataset_validation_origin_size = len(dataset["train"]), len(dataset["validation"])
        # Concate train dataset lists
        dataset_train_list = [dataset["train"] for _ in range(self.data_args.duplicate_factor)]
        dataset["train"] = datasets.concatenate_datasets(dataset_train_list)
        # Concate validation dataset lists
        dataset_validation_list = [dataset["validation"] for _ in range(self.data_args.duplicate_factor)]
        dataset["validation"] = datasets.concatenate_datasets(dataset_validation_list)
        # Shuffle DatasetDict
        dataset = dataset.shuffle(seeds={"train": 42, "validation": 42})
        print(f"Dataset size: {len(dataset)}")
        print(f"Train dataset size: {len(dataset['train'])}")
        print(f"Validation dataset size: {len(dataset['validation'])}")
        assert len(dataset["train"]) == dataset_train_origin_size * self.data_args.duplicate_factor
        assert len(dataset["validation"]) == dataset_validation_origin_size * self.data_args.duplicate_factor
        return dataset

    def _encode_plus(self, examples: Dict) -> Dict:
        outputs = self.bert_tokenizer(examples["text"],
                                      add_special_tokens=self.data_args.add_special_tokens,
                                      padding=self.data_args.padding,
                                      truncation=self.data_args.truncation,
                                      max_length=self.data_args.max_seq_length)
        return outputs

    def __call__(self) -> DatasetDict:
        # Encode input tokens
        tokenized_datasets = self.datasets.map(function=self._encode_plus,
                                               batched=True,
                                               batch_size=1000)
        # Add zh references for WWM
        tokenized_datasets = self.add_zh_references(tokenized_datasets)
        print(tokenized_datasets.column_names)
        # Set format
        colunms = ["attention_mask", "input_ids", "token_type_ids", "chinese_ref"]
        tokenized_datasets.set_format(type="torch", columns=colunms)
        return tokenized_datasets

