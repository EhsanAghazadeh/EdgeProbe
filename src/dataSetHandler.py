import torch
import transformers
import datasets
import json
import pandas as pd
import numpy as np
from numpy.core._multiarray_umath import ndarray

one_hot = lambda idx, length: np.array([
    1 if i in idx else 0 for i in range(length)
])

def get_indices(label2index, labels):
    return [label2index[label] for label in labels]


def tokenize_and_one_hot(sample, **fn_kwargs):
    tokenized_input = fn_kwargs['tokenizer'](sample['text'])
    word_ids = tokenized_input.word_ids()
    tokenized_input['span1_indices'] = list(range(
        word_ids.index(sample["span1"][0]),
        len(word_ids) - 1 - word_ids[::-1].index(sample["span1"][1] - 1) + 1
    ))
    tokenized_input['span1_len'] = len(tokenized_input['span1_indices'])
    tokenized_input['span2_indices'] = list(range(
        word_ids.index(sample["span2"][0]),
        len(word_ids) - 1 - word_ids[::-1].index(sample["span2"][1] - 1) + 1
    ))
    tokenized_input['span2_len'] = len(tokenized_input['span2_indices'])
    label2index = fn_kwargs["label2index"]
    labels_len = fn_kwargs["labels_len"]
    tokenized_input["one_hot_label"] = one_hot(get_indices(
        label2index,
        sample["label"]
    ), labels_len)
    return tokenized_input


class DatasetInfo():
    def __init__(
            self,
            name,
            num_spans,
            frac,
            max_span_len=8
    ):
        self.name = name
        self.num_spans = num_spans
        self.frac = frac
        self.max_span_len = max_span_len


class DatasetHandler():
    unique_labels_list: ndarray

    def __init__(
            self,
            root_dir_path,
            dataset_info: DatasetInfo,
            tokenizer,
            encoding='utf-8'
    ):
        self.label2index = None
        self.unique_labels_list = None
        self.dataset_dict = datasets.DatasetDict()
        self.tokenized_dataset: datasets.DatasetDict = None
        self.encoding = encoding
        self.info = dataset_info
        self.tokenizer = tokenizer

        data_types = ["train", "dev", "test"]
        for data_type in data_types:
            self.dataset_dict[data_type] = self.json2dataset(
                path=root_dir_path + dataset_info.name + '/train.json',
                frac=dataset_info.frac
            )

        self.tokenize_inputs_and_one_hot_labels()

    def json2dataset(self, path, frac):

        data_df = self.json2df(path)
        if frac != 1:
            data_df = data_df.sample(
                frac=frac,
                random_state=None
            ).sort_index().reset_index(
                drop=True
            )

        return datasets.Dataset.from_pandas(data_df)

    def json2df(self, path):
        samples = []
        with open(path, encoding=self.encoding) as f:
            for idx, line in enumerate(f):
                instance = json.loads(line)
                for target in instance['targets']:
                    if self.info.name != 'semeval_data':
                        curr_sample = {'text_id': idx
                            , 'span1': target['span1']
                            , 'span2': target['span2']
                            , 'text': instance['text']
                            , 'label': np.array([target['label']])
                                       }
                        samples.append(curr_sample)
                    else:
                        curr_sample = {'text_id': instance['info']['id']
                            , 'span1': target['span1']
                            , 'span2': target['span2']
                            , 'text': instance['text']
                            , 'label': np.array([target['label']])
                                       }
                        samples.append(curr_sample)

        return pd.DataFrame.from_dict(samples)

    def tokenize_inputs_and_one_hot_labels(self):
        train_labels_df = pd.DataFrame(
            self.dataset_dict['train']['label'],
            columns=['label']
        )
        test_labels_df = pd.DataFrame(
            self.dataset_dict['test']['label'],
            columns=['label']
        )
        dev_labels_df = pd.DataFrame(
            self.dataset_dict['dev']['label'],
            columns=['label']
        )

        all_labels_df = pd.concat(
            [
                train_labels_df,
                test_labels_df,
                dev_labels_df
            ],
            ignore_index=True
        )

        self.unique_labels_list = np.array(pd.unique(all_labels_df['label']))
        self.label2index = {
            val: idx for idx, val in enumerate(self.unique_labels_list)
        }

        self.tokenized_dataset = self.dataset_dict.map(
            tokenize_and_one_hot,
            fn_kwargs= {
                'label2index': self.label2index,
                'labels_len': len(self.unique_labels_list),
                'tokenizer': self.tokenizer
            },
            batched=False
        )