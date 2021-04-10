import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

from src.dataSetHandler import DatasetHandler
from src.model import EPModel
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


class EPTrainer:
    def __init__(
            self,
            pretrained_model,
            dataset_handler: DatasetHandler,
            batch_size=16,
            lr=5e-4,
            device='cuda',
            verbose=True,
            model_checkpoint=None
    ):
        self.current_hidden_states = None
        self.last_sentence_hidden_states = {
            'text_id': None,
            'input_ids': [],
            'hidden_states': []
        }
        self.last_input_ids = None
        self.pretrained_model = pretrained_model
        self.dataset_handler = dataset_handler
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.model_checkpoint = model_checkpoint

        self.num_layers, self.input_span_len, self.embedding_dim, self.num_classes, max_span_len = self.get_pretrained_model_properties()
        self.mlp_device = "cuda"
        if self.model_checkpoint == None:
            self.model = EPModel(
                max_span_len=max_span_len,
                embedding_dim=self.embedding_dim,
                num_classes=self.num_classes,
                num_layers=self.num_layers,
                pool_method='max',
                device=self.mlp_device
            )

        self.loss_hist = {
            'train': [],
            'test': [],
            'dev': []
        }

    def train(self, epochs=20):
        train_tokenized_dataset = self.dataset_handler.tokenized_dataset['train']
        test_tokenized_dataset = self.dataset_handler.tokenized_dataset['test']
        dev_tokenized_dataset = self.dataset_handler.tokenized_dataset['dev']

        self.model.to(self.mlp_device)
        dataset_len = len(train_tokenized_dataset['input_ids'])

        # print('happy')

        self.loss_hist['train'].append(
            self.calc_loss(
                tokenized_dataset=train_tokenized_dataset,
                print_metrics=True
            )
        )

        # print('happy')

        self.loss_hist['test'].append(
            self.calc_loss(
                tokenized_dataset=test_tokenized_dataset,
                print_metrics=True
            )
        )
        self.loss_hist['dev'].append(
            self.calc_loss(
                tokenized_dataset=dev_tokenized_dataset,
                print_metrics=True
            )
        )
        print('[%d] train_loss: %.4f, val_loss: %.4f' % (0, self.loss_hist["train"][-1], self.loss_hist["dev"][-1]))
        for epoch in range(epochs):
            cumulative_loss = 0
            steps = 0

            self.draw_weights()
            for i in tqdm(range(0, dataset_len, self.batch_size)):
                step = self.batch_size
                if i + step > dataset_len:
                    step = dataset_len - i

                spans, max_span_len = self.prepare_batch_data(
                    tokenized_dataset=train_tokenized_dataset,
                    start_idx=i,
                    end_idx=i + step,
                    pad=True
                )

                self.model.max_span_len = max_span_len

                labels = spans['one_hot_labels']

                self.model.optimizer.zero_grad()
                outputs = self.model(spans)
                loss = self.model.criterion(
                    outputs.to(self.device),
                    labels.float().to(self.device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(),
                    5.0
                )
                self.model.optimizer.step()

                cumulative_loss += loss.item()
                steps += 1

            self.loss_hist['train'].append(
                cumulative_loss / steps
            )
            self.loss_hist['dev'].append(
                self.calc_loss(
                    tokenized_dataset=dev_tokenized_dataset,
                    print_metrics=True
                )
            )
            self.loss_hist['test'].append(
                self.calc_loss(
                    tokenized_dataset=test_tokenized_dataset,
                    print_metrics=True
                )
            )
            print('[%d] loss: %.4f, val_loss: %.4f, test_loss: %.4f' % (
                epoch + 1, self.loss_hist["train"][-1], self.loss_hist["dev"][-1], self.loss_hist["test"][-1]))

    def draw_weights(self):
        weights = self.model.weights.tolist()
        plt.bar(
            np.arange(len(weights)),
            weights
        )
        plt.ylabel('Weight')
        plt.xlabel('Layer')
        plt.show()

    def calc_loss(
            self,
            tokenized_dataset,
            batch_size=64,
            print_metrics=False,
            just_micro=False
    ):
        with torch.no_grad():
            cumulative_loss = 0
            dataset_len = len(tokenized_dataset['input_ids'])
            steps = 0
            preds = None
            for i in tqdm(range(0, dataset_len, batch_size)):
                step = batch_size
                if i + batch_size > dataset_len:
                    step = dataset_len - i

                spans_dict, max_span_len = self.prepare_batch_data(
                    tokenized_dataset=tokenized_dataset,
                    start_idx=i,
                    end_idx=i + step,
                    pad=True
                )
                outputs = self.model(spans_dict)
                preds = outputs if i == 0 else torch.cat(
                    (preds, outputs),
                    dim=0
                )
                loss = self.model.criterion(
                    outputs,
                    spans_dict['one_hot_labels']
                )
                cumulative_loss += loss.item()
                steps += 1

        if print_metrics:
            preds = preds.cpu().argmax(-1)  # I don't knwo what's happening here
            y = np.asarray(tokenized_dataset['one_hot_label']).argmax(-1)
            print(preds[0:9])
            print(y[0:9])
            labels_list = self.dataset_handler.unique_labels_list
            if not just_micro:
                print(classification_report(
                    preds,
                    y,
                    target_names=labels_list,
                    labels=range(len(labels_list))
                ))
            print("MICRO F1 score is: {}".format(
                f1_score(
                    preds,
                    y,
                    average='micro'
                )
            ))

        return cumulative_loss / steps

    def prepare_batch_data(
            self,
            tokenized_dataset,
            start_idx,
            end_idx,
            pad=False
    ):
        max_sent_len = max(
            tokenized_dataset[start_idx:end_idx]['sent_len']
        )

        span_rerps_dict, max_span_len = self.extract_embeddings_from_pretrained_model(
            tokenized_dataset=tokenized_dataset,
            start_idx=start_idx,
            end_idx=end_idx,
            pad=pad,
            max_sent_len=max_sent_len
        )

        encoded_repr = torch.tensor(
            np.asarray(span_rerps_dict['encoded_repr'])
        ).float().to(self.mlp_device)

        span1_start_ids = torch.tensor(
            np.asarray(span_rerps_dict['span1_start_ids'])
        ).float().to(self.mlp_device)

        span1_end_ids = torch.tensor(
            np.asarray(span_rerps_dict['span1_end_ids'])
        ).float().to(self.mlp_device)

        span2_start_ids = torch.tensor(
            np.asarray(span_rerps_dict['span2_start_ids'])
        ).float().to(self.mlp_device)

        span2_end_ids = torch.tensor(
            np.asarray(span_rerps_dict['span2_end_ids'])
        ).float().to(self.mlp_device)

        one_hot_labels = torch.tensor(
            np.asarray(
                span_rerps_dict['one_hot_label']
            )
        ).float().to(self.mlp_device)

        return {
                   'encoded_repr': encoded_repr,
                   'span1_start_ids': span1_start_ids,
                   'span1_end_ids': span1_end_ids,
                   'span2_start_ids': span2_start_ids,
                   'span2_end_ids': span2_end_ids,
                   'one_hot_labels': one_hot_labels
               }, max_span_len

    def get_pretrained_model_properties(self):

        max_sent_len = max(
            self.dataset_handler.tokenized_dataset['train'][0:3]['sent_len']
        )
        span_reprs_dict, max_span_len = self.extract_embeddings_from_pretrained_model(
            self.dataset_handler.tokenized_dataset['train'],
            0,
            3,
            max_sent_len=max_sent_len,
            pad=True,
            cache=True
        )

        for val in span_reprs_dict['encoded_repr']:
            print(
                f"Pretrained language model embeddings shape: {val.shape}"
            )
        span_reps_sample = span_reprs_dict['encoded_repr'][0]
        assert len(span_reps_sample.shape) == 3
        num_layers = span_reps_sample.shape[0]
        span_len = span_reps_sample.shape[1]
        embedding_dim = span_reps_sample.shape[2]
        return num_layers, span_len, embedding_dim, len(self.dataset_handler.unique_labels_list), max_span_len

    def extract_embeddings_from_pretrained_model(
            self,
            tokenized_dataset,
            start_idx,
            end_idx,
            max_sent_len,
            pad=True,
            cache=True
    ):
        max_span_len = max(
            max(
                tokenized_dataset[
                start_idx:end_idx
                ]['span1_len']
            ),
            max(
                tokenized_dataset[
                start_idx:end_idx
                ]['span2_len']
            )
        )

        num_spans = self.dataset_handler.info.num_spans
        language_model_dict = self.init_span_dict(num_spans, pad)

        for i in range(start_idx, end_idx):
            if self.last_sentence_hidden_states['text_id'] != None and tokenized_dataset[
                                                                       i:i + 1
                                                                       ]['text_id'] == self.last_sentence_hidden_states[
                'text_id']:
                idx = self.last_sentence_hidden_states[
                    'input_ids'
                ].index(
                    tokenized_dataset[i:i + 1][
                        'input_ids'
                    ][0]
                )
                self.current_hidden_states = self.last_sentence_hidden_states[
                    'hidden_states'
                ][idx]
            else:
                with torch.no_grad():
                    sample_dict = tokenized_dataset[i:i + 1]
                    # print(sample_dict)
                    text_id = sample_dict['text_id']
                    # print(f"text id is: {text_id}")
                    pretrained_model_input = torch.tensor(
                        np.array(
                            sample_dict['input_ids']
                        )
                    ).to(self.device)
                    outputs = self.pretrained_model(
                        pretrained_model_input
                    )
                    current_hidden_states = np.asarray([
                        val.detach().cpu().numpy() for val in outputs.hidden_states
                    ])[:, 0]

                    if cache:
                        self.last_sentence_hidden_states['text_id'] = text_id
                        self.last_sentence_hidden_states['input_ids'] = sample_dict['input_ids']
                        self.last_sentence_hidden_states['hidden_states'] = current_hidden_states

            row = tokenized_dataset[i]

            curr_encoded = self.last_sentence_hidden_states[
                'hidden_states'
            ]
            language_model_dict['encoded_repr'].append(
                np.concatenate(
                    [
                        curr_encoded,
                        np.zeros((
                            curr_encoded.shape[0],
                            max_sent_len - curr_encoded.shape[1],
                            curr_encoded.shape[2]
                        ))
                    ],
                    axis=1
                )
            )

            language_model_dict['span1_start_ids'].append(row[
                                                              'span1_indices'
                                                          ][0])

            language_model_dict['span1_end_ids'].append(row[
                                                            'span1_indices'
                                                        ][-1])

            language_model_dict['one_hot_label'].append(
                row['one_hot_label']
            )
            language_model_dict['label'].append(
                row['label']
            )

            if num_spans == 2:
                language_model_dict['span2_start_ids'].append(row[
                                                                  'span2_indices'
                                                              ][0])

                language_model_dict['span2_end_ids'].append(row[
                                                                'span2_indices'
                                                            ][-1])

        return language_model_dict, max_span_len

    @staticmethod
    def init_span_dict(num_spans, pad):
        language_model_dict = {
            'encoded_repr': [],
            'label': [],
            'one_hot_label': [],
            'span1_start_ids': [],
            'span1_end_ids': []
        }
        if num_spans == 2:
            language_model_dict['span2_start_ids'] = []
            language_model_dict['span2_end_ids'] = []

        return language_model_dict

    @staticmethod
    def pad_span(
            span_repr,
            span_max_len
    ):
        num_layers = span_repr.shape[0]
        span_len = span_repr.shape[1]
        embedding_dim = span_repr.shape[2]
        if span_len > span_max_len:
            raise Exception(
                f"Error: {span_len} is more than span_max_len{span_max_len}"
            )
        attention_mask = np.asarray(
            [1] * span_len + [0] * (span_max_len - span_len)
        )
        padded_span_repr = np.concatenate(
            [
                span_repr,
                np.zeros((
                    num_layers,
                    span_max_len - span_len,
                    embedding_dim
                ))
            ],
            axis=1
        )

        assert attention_mask.shape == (span_max_len,), f"""
incorrect shape for attention mask: {attention_mask.shape} != ({span_max_len}, )"""
        assert padded_span_repr.shape == (num_layers, span_max_len, embedding_dim), f"""
incorrect shape for span representation: {padded_span_repr.shape} != ({num_layers}, {span_max_len}, {embedding_dim})"""
        return padded_span_repr, attention_mask
