import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

from src.dataSetHandler import DatasetHandler
from src.model import CorefModel
from src.model import EPModel
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def isCached(current, cache):
    for cached in cache:
        if np.array_equal(current, cached):
            return True

    return False

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
        # self.cached_hidden_states = {
        #     'input_ids': [],
        #     'hidden_states': []
        # }
        self.last_setance_hidden_states = {
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

        self.num_layers, self.input_span_len, self.embedding_dim, self.num_classes = self.get_pretrained_model_properties()
        self.mlp_device = "cuda"
        if self.model_checkpoint == None:
            self.model = EPModel(
                input_span_len=self.input_span_len,
                embedding_dim=self.embedding_dim,
                num_classes=self.num_classes,
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

        self.loss_hist['train'].append(
            self.calc_loss(
                tokenized_dataset=train_tokenized_dataset,
                print_metrics=True
            )
        )
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

                spans = self.prepare_batch_data(
                    tokenized_dataset=train_tokenized_dataset,
                    start_idx=i,
                    end_idx=i + step,
                    pad=True
                )

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

            # self.cached_hidden_states = {
            #     'input_ids': [],
            #     'hidden_states': []
            # }

            gc.collect()

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

                spans_dict = self.prepare_batch_data(
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
        span_rerps_dict = self.extract_embeddings_from_pretrained_model(
            tokenized_dataset=tokenized_dataset,
            start_idx=start_idx,
            end_idx=end_idx,
            pad=pad
        )

        span1 = torch.tensor(
            np.asarray(span_rerps_dict['span1_reprs'])
        ).float().to(self.mlp_device)
        span2 = torch.tensor(
            np.asarray(span_rerps_dict['span2_reprs'])
        ).float().to(self.mlp_device)
        span1_attention_mask = torch.tensor(
            np.asarray(
                span_rerps_dict[
                    'span1_attention_mask'
                ]
            )
        ).float().to(self.mlp_device)
        span2_attention_mask = torch.tensor(
            np.asarray(
                span_rerps_dict[
                    'span2_attention_mask'
                ]
            )
        ).float().to(self.mlp_device)

        one_hot_labels = torch.tensor(
            np.asarray(
                span_rerps_dict['one_hot_label']
            )
        ).float().to(self.mlp_device)

        return {
            'span1': span1,
            'span2': span2,
            'span1_attention_mask': span1_attention_mask,
            'span2_attention_mask': span2_attention_mask,
            'one_hot_labels': one_hot_labels
        }

    def get_pretrained_model_properties(self):
        span_reprs_dict = self.extract_embeddings_from_pretrained_model(
            self.dataset_handler.tokenized_dataset['train'],
            0,
            3,
            pad=True,
            cache=True
        )
        for val in span_reprs_dict['span1_reprs']:
            print(f"span1 shape: {val.shape}")
        span_reps_sample = span_reprs_dict['span1_reprs'][0]
        assert len(span_reps_sample.shape) == 3
        num_layers = span_reps_sample.shape[0]
        span_len = span_reps_sample.shape[1]
        embedding_dim = span_reps_sample.shape[2]
        if self.verbose:
            display(pd.DataFrame(span_reprs_dict).head(4))

        return num_layers, span_len, embedding_dim, len(self.dataset_handler.unique_labels_list)

    def extract_embeddings_from_pretrained_model(
            self,
            tokenized_dataset,
            start_idx,
            end_idx,
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
        span_repr = self.init_span_dict(num_spans, pad)

        for i in range(start_idx, end_idx):
            if self.last_setance_hidden_states['text_id'] != None and tokenized_dataset[
                                                                      i:i + 1
            ]['text_id']==self.last_setance_hidden_states['text_id']:
                idx = self.last_setance_hidden_states[
                    'input_ids'
                ].index(
                    tokenized_dataset[i:i+1][
                        'input_ids'
                    ][0]
                )
                self.current_hidden_states = self.last_setance_hidden_states[
                    'hidden_states'
                ][idx]
            else:
                with torch.no_grad():
                    sample_dict = tokenized_dataset[i:i+1]
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
                        self.last_setance_hidden_states['text_id'] = text_id
                        self.last_setance_hidden_states['input_ids'] = sample_dict['input_ids']
                        self.last_setance_hidden_states['hidden_states'] = current_hidden_states


            row = tokenized_dataset[i]

            span1_hidden_states = self.last_setance_hidden_states['hidden_states'][
                                  :, row['span1_indices'], :
                                  ]
            if pad:
                curr_padded_span_repr, curr_attention_mask = self.pad_span(
                    span1_hidden_states,
                    max_span_len
                )
                span_repr['span1_reprs'].append(
                    curr_padded_span_repr
                )
                span_repr['span1_attention_mask'].append(
                    curr_attention_mask
                )
            else:
                span_repr['span1_reprs'].append(
                    span1_hidden_states
                )

            if num_spans == 2:
                span2_hidden_states = self.last_setance_hidden_states['hidden_states'][
                                      :, row['span2_indices'], :
                                      ]
                if pad:
                    curr_padded_span_repr, curr_attention_mask = self.pad_span(
                        np.array(span2_hidden_states),
                        max_span_len
                    )
                    span_repr['span2_reprs'].append(
                        curr_padded_span_repr
                    )
                    span_repr['span2_attention_mask'].append(
                        curr_attention_mask
                    )
                else:
                    span_repr['span2_reprs'].append(
                        span2_hidden_states
                    )

            span_repr['one_hot_label'].append(
                row['one_hot_label']
            )
            span_repr['label'].append(
                row['label']
            )

        return span_repr

    def init_span_dict(self, num_spans, pad):
        if num_spans == 2:
            span_repr = {
                'span1_reprs': [],
                'span2_reprs': [],
                'label': [],
                'one_hot_label': []
            }
        else:
            span_repr = {
                'span1_reprs': [],
                'label': [],
                'one_hot_label': []
            }

        if pad:
            span_repr['span1_attention_mask'] = []
            span_repr['span2_attention_mask'] = []

        return span_repr

    @staticmethod
    def pad_span(
            span_repr,
            span_max_len
    ):
        num_layers = span_repr.shape[0]
        # print("number of layers: {}".format(num_layers))
        # print("span shape is: {}".format(span_repr.shape))
        span_len = span_repr.shape[1]
        embedding_dim = span_repr.shape[2]
        # padded_span_repr = np.zeros((num_layers, span_max_len, embedding_dim))
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
