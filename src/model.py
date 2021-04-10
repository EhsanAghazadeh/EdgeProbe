import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from src.spanRepClasses import get_span_module


class EPModel(nn.Module):
    def __init__(
            self,
            max_span_len,
            embedding_dim,
            num_classes,
            pool_method,
            use_proj=True,
            proj_dim = 256,
            hidden_dim = 256,
            num_layers=13,
            device='cuda',
            p=0.2,
            criterion=nn.BCELoss,
            optimizer=optim.Adam,
            lr=5e-4
    ):
        super(EPModel, self).__init__()
        self.max_span_len = max_span_len
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pool_method = pool_method
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weights = nn.Parameter(
            torch.ones(
                self.num_layers
            )
        )
        self.device = device

        if use_proj:
            self.proj1_net = nn.Linear(
                embedding_dim,
                proj_dim
            )
            self.proj2_net = nn.Linear(
                embedding_dim,
                proj_dim
            )
            self.input_dim = 2 * proj_dim
        else:
            self.input_dim = 2 * embedding_dim

        self.span1_pooling_net = get_span_module(
            input_dim=embedding_dim,
            method=pool_method,
            max_span_len=self.max_span_len
        )
        self.span2_pooling_net = get_span_module(
            input_dim=embedding_dim,
            method=pool_method,
            max_span_len=self.max_span_len
        )

        self.label_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(p),
            nn.Linear(self.hidden_dim, self.num_classes),
            nn.Sigmoid()
        )

        self.criterion = criterion()
        self.optimizer = optimizer(
            params=self.parameters(),
            lr=lr,
            weight_decay=0
        )

    def forward(self, spans_dict):
        encodings = spans_dict['encoded_repr']

        if self.use_proj:
            span1_reprs = self.proj1_net(deepcopy(encodings))
            span2_reprs = self.proj2_net(deepcopy(encodings))

        pooled_span1_reprs = self.span1_pooling_net(
            span1_reprs,
            spans_dict['span1_start_ids'],
            spans_dict['span1_end_ids']
        )
        pooled_span2_reprs = self.span2_pooling_net(
            span2_reprs,
            spans_dict['span2_start_ids'],
            spans_dict['span2_end_ids']
        )
        concatenated_reprs = torch.cat(
            [
                pooled_span1_reprs,
                pooled_span2_reprs
            ],
            dim=-1
        )

        wtd_encoded_repr = 0
        soft_weights = nn.functional.softmax(
            self.weights,
            dim=0
        )
        for i in range(self.num_layers):
            wtd_encoded_repr += soft_weights[i] * concatenated_reprs[:, i, :]

        net_input = wtd_encoded_repr
        pred = self.label_net(net_input)
        pred = torch.squeeze(pred, dim=-1)
        return pred