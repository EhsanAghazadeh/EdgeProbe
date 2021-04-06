import torch
import torch.nn as nn
import torch.optim as optim

from src.spanRepClasses import get_span_module


# class CorefModel(nn.Module):
#     def __init__(
#             self,
#             input_dim=768,
#             output_dim=1,
#             span_dim=256,
#             all_layers=True,
#             num_layers=13,
#             drop_out_param=0.2
#     ):
#         super(CorefModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.span_dim = span_dim
#         self.all_layers = all_layers
#         self.num_layers = num_layers
#         self.drop_out_param = drop_out_param
#
#         if all_layers:
#             self.weights = nn.Parameter(torch.ones(self.num_layers))
#
#         self.net = nn.Sequential(
#             nn.Linear(2*self.input_dim, self.span_dim),
#             nn.Tanh(),
#             nn.LayerNorm(self.span_dim),
#             nn.Dropout(self.drop_out_param),
#             nn.Linear(self.span_dim, self.output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, span1, span2):
#         embedding1 = torch.zeros(size=(self.input_dim, ))
#         embedding2 = torch.zeros(size=(self.input_dim, ))
#         if self.all_layers:
#             soft_weights = nn.functional.softmax(self.weights, dim=0)
#             for layer in range(self.num_layers):
#                 embedding1 += soft_weights[layer] * span1[layer]
#                 embedding2 += soft_weights[layer] * span2[layer]
#         else:
#             embedding1 = span1
#             embedding2 = span2
#
#         net_input = torch.cat(
#             [embedding1, embedding2],
#             dim=0
#         )
#
#         return self.net(net_input)


class EPModel(nn.Module):
    def __init__(
            self,
            input_span_len,
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
        self.input_span_len = input_span_len
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
            method=pool_method
        )
        self.span2_pooling_net = get_span_module(
            input_dim=embedding_dim,
            method=pool_method
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
        span1_reprs = spans_dict['span1']
        span2_reprs = spans_dict['span2']
        span1_attention_mask = spans_dict[
            'span1_attention_mask'
        ]
        span2_attention_mask = spans_dict[
            'span2_attention_mask'
        ]

        if self.use_proj:
            span1_reprs = self.proj1_net(span1_reprs)
            span2_reprs = self.proj2_net(span2_reprs)

        pooled_span1_reprs = self.span1_pooling_net(
            span1_reprs, span1_attention_mask
        )
        pooled_span2_reprs = self.span2_pooling_net(
            span2_reprs, span2_attention_mask
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