from abc import ABC, abstractmethod
import torch.nn as nn
import torch


def get_span_mask(start_ids, end_ids, max_len):
    tmp = torch.arange(max_len).unsqueeze(0).expand(start_ids.shape[0], -1)
    batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
    batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)
    if torch.cuda.is_available():
        tmp = tmp.cuda()
        batch_start_ids = batch_start_ids.cuda()
        batch_end_ids = batch_end_ids.cuda()
    mask = ((tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float()).unsqueeze(2)
    return mask


class SpanRepr(ABC, nn.Module):
    """Abstract class describing span representation."""

    def __init__(
            self,
            input_dim,
            max_span_len,
            use_proj=False,
            proj_dim=256
    ):
        super(SpanRepr, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.use_proj = use_proj
        self.max_span_len = max_span_len

    @abstractmethod
    def forward(self, encoded_input, start_ids, end_ids):
        raise NotImplementedError

    def get_input_dim(self):
        return self.input_dim

    @abstractmethod
    def get_output_dim(self):
        raise NotImplementedError


class AvgSpanRepr(SpanRepr, nn.Module):
    ...


class MaxSpanRepr(SpanRepr, nn.Module):
    """Class implementing the max-pool span representation."""

    def forward(self, encoded_input, start_ids, end_ids):
        if self.use_proj:
            encoded_input = self.proj(encoded_input)
        span_masks = get_span_mask(start_ids, end_ids, encoded_input.shape[2])
        # put -inf to irrelevant positions
        span_masks_shape = span_masks.shape
        span_masks = span_masks.reshape(
            span_masks_shape[0],
            1,
            span_masks_shape[1],
            span_masks_shape[2]
        ).expand_as(encoded_input)
        tmp_repr = encoded_input * span_masks - 1e10 * (1 - span_masks)
        span_repr = torch.max(tmp_repr, dim=2)[0]
        return span_repr

    def get_output_dim(self):
        if self.use_proj:
            return self.proj_dim
        else:
            return self.input_dim


class DiffSpanRepr(SpanRepr, nn.Module):
    ...


class DiffSumSpanRepr(SpanRepr, nn.Module):
    ...


class EndPointRepr(SpanRepr, nn.Module):
    ...


class CoherentSpanRepr(SpanRepr, nn.Module):
    ...


class CoherentOrigSpanRepr(SpanRepr, nn.Module):
    ...


class AttnSpanRepr(SpanRepr, nn.Module):
    ...


def get_span_module(
        input_dim,
        max_span_len,
        method="max",
        use_proj=False,
        proj_dim=256
):
    """Initializes the appropriate span representation class and returns the object.
    """
    if method == "avg":
        return AvgSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "max":
        return MaxSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff":
        return DiffSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "diff_sum":
        return DiffSumSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "endpoint":
        return EndPointRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent":
        return CoherentSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coherent_original":
        return CoherentOrigSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "attn":
        return AttnSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim)
    elif method == "coref":
        return AttnSpanRepr(input_dim, max_span_len, use_proj=use_proj, proj_dim=proj_dim, use_endpoints=True)
    else:
        raise NotImplementedError
