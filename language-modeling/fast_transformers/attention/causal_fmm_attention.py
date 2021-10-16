"""Implement causally masked FMM attention."""

import torch
import torch.nn as nn
from torch.nn import Module, Dropout
from math import sqrt

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, Float, \
    Bool, EventDispatcherInstance
from ..events import EventDispatcher, AttentionEvent
from ..causal_product import causal_dot_product
from ..feature_maps import tanh_feature_map, elu_feature_map, elu2_feature_map

def causal_linear(Q, K, V):
    Q = Q.permute(0,2,1,3).contiguous()
    K = K.permute(0,2,1,3).contiguous()
    V = V.permute(0,2,1,3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0,2,1,3).contiguous()

class CausalFMMAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        bandwidth: The size of the diagonal vector in the near field component
                        (default: 10)
        kernels: The number of unique feature maps to be used in the
             far field component (default: 1)
        sparse: Bool, whether to include near field in the FMM transformer
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, eps=1e-6,
                  softmax_temp=None, attention_dropout=0.1,
                  bandwidth=10, kernels=1, sparse=False, event_dispatcher="",):

        super(CausalFMMAttention, self).__init__()

        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

        # kernel functions
        self.kernels = kernels
        if self.kernels > 0:
            self.W2 = nn.Parameter(torch.ones([1, 1, 8, 32]) / self.kernels)
            self.feature_map1 = elu_feature_map(query_dimensions)
        if self.kernels > 1:
            self.W3 = nn.Parameter(torch.ones([1, 1, 8, 32]) / self.kernels)
            self.feature_map2 = elu2_feature_map(query_dimensions)
        if self.kernels > 2:
            self.W4 = nn.Parameter(torch.ones([1, 1, 8, 32]) / self.kernels)
            self.feature_map3 = tanh_feature_map(query_dimensions)

        # Sparse components
        self.sparse = sparse
        if self.sparse:
            self.softmax_temp = softmax_temp
            self.dropout = Dropout(attention_dropout)
            self.event_dispatcher = EventDispatcher.get(event_dispatcher)
            self.bandwidth = bandwidth
            # Sparse weights
            self.W1 = nn.Parameter(torch.zeros([1, 1, 8, 32]))

    def _make_sizes_compatible(self, Q, K):
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K

        if L < S:
            return Q, K[:, :L, :, :]

        if L > S:
            return Q, torch.cat([K, K.new_zeros(N, L-S, H, E)], dim=1)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):

        if self.kernels > 0:
            # Apply the key padding mask and make sure the attn_mask is a
            # lower triangular causal mask
            if not attn_mask.lower_triangular:
                raise RuntimeError(("CausalLinearAttention only supports full "
                                    "lower triangular masks"))

            # Apply the feature map to the queries and keys
            self.feature_map1.new_feature_map(queries.device)
            Q = self.feature_map1.forward_queries(queries)
            K = self.feature_map1.forward_keys(keys)

            K = K * key_lengths.float_matrix[:, :, None, None]
            # Ensure that Q and K have compatible sizes for the following
            # computations, namely L == S
            Q, K = self._make_sizes_compatible(Q, K)

            # Normalizing
            Q = Q / Q.sum(-1, keepdim=True)
            K = K / K.sum(-1, keepdim=True)

            # Compute the normalizers
            Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)

            # Compute the unnormalized result
            V = causal_linear(
                Q,
                K,
                values
            )

            LV = self.W2 * (V * Z[:, :, :, None])

        if self.kernels > 1:
            self.feature_map2.new_feature_map(queries.device)
            Q = self.feature_map2.forward_queries(queries)
            K = self.feature_map2.forward_keys(keys)
            K = K * key_lengths.float_matrix[:, :, None, None]
            Q, K = self._make_sizes_compatible(Q, K)
            Q = Q / Q.sum(-1, keepdim=True)
            K = K / K.sum(-1, keepdim=True)
            Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
            V = causal_linear(
                Q,
                K,
                values
            )
            LV += self.W3 * (V * Z[:, :, :, None])

        if self.kernels > 2:
            self.feature_map3.new_feature_map(queries.device)
            Q = self.feature_map3.forward_queries(queries)
            K = self.feature_map3.forward_keys(keys)
            K = K * key_lengths.float_matrix[:, :, None, None]
            Q, K = self._make_sizes_compatible(Q, K)
            Q = Q / Q.sum(-1, keepdim=True)
            K = K / K.sum(-1, keepdim=True)
            Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
            V = causal_linear(
                Q,
                K,
                values
            )
            LV += self.W4 * (V * Z[:, :, :, None])

        if self.sparse:
            # Extract some shapes and compute the temperature
            N, L, H, E = queries.shape
            _, S, _, D = values.shape
            softmax_temp = self.softmax_temp or 1./sqrt(E)

            # Scale the queries instead of applying the softmax temperature to the
            # dot products
            queries = queries * softmax_temp

            # Compute the unnormalized attention and apply the masks
            QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
            if not attn_mask.all_ones:
                QK = QK + attn_mask.additive_matrix
            if not key_lengths.all_ones:
                QK = QK + key_lengths.additive_matrix[:, None, None]

            # sparse masking
            b, h, q, k = QK.shape
            sparse_mask = torch.ones(q, k).to(QK).to(torch.bool)
            sparse_mask = torch.tril(sparse_mask, diagonal=-self.bandwidth)
            QK.masked_fill_(sparse_mask[None, None, :, :], -float('inf'))
            A = self.dropout(torch.softmax(QK, dim=-1))
            
            # Let the world know of the attention matrix
            self.event_dispatcher.dispatch(AttentionEvent(self, A))

            # Make sure that what we return is contiguous
            SV = torch.einsum("nhls,nshd->nlhd", A, values)
        
            if self.kernels > 0:
                return (self.W1 * SV.contiguous()) + LV
            else:
                return (self.W1 * SV.contiguous())
        else:
            return LV

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "FMM", CausalFMMAttention,
    [  
        ("query_dimensions", Int),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("bandwidth", Optional(Int)),
        ("kernels", Optional(Int)),
        ("sparse", Optional(Bool, False))
    ]
)
