# Fast weight layers using custom kernels.
# Many code duplications to be refactored!

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.fast_fast_weight import fast_weight_memory
from utils.fast_transformers import fast_weight_sum
from utils.performer_helper import prime, draw_orthogonal_random_matrix


# Linear Transformer version
# our update rule + Katharopoulos et al's ELU based attention
class CudaFastWeightLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True):
        # skip_attn_normalization is now set to True by default, thus it can
        # be removed.
        # Originally, with skip_attn_normalization set to False,
        # we had a version of the model which applies attention normalization
        # to the output (but not when we retrieve with the key for removal).
        super(CudaFastWeightLinearTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output

# Linear Transformer version
# our update rule + Katharopoulos et al's ELU based attention
# with attention normalization
class CudaNormFastWeightLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False):
        super(CudaNormFastWeightLinearTransformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.d_head],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom 
                head_k = head_k / (key_denom + self.eps)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.d_head],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Performer version, our update rule + FAVOR+
class CudaFastWeightPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=True,
                 proj_dim=256, device='cuda'):
        super(CudaFastWeightPerformerLayer, self).__init__()
        print(f"Using CudaFastWeightPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Performer version, our update rule + FAVOR+
# with attention normalization
class CudaNormFastWeightPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=False,
                 proj_dim=256, device='cuda'):
        super(CudaNormFastWeightPerformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.proj_dim * 2],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros([bsz, self.n_head, 1, self.proj_dim * 2],
                                 device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Katharopoulos et al's Linear Transformer https://arxiv.org/abs/2006.16236
# = Sum update rule + ELU based attention function
class CudaFastWeightSumLinearTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False):
        super(CudaFastWeightSumLinearTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightSumLinearTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkv_net = nn.Linear(
            d_model, n_head * 3 * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 3 * self.d_head)
        head_q, head_k, head_v = torch.split(
            qkv, (self.d_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.
        
#         head_q = F.tanh(head_q) + 1.
#         head_k = F.tanh(head_k) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_sum(
            head_q, head_k, head_v, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]
        
        
        #######

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output
        
# Performer https://arxiv.org/abs/2009.14794
# = Sum update rule + FAVOR+
class CudaFastWeightSumPerformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, skip_attn_normalization=False,
                 proj_dim=256, device='cuda'):
        super(CudaFastWeightSumPerformerLayer, self).__init__()
        print(f"Using CudaFastWeightSumPerformerLayer - "
              f"proj_dim: {proj_dim}")

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkv_net = nn.Linear(
            d_model, n_head * 3 * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        self.proj_dim = proj_dim
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkv = self.qkv_net(h)
        qkv = qkv.view(slen, bsz, self.n_head, 3 * self.d_head)
        head_q, head_k, head_v = torch.split(
            qkv, (self.d_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        head_q = prime(head_q, self.proj_matrix)  # (B, n_head, len, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.proj_dim, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_sum(
            head_q, head_k, head_v, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Our update rule + DPFP
class CudaFastWeightDPFPTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, n_roll=2):
        super(CudaFastWeightDPFPTransformerLayer, self).__init__()
        print(f"Using CudaFastWeightDPFPTransformerLayer roll {n_roll} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_roll = n_roll

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.n_roll * self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output


# Our update rule + DPFP, with attention normalization
class CudaNormFastWeightDPFPTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=False, n_roll=2):
        super(CudaNormFastWeightDPFPTransformerLayer, self).__init__()
        print(f"Using CudaNormFastWeightDPFPTransformerLayer roll {n_roll} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_roll = n_roll

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # (3 * d_head * n_head) for qkv and (1 * n_head) for beta.
        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, 2 * self.n_roll * self.d_head, self.d_head,
                device=head_k.device)
            if self.normalize_attn_scores:
                # key_denom = z(i-1) * key(i) and 1 if i=1
                # z(i) = denominator_acc
                key_denom = torch.cat(
                    [torch.zeros(
                        [bsz, self.n_head, 1, 2 * self.n_roll * self.d_head],
                        device=head_q.device),
                     denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = torch.einsum('lbij,lbij->lbi', key_denom, head_k)
                key_denom = torch.cat(
                    [torch.ones([bsz, self.n_head, 1], device=head_q.device),
                     key_denom[:, :, 1:].clone()], dim=2).unsqueeze(-1)
                head_beta = head_beta * key_denom 
                head_k = head_k / (key_denom + self.eps)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                key_denom = torch.cat(
                    [torch.zeros(
                        [bsz, self.n_head, 1, 2 * self.n_roll * self.d_head],
                        device=head_q.device),
                        denominator_acc[:, :, :-1, :].clone()], dim=2)
                key_denom = key_denom + fast_denom[:bsz]
                denominator_acc = denominator_acc + fast_denom[:bsz]

                key_denom = torch.einsum(
                    'lbij,lbij->lbi', key_denom, head_k).unsqueeze(-1)
                head_beta = head_beta * key_denom
                head_k = head_k / (key_denom + self.eps)

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_memory(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # print(mem_fast_weights.norm())
        # print(denominator_acc[:, :, -1, :].norm())

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output
