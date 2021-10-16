import torch
import torch.nn as nn
import math
import json
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint
import pdb
from gpytorch.kernels.kernel import Distance
from copy import deepcopy
 

class MultiGaussKernel(nn.Module):
    def __init__(self, config):
        super(MultiGaussKernel, self).__init__()
        self.var = config.head_dim**0.5
        self.mu = nn.Parameter((torch.empty( 2, config.num_head, config.head_dim).normal_(mean = 0.0, std = .5))*torch.tensor([0.,1.])[:, None, None], requires_grad= True)
        self.hard_em = config.hard_em
        self.soft_em = config.soft_em
 
        pi0 = torch.FloatTensor(config.num_head, config.max_seq_len).uniform_(config.pi0, 1.)
        if self.hard_em:
            self.pi = None
        elif self.soft_em:
            # self.pi = nn.Parameter(0.5*torch.ones(config.num_head, config.max_seq_len), requires_grad= False)
            self.register_buffer('pi',0.5*torch.ones(config.num_head, config.max_seq_len, requires_grad= False))
            # self.register_buffer('pi', torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, 1024))
            # self.pi = torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, 1024).to('cuda')
        else:
            # self.pi = nn.Parameter(0.5*torch.ones(2, config.num_head, config.max_seq_len), requires_grad= True)
#             self.pi = nn.Parameter(torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, config.max_seq_len), requires_grad= True)
             self.pi = nn.Parameter(torch.tensor([0.5, 0.5]),requires_grad= True)
        self.split_attn_type = config.split_attn_type
        self.head_dim = config.head_dim
        self.norm = config.norm_qk
        self.dist = Distance()
        # pdb.set_trace()
        
    def forward(self, query, key, mask):
        l = query.shape[-2]
        
        
        if self.split_attn_type:
            attn = 0.
            dot = torch.matmul(query, torch.transpose(key, -2, -1))
            dot = dot / math.sqrt(self.head_dim)
            dot = dot - 1e6 * (1 - mask[:, None, None, :])
            attn = nn.functional.softmax(dot, dim = -1)*self.pi[0][None, :, :l, None]
 
            if self.norm:
                query = query/torch.norm(query, p = 2, keepdim= True, dim = -1)
                key = key/torch.norm(key, p = 2, keepdim= True, dim = -1)
            QK_distance = (-1/(2*self.var))*self.dist._sq_dist(query, key - self.mu[-1][None, :, None, :], postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
            attn = attn + nn.functional.softmax(QK_distance, dim = -1)*self.pi[-1][None, :, :l, None]
 
        else:
            
            if self.hard_em:
                QK1_distance = self.dist._sq_dist(query, key - self.mu[0][None, :, None, :], postprocess = False)
                QK2_distance = self.dist._sq_dist(query, key - self.mu[1][None, :, None, :], postprocess = False)
                dist_min = torch.minimum(QK1_distance, QK2_distance)
                attn = nn.functional.softmax((-1/(2*self.var))*dist_min - 1e6 * (1 - mask[:, None, None, :]), dim = -1)
                # pdb.set
                
 
                ### check every step of this, consider numerical issue
            elif self.soft_em:
                max_l = (mask.sum(0)!=0.).sum()
                pi = self.pi.clone().detach()
                QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(query, key - self.mu[0][None, :, None, :], postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(query, key - self.mu[1][None, :, None, :], postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
            
 
                attn = torch.exp(QK1_distance)*torch.clamp(pi, min = 0., max = 1.)[None, :, None, :l] + (1 - torch.clamp(pi, min = 0., max = 1.)[None, :, None, :l])*torch.exp(QK2_distance)
                
                
                if self.training:
                #update self.pi, using mask
                    
                    N1 = torch.einsum('nl,nhlk->hk',mask,(torch.exp(QK1_distance)*pi[None, :, None, :l])/(attn + 1e-6))
                    # N2 = torch.einsum('nhlk->hk',torch.exp(QK2_distance)*pi[None, :, :l, None]/(attn + 1e-6))
                    N = torch.einsum('ln,nk->k', mask.T, mask)[None, :] + 1e-6
                    
                    #(h,l)
                    pi_new = self.pi.clone().detach()
                    pi_new[:, :max_l] = (N1/N).detach()[:,:max_l]
                    pi_new.to(query)
 
                    # print(N1, 'hihi')
 
                    self.pi.copy_(pi_new.detach())
 
                # pdb.set_trace()
                attn = attn/(attn.sum(dim = -1)[:, :, :, None])
             
            else: 
                #check if self.pi really 
                # print(self.pi)  
                QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(query, key - self.mu[0][None, :, None, :], postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(query, key - self.mu[1][None, :, None, :], postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                
#                 attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0][None, :, None, :l] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1][None, :, None, :l]
                attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1]
                attn = attn/(attn.sum(dim = -1)[:, :, :, None])
 

 
        return attn

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
        self.head_dim = config.head_dim
        self.multi_gauss = config.multi_gauss
        
 
        self.save_attn = config.save_attn
        self.norm = config.norm_qk
        self.key2 = config.key2
        self.track_kk = config.track_kk
        if self.key2:
            
            # self.pi = torch.tensor([config.pi0, 1. - config.pi0])
            self.var = config.head_dim**0.5
            pi0 = torch.FloatTensor(config.num_head, config.max_seq_len).uniform_(config.pi0, 1.)
            self.hard_em = config.hard_em
            self.soft_em = config.soft_em
 
            if self.hard_em:
                self.pi = None
            elif self.soft_em:
                ## be very careful with to.('cuda')
                self.register_buffer('pi',0.5*torch.ones(config.num_head, config.max_seq_len, requires_grad= False))
                if self.track_kk:
                    self.register_buffer('kk_distance',torch.tensor(0., requires_grad = False))
                # self.pi = nn.Parameter(0.5*torch.ones(config.num_head, config.max_seq_len), requires_grad= False)
                # self.pi = torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, config.max_seq_len)
            else:
                self.pi = nn.Parameter(torch.tensor([0.5, 0.5]),requires_grad= True)
                # self.pi = nn.Parameter(0.5*torch.ones(2, config.num_head, config.max_seq_len), requires_grad= True)
#                 self.pi = nn.Parameter(torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, config.max_seq_len), requires_grad= True)
                if self.track_kk:
                    self.register_buffer('kk_distance',torch.tensor(0., requires_grad = False))
 
            self.dist = Distance()
 
        elif self.multi_gauss:
            self.multi_gauss_kernel = MultiGaussKernel(config)
            self.pi = self.multi_gauss_kernel.pi
        else:
            self.pi = None
 
#         print(self.norm)
 
    def forward(self, Q, K, V, mask):
        l = Q.shape[-2]
        if self.multi_gauss:
            assert not self.key2
            attn = self.multi_gauss_kernel(Q, K, mask)
        else:
            # if self.norm:
            #     Q = Q/torch.norm(Q, p = 2, keepdim= True, dim = -1)
            #     K = K/torch.norm(K, p = 2, keepdim= True, dim = -1)
            if not self.key2:
                dot = torch.matmul(Q, torch.transpose(K, -2, -1))
                dot = dot / math.sqrt(self.head_dim)
                dot = dot - 1e6 * (1 - mask[:, None, None, :])
                attn = nn.functional.softmax(dot, dim = -1)
                # pdb.set_trace()
            else:
                
                K1 = K[:, :, :, : self.head_dim]
                K2 = K[:, :, :, self.head_dim :]
 
                #### check every step of this, consider numerical issue of dist_min = torch.tensor(1e6)
                if self.hard_em:
                    QK1_distance = self.dist._sq_dist(Q, K1, postprocess = False)
                    QK2_distance = self.dist._sq_dist(Q, K2, postprocess = False)
                    dist_min = torch.minimum(QK1_distance, QK2_distance)
                    attn = nn.functional.softmax((-1/(2*self.var))*dist_min - 1e6 * (1 - mask[:, None, None, :]), dim = -1)
                    # pdb.set_trace()
                ### check every step of this, consider numerical issue
                elif self.soft_em:
                    max_l = (mask.sum(0)!=0.).sum()
                    pi = self.pi.clone().detach()
                    QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K1, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K2, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                
 
                    attn = torch.exp(QK1_distance)*pi[None, :, None,:l] + (1 - pi[None, :, None, :l])*torch.exp(QK2_distance)
                    
                    
                    if self.training:
                    #update self.pi, using mask
                        
                        N1 = torch.einsum('nl,nhlk->hk',mask,(torch.exp(QK1_distance)*pi[None, :, None, :l])/(attn + 1e-6))
                        # N2 = torch.einsum('nhlk->hk',torch.exp(QK2_distance)*pi[None, :, :l, None]/(attn + 1e-6))
                        N = torch.einsum('ln,nk->k', mask.T, mask)[None, :] + 1e-6
                        
                        #(h,l)
                        pi_new = self.pi.clone().detach()
                        pi_new[:, :max_l] = (N1/N).detach()[:,:max_l]
                        pi_new.to(Q)
                        # print(N1, 'hihi')
 
                        self.pi.copy_(pi_new.detach())
 
                    
                    attn = attn/(attn.sum(dim = -1)[:, :, :, None])
                    if self.track_kk:
                        kk_distance = self.dist._sq_dist(K1, K2, postprocess = False).detach().mean()
                        kk_distance.to(Q)
                        self.kk_distance.copy_(kk_distance)
                    # pdb.set_trace()
 
                else: 
                    ###find a suitable way to make self.pi learnable here
                    QK1_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K1, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    QK2_distance = (-1/(2*self.var))*self.dist._sq_dist(Q, K2, postprocess = False) - 1e6 * (1 - mask[:, None, None, :])
                    
#                     attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0][None, :, None, :l] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1][None, :, None, :l]
                    attn = torch.exp(QK1_distance)*torch.clamp(self.pi, min = 0., max = 1.)[0] + torch.exp(QK2_distance)*torch.clamp(self.pi, min = 0., max = 1.)[1]
                    attn = attn/(attn.sum(dim = -1)[:, :, :, None])
 
                    if self.track_kk:
                        kk_distance = self.dist._sq_dist(K1, K2, postprocess = False).detach().mean()
                        kk_distance.to(Q)
#                         self.kk_distance.copy_(kk_distance)
                  
                    # pdb.set_trace()
#         self.kk_distance = kk_distance.mean()
#         self.attn_matrix = attn
        X = torch.matmul(self.drop_attn(attn), V)
        return X

class NoneAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
 
    def forward(self, Q, K, V, mask):
        return V
 
 
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
 
        self.grad_checkpointing = config.attention_grad_checkpointing
 
        self.dim = config.transformer_dim
        self.head_dim = config.head_dim
        self.num_head = config.num_head
 
        self.attn_type = config.attn_type
 
        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
 
        self.dconv_fc = None
        self.key2 = config.key2
 
        if self.attn_type == "softmax":
            self.attn = SoftmaxAttention(config)
        elif self.attn_type == "none":
            self.attn = NoneAttention(config)
        elif self.attn_type.startswith("linformer"):
            from attention_linformer import LinformerAttention
            self.attn = LinformerAttention(config)
 
        elif self.attn_type.startswith("reformer"):
            from attention_reformer import LSHAttention
            self.attn = LSHAttention(config, self.W_q, self.W_k, self.W_v)
        elif self.attn_type.startswith("nystrom"):
            from attention_nystrom import NystromAttention
            self.attn = NystromAttention(config)
        elif self.attn_type.startswith("performer"):
            from attention_performer import PerformerAttention
            self.attn = PerformerAttention(config)
        elif self.attn_type.startswith("linear"):
            from attention_linear import LinearAttention
            self.attn = LinearAttention(config)
        elif self.attn_type.startswith("two_performer"):
            from attention_performer import TwoPerformerAttention
            self.attn = TwoPerformerAttention(config)
        
        if ((self.attn_type in ["softmax", "linear"]) and self.key2):
            self.key2 = config.key2
            self.W_k2 = nn.Linear(self.dim, self.num_head * self.head_dim)
 
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)
        
 
    def forward(self, X, mask):
      
        if self.attn_type.startswith("longformer") or self.attn_type.startswith("reformer"):
            with torch.cuda.amp.autocast(enabled = False):
                attn_out = self.attn(X.float(), mask.float())
        else:
            Q = self.split_heads(self.W_q(X))
            K = self.split_heads(self.W_k(X))
            if self.key2:
                K2 = self.split_heads(self.W_k2(X))
                ## im runing in paralell here
                K = torch.cat([K, K2], dim = -1) ## (B, H, L, 2E)
            V = self.split_heads(self.W_v(X))
            # pdb.set_trace()
 
            with torch.cuda.amp.autocast(enabled = False):
                if self.grad_checkpointing:
                    attn_out = checkpoint(self.attn, Q.float(), K.float(), V.float(), mask.float())
                else:
                    attn_out = self.attn(Q.float(), K.float(), V.float(), mask.float())
            attn_out = self.combine_heads(attn_out)
 
        out = self.ff(attn_out)
 
        return out
 
 
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X
 
    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
 

