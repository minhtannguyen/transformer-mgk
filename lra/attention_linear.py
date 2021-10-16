import torch
import torch.nn as nn
import math
from attention_performer import PerformerAttention
import pdb
 
class LinearAttention(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.key2 = config.key2
        self.multi_gauss = config.multi_gauss
        self.head_dim = config.head_dim
        self.pi = None
        self.add_performer = config.add_performer
        if self.add_performer:
            self.performer = PerformerAttention(config)
        self.drop_attn = torch.nn.Dropout(p=config.attention_dropout)
 
        if self.key2 or self.multi_gauss:
            self.pi = nn.Parameter(torch.tensor([0.5, 0.5]),requires_grad= True)
#             pi0 = torch.FloatTensor(config.num_head, config.max_seq_len).uniform_(config.pi0, 1.)
#             self.pi = nn.Parameter(torch.cat([pi0, 1-pi0], dim = 0).view(2, config.num_head, config.max_seq_len), requires_grad= True)
     
 
        if self.multi_gauss:
            self.mu = nn.Parameter((torch.empty( 2, config.num_head, config.head_dim).normal_(mean = 0.0, std = .5))*torch.tensor([0.,1.])[:, None, None], requires_grad= True)
        
    def forward(self, Q, K, V, mask):
 
        l = Q.shape[-2]
        
        if not (self.key2 or self.multi_gauss):
            X = self.linear_attn(Q, K, V, mask)
#             dot = self.dot_(Q, K, mask)
#             attn = nn.functional.softmax(dot, dim = -1)
        else:
            if self.key2:
                K1 = K[:, :, :, : self.head_dim]
                K2 = K[:, :, :, self.head_dim :]
 
            elif self.multi_gauss:
                K1 = K - self.mu[0][None, :, None, :]
                K2 = K - self.mu[1][None, :, None, :]
           
            K = K1*(torch.clamp(self.pi, min=0., max = 1.)[0]) + K2*(torch.clamp(self.pi, min=0., max = 1.)[1])
            X = self.linear_attn(Q,K,V,mask)
#             dot = self.dot_(Q, K1, mask)*torch.clamp(self.pi, min = 0., max = 1.)[0][None, :, None, :l] + self.dot_(Q, K2, mask)*(torch.clamp(self.pi, min=0., max = 1.)[1][None, :, None, :l])
#             attn = nn.functional.softmax(dot, dim = -1)
        
#         self.attn_matrix = attn
#         X = torch.matmul(self.drop_attn(attn), V)
        
 
        return X
    
    def linear_attn(self, Q, K, V, mask, scale = 1.):
        Q = (nn.functional.elu(Q) + 1) / math.sqrt(math.sqrt(Q.size(2)))
        K = (nn.functional.elu(K) + 1) * mask[:, None, :, None] / math.sqrt(math.sqrt(K.size(2)))
        V = V * mask[:, None, :, None]

        X = torch.matmul(Q, torch.matmul(torch.transpose(K, -2, -1), V))

        return X
    
    def dot_(self, Q, K, mask, scale = 1.):
        Q = (nn.functional.elu(Q) + 1) 
        K = (nn.functional.elu(K) + 1) 
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
#         print(mask[:, None, None, :].shape,dot.shape )
        dot = dot - 1e6 * (1 - mask[:, None, None, :])
        return dot