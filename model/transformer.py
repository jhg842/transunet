import torch

import torch.nn as nn

import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.EncoderLayer = EncoderLayer()
        
        self.transformerencoder = TransformerEncoder()
        
        
class TransformerEncoder(nn.Module):
    def __init__(self,)
    
    
    def forward(self):
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward, activation):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.activation = _get_activation_fn(activation)
        self.MLP = nn.Sequential(self.linear1, self.linear2)
        
    def forward(self, x):
        src1 = self.norm1(x)
        q = k = v = x
        src2 = self.attention(q,k,v, attn_mask=None, key_padding_mask = None)[0]
        src2 += x
        
        src3 = self.norm2(src2)
        src3 = self.MLP(src3)
        src3 + src2
        
        return src3
    

        
        
        