import torch

import torch.nn as nn

import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward, activation):
        super().__init__()
        
        self.EncoderLayer = EncoderLayer(d_model, nhead, dropout, dim_feedforward, activation)
        
        self.transformerencoder = TransformerEncoder()
        
        
class TransformerEncoder(nn.Module):
    def __init__(self, encoderlayer, num_layers):
        self.layers = _get_repeat(encoder_layer, num_layers)
        self.num_layers = num_layers
        
    def forward(self):
        for layer in self.layers:
            output = layer()
            
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward, activation):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)
        
        
    def forward(self, x):
        src1 = self.norm1(x)
        q = k = v = src1
        src2 = self.attention(q,k,v, attn_mask=None, key_padding_mask = None)[0]
        src2 += x
        
        src3 = self.norm2(src2)
        src3 = self.linear1(src3)
        src3 = self.activation(src3)
        src3 = self.linear2(src3)
        src3 + src2
        
        return src3


def _get_repeat(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not{activation}.')
        

        