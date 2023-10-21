import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

def reverse_layer_f(input_, alpha):
    return input_

def mlp(input_dim, embed_dims, dropout, output_layer=True):
    layers = []
    for embed_dim in embed_dims:
        layers.append(nn.Linear(input_dim, embed_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        input_dim = embed_dim
    if output_layer:
        layers.append(nn.Linear(input_dim, 1)
    return nn.Sequential(*layers)

def cnn_extractor(feature_kernel, input_size):
    convs = [nn.Conv1d(input_size, feature_num, kernel) for kernel, feature_num in feature_kernel.items()]
    
    def forward(input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in convs]
        feature = [F.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1])
        return feature
    
    return forward

def mask_attention(input_shape):
    attention_layer = nn.Linear(input_shape, 1)
    
    def forward(inputs, mask=None):
        scores = attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores

def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn