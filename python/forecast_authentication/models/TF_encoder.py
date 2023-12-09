import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embbeding(nn.Module):
    # 
    def __init__(self, num_features, d_model=512):
        super(Embbeding, self).__init__()
        # num_features -> each timestamp has many features, i.e. right_hand_x, right_hand_y, right_hand_z, ...
        self.c1 = nn.Conv1d(num_features, d_model, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, Q, K, V, mask=None):

        bs, len_q, len_k, len_v = Q.size(0), Q.size(1), K.size(1), V.size(1)

        residual = Q

        q = self.W_Q(Q).view(bs, len_q, self.n_head, self.d_k).transpose(1,2)  #[bs, n_head, len_q, d_k]
        k = self.W_K(K).view(bs, len_k, self.n_head, self.d_k).transpose(1,2)  #[bs, n_head, len_k, d_k]
        v = self.W_V(V).view(bs, len_v, self.n_head, self.d_v).transpose(1,2)  #[bs, n_head, len_k, d_v]

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        # mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # shape of q: [bs, n_head, len_q, d_v]
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(bs, -1, self.n_head * self.d_v)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=512, d_hid=2048, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hid)
        self.w_2 = nn.Linear(d_hid, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_head=8, d_k=64, d_v=64, d_model=512, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(
        self, 
        n_head=8,
        d_k=64,
        d_v=64,
        seq_len=50,
        num_features=4, 
        d_model=512, 
        d_ff=2048,
        n_layers=6,
        n_class=2,
        dropout=0.1
        ):
        super(Encoder, self).__init__()
        self.src_emb = Embbeding(num_features, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(n_head, d_k, d_v, d_model, d_ff, dropout) for _ in range(n_layers)
            ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.fc = nn.Linear(d_model*seq_len, n_class)   # for classification

    def forward(self, enc_input, mask=None):
        # size of enc_input is [bs, seq_len, num_features]
        enc_output = self.src_emb(enc_input)    # [bs, seq_len, d_model]
        enc_output = self.pos_emb(enc_output.transpose(0, 1)).transpose(0, 1)   # [bs, seq_len, d_model]
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        enc_attns = []
        for layer in self.layers:
            enc_output, enc_self_attn = layer(enc_output, mask)
            enc_attns.append(enc_self_attn)

        enc_output = self.fc(enc_output.view(enc_output.size()[0], -1))    # for classification

        return enc_output, enc_attns


