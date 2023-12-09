import torch
import torch.nn as nn
import torch.nn.functional as F

from models.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.informer import Informer

from models.FCN import FCN as FCN_model
from models.TF_encoder import Encoder as tf_endoder


class Fore_Auth_Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
        num_feats=4, num_class=2, classification_model="TF",
        d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=2048, 
        dropout=0.0, activation='gelu', output_attention = True,
        device=torch.device('cuda:0')
    ):
        super(Fore_Auth_Model, self).__init__()
        self.classification_model = classification_model

        self.informer = Informer(
            enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor=5, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff, 
            dropout=0.0, attn='full', embed='timeF', freq='h', activation='gelu', 
            output_attention=True, distil=False, mix=False,
            device=device
        ).float()

        self.fcn = FCN_model(
            data_len = seq_len + out_len, 
            num_features=num_feats,
            num_class=num_class
        )

        self.transformer = tf_endoder(
            n_head=n_heads,
            d_k=64,
            d_v=64,
            seq_len=seq_len + out_len,
            num_features=num_feats, 
            d_model=d_model, 
            d_ff=d_ff,
            n_layers=e_layers,
            n_class=num_class,
            dropout=0.1
        )


    def forward(self, en_in, en_time, de_in, de_time, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        forecasting_output, attn_map = self.informer(en_in, en_time, de_in, de_time)

        forecasting_input = torch.concat((en_in, forecasting_output), dim=1)
        if self.classification_model == "FCN":
            classification_output = self.fcn(forecasting_input)
        elif self.classification_model == "TF":
            classification_output = self.transformer(forecasting_input)
        else:
            raise ValueError("classification_model must be FCN or TF")

        return attn_map, forecasting_output, classification_output