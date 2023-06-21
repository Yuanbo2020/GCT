import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, math
import framework.config as config



class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=config.d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, d_model)

    def forward(self, Q, K, V, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask()

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn




class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=config.d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask,
                d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=config.d_model, d_ff=config.d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        inputs = inputs.transpose(1, 2)
        output = nn.ReLU()(self.conv1(inputs))
        output = self.conv2(output)
        output = output.transpose(1, 2)
        residual_ouput = self.layerNorm(output + residual)
        return residual_ouput



class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                encoder_decoder_att_padding_mask,
                using_reverse=True,
                reverse_mask=None, reverse_dec_inputs=None):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      encoder_decoder_att_padding_mask)
        # torch.Size([64, 15, 512]),   torch.Size([64, 8, 15, 15])
        dec_outputs = self.pos_ffn(dec_outputs)   # torch.Size([64, 15, 512])
        # print(dec_outputs.size())
        if using_reverse:
            reverse_dec_outputs, reverse_dec_self_attn = self.dec_self_attn(reverse_dec_inputs,
                                                                            reverse_dec_inputs,
                                                                            reverse_dec_inputs, reverse_mask)
            reverse_dec_outputs, reverse_dec_enc_attn = self.dec_enc_attn(reverse_dec_outputs, enc_outputs, enc_outputs,
                                                          encoder_decoder_att_padding_mask)
            reverse_dec_outputs = self.pos_ffn(reverse_dec_outputs)
            # print(reverse_dec_outputs.size())   # torch.Size([64, 15, 512])

            return dec_outputs, dec_self_attn, dec_enc_attn, \
                   reverse_dec_outputs, reverse_dec_self_attn, reverse_dec_enc_attn
        else:
            return dec_outputs, dec_self_attn, dec_enc_attn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_padding_mask(seq_providing_expand_dim, seq_with_padding):
    batch_size, len_q = seq_providing_expand_dim.size()
    batch_size, len_k = seq_with_padding.size()
    pad_attn_mask = seq_with_padding.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_padding_mask_with_acoustic_feature(seq_providing_expand_dim, seq_with_padding):
    batch_size, len_q, _ = seq_providing_expand_dim.size()
    batch_size, len_k = seq_with_padding.size()
    pad_attn_mask = seq_with_padding.data.eq(0).unsqueeze(-1)
    return pad_attn_mask.expand(batch_size, len_k, len_q)


def get_attn_subsequent_mask(seq):
    device = seq.device
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).bool()
    return subsequent_mask.to(device)


class Decoder(nn.Module):
    def __init__(self, ntoken, n_layers, d_model=config.d_model):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(ntoken, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs,
                using_reverse, batch_y_len=None, reverse_dec_inputs=None):

        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)
        dec_self_attn_pad_mask = get_padding_mask(seq_providing_expand_dim=dec_inputs,
                                                  seq_with_padding=dec_inputs)

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        if using_reverse:
            reverse_dec_outputs = self.tgt_emb(reverse_dec_inputs)
            reverse_dec_outputs = self.pos_emb(reverse_dec_outputs)

            reverse_mask = torch.ones_like(dec_self_attn_mask).bool()
            for k in range(dec_self_attn_mask.size()[0]):
                each = dec_self_attn_mask[k]
                valid_len = batch_y_len[k]
                sub_matrix = each[:valid_len, :valid_len]
                clip = sub_matrix.int().flip(dims=[1]).bool()
                clip[:valid_len-1] = ~clip[:valid_len-1]

                reverse_mask[k, :valid_len, :valid_len] = clip

        encoder_decoder_att_padding_mask = get_padding_mask_with_acoustic_feature(seq_providing_expand_dim=enc_outputs,
                                                            seq_with_padding=dec_inputs)
        dec_self_attns, dec_enc_attns = [], []
        reverse_dec_self_attns, reverse_dec_enc_attns = [], []
        for layer in self.layers:
            if using_reverse:
                dec_outputs, dec_self_attn, dec_enc_attn, \
                reverse_dec_outputs, reverse_dec_self_attn, reverse_dec_enc_attn \
                    = layer(dec_inputs=dec_outputs,
                            enc_outputs=enc_outputs,
                            dec_self_attn_mask=dec_self_attn_mask,
                            encoder_decoder_att_padding_mask=encoder_decoder_att_padding_mask,
                            using_reverse=using_reverse,
                            reverse_mask=reverse_mask, reverse_dec_inputs=reverse_dec_outputs)
                reverse_dec_self_attns.append(reverse_dec_self_attn)
                reverse_dec_enc_attns.append(reverse_dec_enc_attn)
            else:
                dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                                                 dec_self_attn_mask,
                                                                 encoder_decoder_att_padding_mask,
                                                                 using_reverse=using_reverse)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        if using_reverse:
            return dec_outputs, dec_self_attns, dec_enc_attns, \
                   reverse_dec_outputs, reverse_dec_self_attns, reverse_dec_enc_attns
        else:
            return dec_outputs, dec_self_attns, dec_enc_attns


import torch.nn.functional as F
class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = torch.nn.Conv2d(1, embed_dim, kernel_size=(16, 16), stride=(10, 10))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_score)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_score

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        att, att_score = self.attn(self.norm1(x))
        x = x + att
        x = x + self.mlp(self.norm2(x))
        return x, att_score


class Mlp_convert768_512(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(out_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def init_layer(layer):

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class Encoder(nn.Module):
    def __init__(self, n_layers, embed_dim):
        super(Encoder, self).__init__()

        """patch"""
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        num_heads = 12
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate, inplace=True)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = 1212  # self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(n_layers)])

        self.convert768_512 = Mlp_convert768_512(in_features=embed_dim, out_features=config.d_model,
                                  act_layer=nn.GELU, drop=drop_rate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, enc_inputs):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        enc_inputs = enc_inputs.unsqueeze(1)
        # print(enc_inputs.size())  # torch.Size([64, 1, 1024, 128])
        enc_inputs = enc_inputs.transpose(2, 3)
        # print(enc_inputs.size())  # torch.Size([64, 1, 128, 1024])

        B = enc_inputs.shape[0]

        enc_inputs = self.patch_embed(enc_inputs)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # torch.Size([1, 1, 768])
        dist_token = self.dist_token.expand(B, -1, -1)  # torch.Size([1, 1, 768])
        enc_inputs = torch.cat((cls_tokens, dist_token, enc_inputs), dim=1)  # torch.Size([64, 1214, 768])
        # print(enc_inputs.size())  # torch.Size([64, 1214, 768])
        enc_inputs = enc_inputs + self.pos_embed
        # print(enc_inputs.size())  # torch.Size([64, 1214, 768])
        enc_inputs = self.pos_drop(enc_inputs)

        # print(enc_inputs.size(), len(self.blocks))  # torch.Size([64, 1214, 768]) 2
        enc_self_attns = []
        for blk in self.blocks:
            enc_inputs, enc_self_attn = blk(enc_inputs)
            # print(enc_outputs.size(), enc_self_attn.size())  # torch.Size([64, 1214, 768]) torch.Size([64, 12, 1214, 1214])
            enc_self_attns.append(enc_self_attn)
        ###########################################################################################################
        # print(enc_inputs.size())
        enc_inputs = self.convert768_512(enc_inputs)
        # print(enc_inputs.size())
        return enc_inputs, enc_self_attns



from timm.models.layers import trunc_normal_
from functools import partial
class Gated_cTransformer_patch(nn.Module):
    def __init__(self, ntoken, encoder_layers, decoder_layers, d_model=config.d_model, embed_dim = 768):
        super(Gated_cTransformer_patch, self).__init__()
        self.encoder = Encoder(n_layers=encoder_layers, embed_dim=embed_dim)
        self.decoder = Decoder(ntoken=ntoken, n_layers=decoder_layers)

        self.gated_projection = nn.Linear(ntoken, ntoken)
        self.projection = nn.Linear(d_model, ntoken)
        self.projection2 = nn.Linear(ntoken, ntoken)
        self.layerNorm = nn.LayerNorm(ntoken)

        self.my_init_weight()

    def my_init_weight(self):
        init_layer(self.gated_projection)
        init_layer(self.projection)
        init_layer(self.projection2)

    def forward(self, enc_inputs, dec_inputs, reverse_dec_inputs, using_reverse, batch_y_len):
        # print(enc_inputs.size())
        # torch.Size([64, 1024, 128])
        ##############################################################################################

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        dec_outputs, dec_self_attns, dec_enc_attns, \
        reverse_dec_outputs, reverse_dec_self_attns, reverse_dec_enc_attns = \
            self.decoder(dec_inputs, enc_outputs,
                using_reverse, batch_y_len, reverse_dec_inputs)
        # print(dec_outputs.size())   # torch.Size([64, 15, 512])

        dec_outputs = self.projection(dec_outputs)
        gated_dec_outputs = F.sigmoid(self.gated_projection(dec_outputs))
        sep_dec_outputs = nn.ReLU()(self.projection2(dec_outputs))
        dec_logits = self.layerNorm(sep_dec_outputs * (1 - gated_dec_outputs) +
                                    dec_outputs * gated_dec_outputs)

        reverse_dec_outputs = self.projection(reverse_dec_outputs)
        gated_reverse_dec_outputs = F.sigmoid(self.gated_projection(reverse_dec_outputs))
        sep_reverse_dec_outputs = nn.ReLU()(self.projection2(reverse_dec_outputs))
        reverse_dec_logits = self.layerNorm(sep_reverse_dec_outputs * (1 - gated_reverse_dec_outputs) +
                                    reverse_dec_outputs * gated_reverse_dec_outputs)

        return dec_logits, reverse_dec_logits



