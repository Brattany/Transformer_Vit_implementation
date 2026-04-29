import torch
import torch.nn as nn

#缩放点积注意力
#Attention(Q,K,V)=softmax((QK^T)/√(d_k ))V
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask = None):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weight = torch.softmax(scores, dim = -1)
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight, v)

        return output, attention_weight

#多头注意力
#MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout = 0.0):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0
        self.d_k = d_model // h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1) 

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size,q_len,self.h,self.d_k).transpose(1,2)
        k = k.view(batch_size,k_len,self.h,self.d_k).transpose(1,2)
        v = v.view(batch_size,v_len,self.h,self.d_k).transpose(1,2)

        assert k_len == v_len
        attention_output, attention_weight = self.attention(q,k,v,mask)
        
        attention_output = attention_output.transpose(1,2).contiguous().view(
            batch_size,
            q_len,
            self.d_model
        )
        output = self.out_proj(attention_output)
        output = self.dropout(output)

        return output,attention_weight

#FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

#Enoder
class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)   
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask = None):
        #norm->attention->add->norm->FF->add(Residual Connect)
        norm_x = self.norm1(x)
        attention_output, attention_weights = self.attention(
            query = norm_x,
            key = norm_x,
            value = norm_x,
            mask = mask
        )
        x = x + attention_output

        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x, attention_weights

#Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout = 0.0):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(N):
            layer = Encoder(
                d_model = d_model,
                h = h,
                d_ff = d_ff,
                dropout = dropout
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attention_weights_list = []

        for layer in self.layers:
            x,attention_weights = layer(x,mask)
            attention_weights_list.append(attention_weights)
        
        x = self.norm(x)

        return x,attention_weights_list

#Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.maskAttention = MultiHeadAttention(d_model, h, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.crossAttention = MultiHeadAttention(d_model, h, dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # self.linear = nn.Linear(d_model, ) #词表大小，在vit中不需要DEcoder

    def forward(self,x,encoder_output,mask1 = None,mask2 = None):
        #norm->maskAtten->add->norm->crossAtten->add->norm->ff->Linear->softmax
        norm_x = self.norm1(x)
        maskAttention_output, maskAttention_weight = self.maskAttention(
            query = norm_x,
            key = norm_x,
            value = norm_x,
            mask = mask1
        )
        x = x + maskAttention_output

        crossAttention_output, crossAttention_weight = self.crossAttention(
            query = self.norm2(x),
            key = encoder_output,
            value = encoder_output,
            mask = mask2
        )
        x = x + crossAttention_output

        ff_output = self.feed_forward(self.norm3(x))
        x = x + ff_output

        return x,maskAttention_weight,crossAttention_weight

#TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout = 0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(N):
            layer = Decoder(
                d_model = d_model,
                h = h,
                d_ff = d_ff,
                dropout = dropout
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output,mask1 = None, mask2 = None):
        maskAttention_weight_list = []
        crossAttention_weight_list = []

        for layer in self.layers:
            x,maskAttention_weight, crossAttention_weight = layer(x,encoder_output,mask1,mask2)
            maskAttention_weight_list.append(maskAttention_weight)
            crossAttention_weight_list.append(crossAttention_weight)
        
        x = self.norm(x)

        return x, maskAttention_weight_list, crossAttention_weight_list
