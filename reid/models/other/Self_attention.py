import torch
import torch.nn as nn
import numpy as np

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class SelfAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention * scale
        # weighting
        attention = self.softmax(attention)
		# dropout
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=2048, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_q = nn.Linear(model_dim, model_dim)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query):
        # residual connect
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        ##### reshape to every head batch * len * dim_per_head
        key = key.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)
        value = value.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)
        query = query.transpose(2,1).contiguous().view(batch_size * num_heads, dim_per_head, -1).transpose(2,1)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.transpose(2,1).view(batch_size, num_heads*dim_per_head, -1).tranpose(2,1)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention





def PositionalEncoding( dims, vec_len):

    position_encoding = np.array([
      [pos / np.pow(10000, 2.0 * (j // 2) / dims) for j in range(dims)]
      for pos in range(vec_len)])
    # 偶数列使用sin，奇数列使用cos
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
    position_encoding = torch.from_numpy(position_encoding)
    return position_encoding
        
       