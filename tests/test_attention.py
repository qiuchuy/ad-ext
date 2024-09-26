import torch
import ailang as al
import ailang.nn as nn
import torch.nn.functional as F


class AilangSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    @al.jit
    def __call__(self, x, mask=None):
        seq_len, d_model = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        r = al.from_numpy(np.full((seq_len, seq_len), self.d_k).astype(np.float32))
        qk = al.matmul(Q, al.transpose(K, [1, 0]))
        sqrt = al.sqrt(r)
        scores = al.div(qk, sqrt)
        # attention_weights = al.softmax(scores)
        out = al.matmul(scores, V)
        return out


class TorchSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(TorchSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # 每个head的key和query的维度
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        # 线性变换用于生成Q、K、V
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        # 输出的线性变换
        self.fc_out = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # 通过线性变换生成Q、K、V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        # 计算加权后的值
        out = torch.matmul(attention_weights, V)

        # 拼接所有的头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 通过最后的线性层
        out = self.fc_out(out)

        return out, attention_weights


# 示例用法
batch_size = 1
seq_len = 6
d_model = 3  # 输入序列的维度
num_heads = 1  # 多头注意力

import numpy as np

# 输入序列张量
x = np.random.randn(seq_len, d_model).astype(np.float32)
a = al.from_numpy(x)
# 初始化自注意力层
self_attention = AilangSelfAttention(d_model, num_heads)

# 前向传播
output = self_attention(a)

# print("输出:", output)  # (batch_size, seq_len, d_model)
