import torch
import ailang as al
import ailang.nn as nn
import torch.nn.functional as F


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

        # 分割成多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

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
    
    
class AilangSelfAttention(nn.Module):
    
    
    
    
# 示例用法
batch_size = 2
seq_len = 5
d_model = 128  # 输入序列的维度
num_heads = 8  # 多头注意力

# 输入序列张量
x = torch.rand(batch_size, seq_len, d_model)

# 初始化自注意力层
self_attention = SelfAttention(d_model, num_heads)

# 前向传播
output, attention_weights = self_attention(x)

print("输出:", output.shape)  # (batch_size, seq_len, d_model)
print(
    "注意力权重:", attention_weights.shape
)  # (batch_size, num_heads, seq_len, seq_len)
