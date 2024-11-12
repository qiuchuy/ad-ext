import torch
import ailang as al
import ailang.nn as nn
import numpy as np
import torch.nn.functional as F


def numeric_check(a: al.array, b: np.ndarray):
    return np.allclose(a.tolist(), b.tolist(), rtol=1e-04, atol=1e-06)


class TorchSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.fc_out = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        seq_len, d_model = x.shape  # 假设 x 的形状为 (seq_len, d_model)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        sqrt_d_k = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        qk = torch.matmul(Q, K.transpose(-2, -1)) / sqrt_d_k
        attention_weights = F.softmax(qk, dim=-1)

        out = torch.matmul(attention_weights, V)
        return out


def create_numpy_params(model):
    """为模型生成一个 NumPy 随机参数字典"""
    params_dict = {}
    model_params = (
        model.named_parameters()
        if isinstance(model, torch.nn.Module)
        else model.parameters()
    )
    for name, param in model_params:
        params_dict[name] = param.detach().numpy()
    return params_dict


torch_model = TorchSelfAttention(d_model=3, num_heads=1)
pd = create_numpy_params(torch_model)


class AilangSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = nn.Linear(d_model, d_model)
        self.query.weight = al.from_numpy(pd["query.weight"])
        self.query.bias = al.from_numpy(pd["query.bias"])
        self.key = nn.Linear(d_model, d_model)
        self.key.weight = al.from_numpy(pd["key.weight"])
        self.key.bias = al.from_numpy(pd["key.bias"])
        self.value = nn.Linear(d_model, d_model)
        self.value.weight = al.from_numpy(pd["value.weight"])
        self.value.bias = al.from_numpy(pd["value.bias"])
        self.fc_out = nn.Linear(d_model, d_model)
        self.fc_out.weight = al.from_numpy(pd["fc_out.weight"])
        self.fc_out.bias = al.from_numpy(pd["fc_out.bias"])

    @al.jit
    def __call__(self, x, mask=None):
        # seq_len, d_model = x.shape  # 6 3
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        r = al.from_numpy(np.full((seq_len, seq_len), self.d_k).astype(np.float32))
        sqrt = al.sqrt(r)
        kt = al.transpose(K, [1, 0])
        qk = al.matmul(Q, kt)
        scores = al.div(qk, sqrt)
        attention_weights = al.prim.softmax(scores)
        out = al.matmul(attention_weights, V)
        return out


batch_size = 1
seq_len = 6
d_model = 3
num_heads = 1


def test_attention():
    x = np.random.randn(seq_len, d_model).astype(np.float32)
    a = al.from_numpy(x)
    t = torch.from_numpy(x)
    ailang_model = AilangSelfAttention(d_model, num_heads)
    ailang_res = ailang_model(a)
    torch_res = torch_model(t)
    assert numeric_check(ailang_res, torch_res.detach().numpy())
