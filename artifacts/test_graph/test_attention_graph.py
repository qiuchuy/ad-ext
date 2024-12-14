import torch
import ailang as al
import ailang.nn as nn
import numpy as np
import torch.nn.functional as F


def numeric_check(a: al.array, b: np.ndarray):
    return np.allclose(a.tolist(), b.tolist(), rtol=1e-03, atol=1e-03)


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

class TorchSelfAttentionGPU(torch.nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = torch.nn.Linear(d_model, d_model).cuda()
        self.key = torch.nn.Linear(d_model, d_model).cuda()
        self.value = torch.nn.Linear(d_model, d_model).cuda()
        self.fc_out = torch.nn.Linear(d_model, d_model).cuda()

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
        if param.is_cuda:
            params_dict[name] = param.cpu().detach().numpy()
        else:
            params_dict[name] = param.detach().numpy()
    return params_dict

torch_model = TorchSelfAttention(d_model=3, num_heads=1)
torch_model_gpu = TorchSelfAttentionGPU(d_model=3, num_heads=1)
pd = create_numpy_params(torch_model)
pd_gpu = create_numpy_params(torch_model)


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

    @al.to_static
    def forward(self, x, mask=None):
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

class AilangSelfAttentionGPU(nn.Module):
    def __init__(self, d_model, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.d_k = d_model // num_heads
        assert d_model % num_heads == 0, "d_model必须能够被num_heads整除"

        self.query = nn.Linear(d_model, d_model)
        self.query.weight = al.from_numpy(pd_gpu["query.weight"], device="gpu")
        self.query.bias = al.from_numpy(pd_gpu["query.bias"], device="gpu")
        self.key = nn.Linear(d_model, d_model)
        self.key.weight = al.from_numpy(pd_gpu["key.weight"], device="gpu")
        self.key.bias = al.from_numpy(pd_gpu["key.bias"], device="gpu")
        self.value = nn.Linear(d_model, d_model)
        self.value.weight = al.from_numpy(pd_gpu["value.weight"], device="gpu")
        self.value.bias = al.from_numpy(pd_gpu["value.bias"], device="gpu")
        self.fc_out = nn.Linear(d_model, d_model)
        self.fc_out.weight = al.from_numpy(pd_gpu["fc_out.weight"], device="gpu")
        self.fc_out.bias = al.from_numpy(pd_gpu["fc_out.bias"], device="gpu")

    @al.to_static
    def forward(self, x, mask=None):
        # seq_len, d_model = x.shape  # 6 3
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        r = al.from_numpy(np.full((seq_len, seq_len), self.d_k).astype(np.float32), device="gpu")
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
    ailang_res = ailang_model.forward(a)
    torch_res = torch_model(t)
    return ailang_res, torch_res, numeric_check(ailang_res, torch_res.detach().numpy())

def test_attention_gpu():
    x = np.random.randn(seq_len, d_model).astype(np.float32)
    a = al.from_numpy(x, device="gpu")
    t = torch.from_numpy(x)
    ailang_model = AilangSelfAttentionGPU(d_model, num_heads)
    ailang_res = ailang_model.forward(a)
    torch_res = torch_model(t)
    return ailang_res, torch_res, numeric_check(ailang_res, torch_res.detach().numpy())


if __name__ == "__main__":
    # dynamic
    import numpy as np
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.utils import Colors
    from utils.utils import delimiter, center

    np.random.seed(42)

    torch.set_printoptions(precision=6)
    data_type_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for numpy")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance")
    parser.add_argument("--batch_size", type=float, default=1, help="batch_size")
    parser.add_argument("--seq_len", type=float, default=6, help="seq_len")
    parser.add_argument("--d_model", type=float, default=3, help="d_model")
    parser.add_argument("--num_heads", type=float, default=1, help="num_heads")
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        help='Data type (e.g., "float32", "float64")',
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    args.data_type = data_type_map.get(args.data_type, np.float32)  #
    delimiter("=", color=Colors.GREEN)
    center("[CRITERION 3.3] Static Graph - Attention", Colors.GREEN)
    delimiter("=", color=Colors.GREEN)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test network: Static Graph - Attention", Colors.GREEN)
    Colors.print_color(f"    device : cpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    batch_size: {args.batch_size}", Colors.GREEN)
    Colors.print_color(f"    seq_len: {args.seq_len}", Colors.GREEN)
    Colors.print_color(f"    d_model: {args.d_model}", Colors.GREEN)
    Colors.print_color(f"    num_heads: {args.num_heads}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)
    a, t, b = test_attention()
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILang's Result is :\n {a}", Colors.BOLD)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"Pytorch's Result is :\n {t.detach()}", Colors.BOLD)

    if b:
        delimiter("=", color=Colors.GREEN)
        center("Numeric Check Passed!", Colors.GREEN)
        delimiter("=", color=Colors.GREEN)
    else:
        delimiter("=", color=Colors.FAIL)
        center("Numeric Check Failed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test network: Static Graph - Attention", Colors.GREEN)
    Colors.print_color(f"    device : gpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    batch_size: {args.batch_size}", Colors.GREEN)
    Colors.print_color(f"    seq_len: {args.seq_len}", Colors.GREEN)
    Colors.print_color(f"    d_model: {args.d_model}", Colors.GREEN)
    Colors.print_color(f"    num_heads: {args.num_heads}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)
    a, t, b = test_attention_gpu()
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILang's Result is :\n {a}", Colors.BOLD)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"Pytorch's Result is :\n {t.detach()}", Colors.BOLD)

    if b:
        delimiter("=", color=Colors.GREEN)
        center("Numeric Check Passed!", Colors.GREEN)
        delimiter("=", color=Colors.GREEN)
    else:
        delimiter("=", color=Colors.FAIL)
        center("Numeric Check Failed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

