import torch
import torch.nn.functional as F
import numpy as np

# 设置输入数据和参数
operand = torch.from_numpy(np.array([[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [1.0, 2.0]]], dtype=np.float32))
offset = torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
scale = torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
mean = torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
variance = torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32))

# 添加批次维度，使operand成为4D张量
operand = operand.unsqueeze(0)  # 新形状: (1, 2, 2, 2)

# 执行batch normalization
result = F.batch_norm(
    operand,
    mean,
    variance,
    weight=scale,
    bias=offset,
    training=False,
    eps=1e-5
)

print(result)
