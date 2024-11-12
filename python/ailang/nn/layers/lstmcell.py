import ailang as al
import ailang.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.wf = al.random.randn(hidden_size, input_size + hidden_size, dtype=al.f32)
        self.bf = al.random.randn(hidden_size, 1, dtype=al.f32)
        self.wi = al.random.randn(hidden_size, input_size + hidden_size, dtype=al.f32)
        self.bi = al.random.randn(hidden_size, 1, dtype=al.f32)
        self.wc = al.random.randn(hidden_size, input_size + hidden_size, dtype=al.f32)
        self.bc = al.random.randn(hidden_size, 1, dtype=al.f32)
        self.wo = al.random.randn(hidden_size, input_size + hidden_size, dtype=al.f32)
        self.bo = al.random.randn(hidden_size, 1, dtype=al.f32)

    def sigmoid(self, x):
        return al.div(1.0, al.add(1.0, al.exp(al.neg(x))))

    def forward(self, x_t, h_prev, c_prev):
        x = al.cat([h_prev, x_t], axis=0)
        f = self.sigmoid(al.add(al.matmul(self.wf, x), self.bf))
        i = self.sigmoid(al.add(al.matmul(self.wi, x), self.bi))
        c_hat = al.tanh(al.add(al.matmul(self.wc, x), self.bc))
        c = al.add(al.mul(f, c_prev), al.mul(i, c_hat))
        o = self.sigmoid(al.add(al.matmul(self.wo, x), self.bo))
        h = al.mul(o, al.tanh(c))
        return h, c
