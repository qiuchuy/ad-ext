import ailang as al
import ailang.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = al.random.randn((4 * hidden_size, input_size), al.f32)
        self.weight_hh = al.random.randn((4 * hidden_size, hidden_size), al.f32)
        self.bias = al.random.randn((4 * hidden_size), al.f32)

    def __call__(self, x: al.array, hx: al.array, cx: al.array) -> al.array:
        gates = (
            al.matmul(x, self.weight_ih.T) + al.matmul(hx, self.weight_hh.T) + self.bias
        )
        i, f, g, o = al.split(gates, 4, axis=1)

        i = al.sigmoid(i)
        f = al.sigmoid(f)
        g = al.tanh(g)
        o = al.sigmoid(o)

        cy = f * cx + i * g
        hy = o * al.tanh(cy)

        return hy, cy
