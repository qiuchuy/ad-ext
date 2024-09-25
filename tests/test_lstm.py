import numpy as np

np.random.seed(42)


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize combined weights and biases
        self.weight_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float32)
        self.weight_hh = np.random.randn(4 * hidden_size, hidden_size).astype(
            np.float32
        )

        self.bias_ih = np.zeros((4 * hidden_size, 1)).astype(np.float32)
        self.bias_hh = np.zeros((4 * hidden_size, 1)).astype(np.float32)

        # Split weights and biases into four parts
        self.weight_ii, self.weight_if, self.weight_ig, self.weight_io = np.split(
            self.weight_ih, 4, axis=0
        )
        self.weight_hi, self.weight_hf, self.weight_hg, self.weight_ho = np.split(
            self.weight_hh, 4, axis=0
        )

        self.bias_ii, self.bias_if, self.bias_ig, self.bias_io = np.split(
            self.bias_ih, 4, axis=0
        )
        self.bias_hi, self.bias_hf, self.bias_hg, self.bias_ho = np.split(
            self.bias_hh, 4, axis=0
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden

        # Compute gates
        i = self.sigmoid(
            np.dot(self.weight_ii, x)
            + self.bias_ii
            + np.dot(self.weight_hi, h_prev)
            + self.bias_hi
        )
        f = self.sigmoid(
            np.dot(self.weight_if, x)
            + self.bias_if
            + np.dot(self.weight_hf, h_prev)
            + self.bias_hf
        )
        g = np.tanh(
            np.dot(self.weight_ig, x)
            + self.bias_ig
            + np.dot(self.weight_hg, h_prev)
            + self.bias_hg
        )
        o = self.sigmoid(
            np.dot(self.weight_io, x)
            + self.bias_io
            + np.dot(self.weight_ho, h_prev)
            + self.bias_ho
        )

        # Compute new cell and hidden states
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)

        return h_next, c_next

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Example usage
input_size = 3
hidden_size = 2

lstm_cell = LSTMCell(input_size, hidden_size)


import torch
import torch.nn as nn

# Initialize PyTorch LSTM cell
input_size = 3
hidden_size = 2
lstm_cell_torch = nn.LSTMCell(input_size, hidden_size)

# Set weights and biases to match NumPy LSTM cell
with torch.no_grad():
    lstm_cell_torch.weight_ih.data = torch.tensor(
        lstm_cell.weight_ih, dtype=torch.float32
    )
    lstm_cell_torch.weight_hh.data = torch.tensor(
        lstm_cell.weight_hh, dtype=torch.float32
    )
    lstm_cell_torch.bias_ih.data = torch.tensor(
        lstm_cell.bias_ih.flatten(), dtype=torch.float32
    )
    lstm_cell_torch.bias_hh.data = torch.tensor(
        lstm_cell.bias_hh.flatten(), dtype=torch.float32
    )

# Create input and initial states for comparison
x = np.random.randn(input_size, 1).astype(np.float32)
h_prev = np.zeros((hidden_size, 1)).astype(np.float32)
c_prev = np.zeros((hidden_size, 1)).astype(np.float32)

# Convert NumPy inputs to PyTorch tensors
x_torch = torch.tensor(x.flatten(), dtype=torch.float32).unsqueeze(0)
h_prev_torch = torch.tensor(h_prev.flatten(), dtype=torch.float32).unsqueeze(0)
c_prev_torch = torch.tensor(c_prev.flatten(), dtype=torch.float32).unsqueeze(0)

# Forward pass through PyTorch LSTM cell
h_next_torch, c_next_torch = lstm_cell_torch(x_torch, (h_prev_torch, c_prev_torch))

w_ih = lstm_cell.weight_ih
w_hh = lstm_cell.weight_hh
b_ih = lstm_cell.bias_ih
b_hh = lstm_cell.bias_hh
w_ii = w_ih[:hidden_size]
w_if = w_ih[hidden_size : 2 * hidden_size]
w_ig = w_ih[2 * hidden_size : 3 * hidden_size]
w_io = w_ih[3 * hidden_size :]
w_hi = w_hh[:hidden_size]
w_hf = w_hh[hidden_size : 2 * hidden_size]
w_hg = w_hh[2 * hidden_size : 3 * hidden_size]
w_ho = w_hh[3 * hidden_size :]
b_ii = b_ih[:hidden_size]
b_if = b_ih[hidden_size : 2 * hidden_size]
b_ig = b_ih[2 * hidden_size : 3 * hidden_size]
b_io = b_ih[3 * hidden_size :]
b_hi = b_hh[:hidden_size]
b_hf = b_hh[hidden_size : 2 * hidden_size]
b_hg = b_hh[2 * hidden_size : 3 * hidden_size]
b_ho = b_hh[3 * hidden_size :]

import ailang as al


class ALLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ii = al.from_numpy(w_ii)
        self.weight_if = al.from_numpy(w_if)
        self.weight_ig = al.from_numpy(w_ig)
        self.weight_io = al.from_numpy(w_io)
        self.weight_hi = al.from_numpy(w_hi)
        self.weight_hf = al.from_numpy(w_hf)
        self.weight_hg = al.from_numpy(w_hg)
        self.weight_ho = al.from_numpy(w_ho)
        self.bias_ii = al.from_numpy(b_ii)
        self.bias_if = al.from_numpy(b_if)
        self.bias_ig = al.from_numpy(b_ig)
        self.bias_io = al.from_numpy(b_io)
        self.bias_hi = al.from_numpy(b_hi)
        self.bias_hf = al.from_numpy(b_hf)
        self.bias_hg = al.from_numpy(b_hg)
        self.bias_ho = al.from_numpy(b_ho)

    def sigmoid(self, x):
        return al.div(1.0, al.add(1.0, al.exp(al.neg(x))))

    @al.jit
    def forward(self, x_t, h_prev, c_prev):
        i = self.sigmoid(
            al.add(
                al.add(al.matmul(self.weight_ii, x_t), self.bias_ii),
                al.add(al.matmul(self.weight_hi, h_prev), self.bias_hi),
            )
        )
        f = self.sigmoid(
            al.add(
                al.add(al.matmul(self.weight_if, x_t), self.bias_if),
                al.add(al.matmul(self.weight_hf, h_prev), self.bias_hf),
            )
        )
        g = al.tanh(
            al.add(
                al.add(al.matmul(self.weight_ig, x_t), self.bias_ig),
                al.add(al.matmul(self.weight_hg, h_prev), self.bias_hg),
            )
        )
        o = self.sigmoid(
            al.add(
                al.add(al.matmul(self.weight_io, x_t), self.bias_io),
                al.add(al.matmul(self.weight_ho, h_prev), self.bias_ho),
            )
        )
        c_next = al.add(al.mul(f, c_prev), al.mul(i, g))
        h_next = al.mul(o, al.tanh(c_next))
        return h_next, c_next


al_lstm_cell = ALLSTMCell(input_size, hidden_size)

al_x_t = al.from_numpy(x)
al_h_prev = al.from_numpy(h_prev)
al_c_prev = al.from_numpy(c_prev)

al_h_t, al_c_t = al_lstm_cell.forward(al_x_t, al_h_prev, al_c_prev)

h_next_torch_numpy = h_next_torch.detach().numpy()
c_next_torch_numpy = c_next_torch.detach().numpy()

def test_lstm_cell():
    assert np.allclose(
        h_next_torch_numpy,
        np.array(al_h_t.tolist()).flatten(),
        rtol=1e-4,
        atol=1e-6,
    )
    assert np.allclose(
        c_next_torch_numpy,
        np.array(al_c_t.tolist()).flatten(),
        rtol=1e-4,
        atol=1e-6,
    )
