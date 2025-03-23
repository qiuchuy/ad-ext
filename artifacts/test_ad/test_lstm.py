import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import Colors
from utils.utils import delimiter, center


np.random.seed(42)

debug_value = None


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
        global debug_value
        debug_value = np.tanh(c_next)
        return np.sum(h_next + c_next)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Example usage
#input_size = 3
#hidden_size = 2
input_size = 4
hidden_size = 4

# Create input and initial states for comparison
x = np.random.randn(input_size, 1).astype(np.float32)
h_prev = np.zeros((hidden_size, 1)).astype(np.float32)
c_prev = np.zeros((hidden_size, 1)).astype(np.float32)

lstm_cell = LSTMCell(input_size, hidden_size)
np_hc_t_sum = lstm_cell.forward(x, (h_prev, c_prev))


import torch
import torch.nn as nn

lstm_cell_torch = nn.LSTMCell(input_size, hidden_size)
lstm_cell_torch_gpu = nn.LSTMCell(input_size, hidden_size).cuda()

# Set weights and biases to match NumPy LSTM cell
lstm_cell_torch.weight_ih.data = torch.tensor(lstm_cell.weight_ih, dtype=torch.float32)
lstm_cell_torch.weight_hh.data = torch.tensor(lstm_cell.weight_hh, dtype=torch.float32)
lstm_cell_torch.bias_ih.data = torch.tensor(
    lstm_cell.bias_ih.flatten(), dtype=torch.float32
)
lstm_cell_torch.bias_hh.data = torch.tensor(
    lstm_cell.bias_hh.flatten(), dtype=torch.float32
)

lstm_cell_torch_gpu.weight_ih.data = torch.tensor(lstm_cell.weight_ih, dtype=torch.float32)
lstm_cell_torch_gpu.weight_hh.data = torch.tensor(lstm_cell.weight_hh, dtype=torch.float32)
lstm_cell_torch_gpu.bias_ih.data = torch.tensor(
    lstm_cell.bias_ih.flatten(), dtype=torch.float32
)
lstm_cell_torch_gpu.bias_hh.data = torch.tensor(
    lstm_cell.bias_hh.flatten(), dtype=torch.float32
)


# Convert NumPy inputs to PyTorch tensors
x_torch = torch.tensor(x.flatten(), dtype=torch.float32, requires_grad=True).unsqueeze(
    0
)
h_prev_torch = torch.tensor(
    h_prev.flatten(), dtype=torch.float32, requires_grad=True
).unsqueeze(0)
c_prev_torch = torch.tensor(
    c_prev.flatten(), dtype=torch.float32, requires_grad=True
).unsqueeze(0)
x_torch.retain_grad()
h_prev_torch.retain_grad()
c_prev_torch.retain_grad()

x_torch_gpu = torch.tensor(x.flatten(), dtype=torch.float32, requires_grad=True).unsqueeze(
    0
)
h_prev_torch_gpu = torch.tensor(
    h_prev.flatten(), dtype=torch.float32, requires_grad=True
).unsqueeze(0)
c_prev_torch_gpu = torch.tensor(
    c_prev.flatten(), dtype=torch.float32, requires_grad=True
).unsqueeze(0)
x_torch_gpu.retain_grad()
h_prev_torch_gpu.retain_grad()
c_prev_torch_gpu.retain_grad()

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

    @al.grad
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
        return al.sum(al.add(c_next, h_next))

class ALLSTMCellGPU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ii = al.from_numpy(w_ii, device="gpu")
        self.weight_if = al.from_numpy(w_if, device="gpu")
        self.weight_ig = al.from_numpy(w_ig, device="gpu")
        self.weight_io = al.from_numpy(w_io, device="gpu")
        self.weight_hi = al.from_numpy(w_hi, device="gpu")
        self.weight_hf = al.from_numpy(w_hf, device="gpu")
        self.weight_hg = al.from_numpy(w_hg, device="gpu")
        self.weight_ho = al.from_numpy(w_ho, device="gpu")
        self.bias_ii = al.from_numpy(b_ii, device="gpu")
        self.bias_if = al.from_numpy(b_if, device="gpu")
        self.bias_ig = al.from_numpy(b_ig, device="gpu")
        self.bias_io = al.from_numpy(b_io, device="gpu")
        self.bias_hi = al.from_numpy(b_hi, device="gpu")
        self.bias_hf = al.from_numpy(b_hf, device="gpu")
        self.bias_hg = al.from_numpy(b_hg, device="gpu")
        self.bias_ho = al.from_numpy(b_ho, device="gpu")

    def sigmoid(self, x):
        return al.div(1.0, al.add(1.0, al.exp(al.neg(x))))

    @al.grad
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
        return al.sum(al.add(c_next, h_next))


def test_lstm_grad():
    assert np.allclose(
        (np.sum(tc_hc_t_sum.detach().numpy())).flatten(),
        np.array(al_hc_t_sum.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )
    assert np.allclose(
        x_torch.grad.detach().numpy().flatten(),
        np.array(grad_x_t.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )
    assert np.allclose(
        h_prev_torch.grad.detach().numpy().flatten(),
        np.array(grad_h_prev.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )
    assert np.allclose(
        h_prev_torch.grad.detach().numpy().flatten(),
        np.array(grad_h_prev.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )
    assert np.allclose(
        c_prev_torch.grad.detach().numpy().flatten(),
        np.array(grad_c_prev.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
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
    parser.add_argument("--input_size", type=int, default=4, help="input_size")
    parser.add_argument("--hidden_size", type=int, default=4, help="hidden_size")
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
    center("[CRITERION 4.2]  Backward - LSTM", Colors.GREEN)
    delimiter("=", color=Colors.GREEN)

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test network: Backward(Grad) - LSTM", Colors.GREEN)
    Colors.print_color(f"    device : cpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    input_size: {args.input_size}", Colors.GREEN)
    Colors.print_color(f"    hidden_size: {args.hidden_size}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)

    al_lstm_cell = ALLSTMCell(input_size, hidden_size)

    al_x_t = al.from_numpy(x)
    al_h_prev = al.from_numpy(h_prev)
    al_c_prev = al.from_numpy(c_prev)

    al_hc_t_sum, grad_x_t, grad_h_prev, grad_c_prev = al_lstm_cell.forward(
        al_x_t, al_h_prev, al_c_prev
    )

    tc_hc_t_sum = torch.sum(h_next_torch + c_next_torch)
    tc_hc_t_sum.backward()

    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILang's grad_x_t Grad is :\n {grad_x_t}", Colors.BOLD)
    Colors.print_color(f"AILang's grad_h_prev Grad is :\n {grad_h_prev}", Colors.BOLD)
    Colors.print_color(f"AILang's grad_c_prev Grad is :\n {grad_c_prev}", Colors.BOLD)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"Pytorch's hc_t_sum Grad is :\n {tc_hc_t_sum}", Colors.BOLD)
    Colors.print_color(
        f"Pytorch's grad_x_t Grad is :\n { x_torch.grad.detach()}", Colors.BOLD
    )
    Colors.print_color(
        f"Pytorch's grad_h_prev Grad is :\n {h_prev_torch.grad.detach()}", Colors.BOLD
    )
    Colors.print_color(
        f"Pytorch's grad_c_prev Grad is :\n {c_prev_torch.grad.detach()}", Colors.BOLD
    )
    if (
        np.allclose(
            x_torch.grad.detach().numpy().flatten(),
            np.array(grad_x_t.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
        and np.allclose(
            h_prev_torch.grad.detach().numpy().flatten(),
            np.array(grad_h_prev.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
        and np.allclose(
            c_prev_torch.grad.detach().numpy().flatten(),
            np.array(grad_c_prev.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
    ):
        delimiter("=", color=Colors.GREEN)
        center("Numeric Check Passed!", Colors.GREEN)
        delimiter("=", color=Colors.GREEN)
    else:
        delimiter("=", color=Colors.FAIL)
        center("Numeric Check Passed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

    assert np.allclose(
        c_prev_torch.grad.detach().numpy().flatten(),
        np.array(grad_c_prev.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )

    Colors.print_color("Test Illustration", Colors.GREEN)
    Colors.print_color(f"    test network: Backward(Grad) - LSTM", Colors.GREEN)
    Colors.print_color(f"    device : gpu", Colors.GREEN)
    Colors.print_color(f"    rtol: {args.rtol}", Colors.GREEN)
    Colors.print_color(f"    atol: {args.atol}", Colors.GREEN)
    Colors.print_color("Input Data", Colors.GREEN)
    Colors.print_color(f"    dtype: float32", Colors.GREEN)
    Colors.print_color(f"    input_size: {args.input_size}", Colors.GREEN)
    Colors.print_color(f"    hidden_size: {args.hidden_size}", Colors.GREEN)
    Colors.print_color(f"    random_seed: {args.seed}", Colors.GREEN)
    Colors.print_color(f"Begin Check: ", Colors.GREEN)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILANG IR", Colors.GREEN)
    al_lstm_cell_gpu = ALLSTMCellGPU(input_size, hidden_size)

    al_x_t = al.from_numpy(x, device="gpu")
    al_h_prev = al.from_numpy(h_prev, device="gpu")
    al_c_prev = al.from_numpy(c_prev, device="gpu")

    al_hc_t_sum, grad_x_t, grad_h_prev, grad_c_prev = al_lstm_cell_gpu.forward(
        al_x_t, al_h_prev, al_c_prev
    )

    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"AILang's hc_t_sum Grad is :\n {al_hc_t_sum}", Colors.BOLD)
    Colors.print_color(f"AILang's grad_x_t Grad is :\n {grad_x_t}", Colors.BOLD)
    Colors.print_color(f"AILang's grad_h_prev Grad is :\n {grad_h_prev}", Colors.BOLD)
    Colors.print_color(f"AILang's grad_c_prev Grad is :\n {grad_c_prev}", Colors.BOLD)
    delimiter("*", color=Colors.GREEN)
    Colors.print_color(f"Pytorch's hc_t_sum Grad is :\n {tc_hc_t_sum}", Colors.BOLD)
    Colors.print_color(
        f"Pytorch's grad_x_t Grad is :\n { x_torch.grad.detach()}", Colors.BOLD
    )
    Colors.print_color(
        f"Pytorch's grad_h_prev Grad is :\n {h_prev_torch.grad.detach()}", Colors.BOLD
    )
    Colors.print_color(
        f"Pytorch's grad_c_prev Grad is :\n {c_prev_torch.grad.detach()}", Colors.BOLD
    )
    if (
        np.allclose(
            x_torch.grad.detach().numpy().flatten(),
            np.array(grad_x_t.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
        and np.allclose(
            h_prev_torch.grad.detach().numpy().flatten(),
            np.array(grad_h_prev.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
        and np.allclose(
            c_prev_torch.grad.detach().numpy().flatten(),
            np.array(grad_c_prev.tolist()).flatten(),
            rtol=1e-3,
            atol=1e-3,
        )
    ):
        delimiter("=", color=Colors.GREEN)
        center("Numeric Check Passed!", Colors.GREEN)
        delimiter("=", color=Colors.GREEN)
    else:
        delimiter("=", color=Colors.FAIL)
        center("Numeric Check Passed!", Colors.FAIL)
        delimiter("=", color=Colors.FAIL)

    assert np.allclose(
        c_prev_torch.grad.detach().numpy().flatten(),
        np.array(grad_c_prev.tolist()).flatten(),
        rtol=1e-3,
        atol=1e-3,
    )
