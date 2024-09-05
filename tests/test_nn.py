import ailang as al
import ailang.nn as nn
import numpy as np

input_size = 4
hidden_size = 3
wf = np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)
bf = np.random.randn(hidden_size, 1).astype(np.float32)
wi = np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)
bi = np.random.randn(hidden_size, 1).astype(np.float32)
wc = np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)
bc = np.random.randn(hidden_size, 1).astype(np.float32)
wo = np.random.randn(hidden_size, input_size + hidden_size).astype(np.float32)
bo = np.random.randn(hidden_size, 1).astype(np.float32)
x_t = np.random.rand(input_size, 1).astype(np.float32)
h_prev = np.random.rand(hidden_size, 1).astype(np.float32)
c_prev = np.random.rand(hidden_size, 1).astype(np.float32)

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.wf = al.from_numpy(wf)
        self.bf = al.from_numpy(bf)
        self.wi = al.from_numpy(wi)
        self.bi = al.from_numpy(bi)
        self.wc = al.from_numpy(wc)
        self.bc = al.from_numpy(bc)
        self.wo = al.from_numpy(wo)
        self.bo = al.from_numpy(bo)

    def sigmoid(self, x):
        return al.standard.div(
            1.0, al.standard.add(1.0, al.standard.exp(al.standard.neg(x)))
        )

    def forward(self, x_t, h_prev, c_prev):
        x = al.standard.cat([h_prev, x_t], 0)
        f = self.sigmoid(al.standard.add(al.standard.matmul(self.wf, x), self.bf))
        i = self.sigmoid(al.standard.add(al.standard.matmul(self.wi, x), self.bi))
        c_hat = al.standard.tanh(al.standard.add(al.standard.matmul(self.wc, x), self.bc))
        c = al.standard.add(al.standard.mul(f, c_prev), al.standard.mul(i, c_hat))
        o = self.sigmoid(al.standard.add(al.standard.matmul(self.wo, x), self.bo))
        h = al.standard.mul(o, al.standard.tanh(c))
        return h, c

lstm_cell = LSTMCell(input_size, hidden_size)

al_x_t = al.from_numpy(x_t)
al_h_prev = al.from_numpy(h_prev)
al_c_prev = al.from_numpy(c_prev)

al_h_t, al_c_t = lstm_cell.forward(al_x_t, al_h_prev, al_c_prev)
print("Next hidden state (h_t):", al_h_t)
print("Next cell state (c_t):", al_c_t)

class NumpyLSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = wf
        self.bf = bf
        self.Wi = wi
        self.bi = bi
        self.Wc = wc
        self.bc = bc
        self.Wo = wo
        self.bo = bo

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.vstack((h_prev, x_t))
        
        # Forget gate
        f_t = self.sigmoid(self.Wf @ combined + self.bf)
        
        # Input gate
        i_t = self.sigmoid(self.Wi @ combined + self.bi)
        c_tilde_t = self.tanh(self.Wc @ combined + self.bc)
        
        # New cell state
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # Output gate
        o_t = self.sigmoid(self.Wo @ combined + self.bo)
        
        # New hidden state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t

lstm_cell = NumpyLSTMCell(input_size, hidden_size)
h_t, c_t = lstm_cell.forward(x_t, h_prev, c_prev)
print("Next hidden state (h_t):", h_t)
print("Next cell state (c_t):", c_t)

assert np.allclose(al_h_t.tolist(), h_t.tolist())
assert np.allclose(al_c_t.tolist(), c_t.tolist())





