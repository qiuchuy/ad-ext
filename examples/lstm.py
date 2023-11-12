import ailang as al


def softmax():
    pass


@al.compile
def forward_pass(inputs, hidden_state, params):
    """
    Computes the forward pass of a vanilla RNN.

    Args:
     `inputs`: sequence of inputs to be processed
     `hidden_state`: an already initialized hidden state
     `params`: the parameters of the RNN
    """
    # First we unpack our parameters
    U, V, W, b_hidden, b_out = params

    # Create a list to store outputs and hidden states
    outputs, hidden_states = [], []

    t = 0
    input_len = len(inputs)
    # For each element in input sequence
    while t < input_len:
        # Compute new hidden state
        hidden_state = al.tanh(
            al.dot(U, inputs[t]) + al.dot(V, hidden_state) + b_hidden
        )

        # Compute output
        out = al.softmax(al.dot(W, hidden_state) + b_out)

        # Save results and continue
        outputs.append(out)

        hidden_states.append(hidden_state.copy())

        # next iteration
        t += 1

    return outputs, hidden_states
