import numpy as np
import torch

def mm(mat_a, mat_b):
    if len(mat_a.shape) == 3:
        # If mat_a is 3D, perform batch matrix multiplication.
        if len(mat_b.shape) == 2:
            # If mat_b is 2D, then add a dimension to make mat_b a
            # tensor of shape [batch_size, D, 1]
            mat_b = mat_b.unsqueeze(2)
        prod = torch.bmm(mat_a, mat_b)
    else:
        assert (len(mat_a.shape) == 2) and (len(mat_b.shape) == 2)
        # If mat_a is 2D, then perform normal matrix multiplication
        prod = torch.mm(mat_a, mat_b)
    return prod

def str_to_act_and_dact_fn(actfn_name):
    if actfn_name == "Linear":
        act_fn = lambda x: x
        dact_fn = lambda x: torch.ones_like(x)
    elif actfn_name == "Tanh":
        act_fn = torch.tanh
        dact_fn = lambda x: 1 - torch.tanh(x)**2
    elif actfn_name == "Sigmoid":
        act_fn = torch.sigmoid
        dact_fn = lambda x: torch.sigmoid(x)*(1-torch.sigmoid(x))
    elif actfn_name == "ReLU":
        act_fn = torch.relu
        dact_fn = lambda x: (x > 0).float()
    return act_fn, dact_fn

def _make_first_dim_last(x):
    for i in range(x.dim()-1):
        x = x.transpose(i,i+1)
    return x

def _make_last_dim_first(x):
    for i in range(x.dim()-1, 0, -1):
        x = x.transpose(i,i-1)
    return x

def _compute_conv_output_shape(input_dim, kernel_size, stride, padding, dilation):
    to_array = lambda x: np.array([x]*2)
    conv_args = [kernel_size, stride, padding, dilation]
    for i,x in enumerate(conv_args):
        if isinstance(x, int):
            conv_args[i] = to_array(x)
    [kernel_size, stride, padding, dilation] = conv_args
    output_dim = 1 + (input_dim + 2*padding - dilation*(kernel_size - 1) -1)/stride
    output_dim = np.floor(output_dim).astype(int)
    return output_dim

def get_common_prefix(strings):
    min_len = min([len(s) for s in strings])
    i = 1
    while (i <= min_len) and all([s[:i] == strings[0][:i] for s in strings]):
        i += 1
    return strings[0][:i-1]

def get_common_suffix(strings):
    rev_strings = [s[::-1] for s in strings]
    rev_suffix = get_common_prefix(rev_strings)
    return rev_suffix[::-1]

def merge_strings(strings):
    prefix = get_common_prefix(strings)
    suffix = get_common_suffix([s.replace(prefix, '') for s in strings])
    if suffix == '':
        return prefix
    ucss = [s[len(prefix):-len(suffix)] for s in strings]    
    new_string = prefix + f"({'-'.join(ucss)})" + suffix
    return new_string