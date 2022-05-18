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