import torch
import numpy as np

def bivariate_gaussian_kernels(M, stds, corrxy):
    '''
    Create N bivariate Gaussian PDF over [(-M/2,-M/2), (M,M)] with std.
    Inputs:
        M (int): kernel size (height = width assumed)
        stds (Tensor): Nx2 tensor containing x and y standard deviation values for N kernels
        corrxy (Tensor): N element tensor containing correlation values of N kernels
    '''
    N = stds.shape[0]

    x = torch.arange(-(M//2), M//2+1, 1)
    xy = torch.stack(torch.meshgrid(x, x), -1) # M x M x 2
    xy = xy.unsqueeze(0).repeat_interleave(stds.shape[0], 0).to(stds.device) # N x M x M x 2
    stds = stds.view(N, 1, 1, 2)
    z = xy / stds

    corrxy = corrxy.view(N, 1, 1)
    #   N x M x M          N x M x M
    w = torch.exp(- ((z ** 2).sum(-1) - 2*torch.prod(z, -1)*corrxy) / (2*(1-corrxy**2+1e-8)))
    w = w / (2*np.pi*torch.prod(stds,-1)*torch.sqrt(1 - corrxy**2)+1e-8)
    return w

def eliptical_dogkerns(ksz, stds, corrs, ratio=np.sqrt(2), divide_by_mean=False):
    '''
    Creates N eleplical Difference of Gaussian filters by calling bivariate_gaussian_kernels.
    Inputs:
        stds (Tensor): Nx2 tensor containing x and y standard deviation values for N kernels
        corrs (Tensor): N element tensor containing correlation values of N kernels
        ratio (float): The ratio of the std of the negative Gaussian to the std of the positive Gaussian

    '''
    gk1 = bivariate_gaussian_kernels(ksz, stds, corrs)
    gk2 = bivariate_gaussian_kernels(ksz, stds*ratio, corrs)
    dogk = gk1-gk2
    return dogk