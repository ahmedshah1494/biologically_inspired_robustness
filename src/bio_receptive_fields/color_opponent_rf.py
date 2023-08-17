from typing import Literal, Union, List, Callable
import torch
from torch import nn
import numpy as np
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define
from adversarialML.biologically_inspired_models.src.bio_receptive_fields.utils import bivariate_gaussian_kernels

class _LinearColorOpponentRF(nn.Module):
    def __init__(self,
                 num_filters:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1,
                 max_std:float=None,
                 min_std:float=None,
                 single_opponent_pc:float=0.25,
                 blue_pc:float=0.02,
                 single_opponent_axis_ratio_bound:float=0.9,
                 single_opponent_corr_bound:float=0.1,
                 dog_max_std_ratio:float=np.sqrt(2),
                 min_opponent_weight:float=0.1,
                 max_loc_offset:float=0.) -> None:
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.single_opponent_pc = single_opponent_pc
        self.blue_pc = blue_pc
        self.single_opponent_axis_ratio_bound = single_opponent_axis_ratio_bound
        self.single_opponent_corr_bound = single_opponent_corr_bound
        self.dog_max_std_ratio = dog_max_std_ratio
        self.min_opponent_weight = min_opponent_weight
        if min_std is not None:
            self.min_std = min_std
        else:
            self.min_std = self.kernel_size / 16
        if max_std is not None:
            self.max_std = max_std
        else:
            self.max_std = self.kernel_size / 2
        if max_loc_offset is not None:
            self.max_loc_offset = max_loc_offset
        else:
            self.max_loc_offset = self.kernel_size / np.sqrt(2)
        
        self._create_single_opponent_filters()
        self._create_double_opponent_filters()

    def _set_kernels_to_channels(self, center_kernels, surround_kernels, out_kernels, by_mask, center_color_mask):
        center_blue_mask = (by_mask & center_color_mask).int().nonzero().flatten()
        if center_blue_mask.any():
            zero_mask = torch.zeros(len(center_blue_mask)).long()
            # set blue channel to center kernel
            out_kernels[center_blue_mask, zero_mask+2] = center_kernels[center_blue_mask]
            # set green and red channels to surround kernels
            out_kernels[center_blue_mask, zero_mask+1] = surround_kernels[center_blue_mask]
            out_kernels[center_blue_mask, zero_mask] = surround_kernels[center_blue_mask]

        center_yellow_mask = (by_mask & (~center_color_mask)).int().nonzero().flatten()
        if center_yellow_mask.any():
            zero_mask = torch.zeros(len(center_yellow_mask)).long()
            # set blue channel to surround kernel
            out_kernels[center_yellow_mask, zero_mask+2] = surround_kernels[center_yellow_mask]
            # set green and red channels to center kernels
            out_kernels[center_yellow_mask, zero_mask+1] = center_kernels[center_yellow_mask]
            out_kernels[center_yellow_mask, zero_mask] = center_kernels[center_yellow_mask]

        center_red_mask = ((~by_mask) & center_color_mask).int().nonzero().flatten()
        if center_red_mask.any():
            zero_mask = torch.zeros(len(center_red_mask)).long()
            # set red channel to center kernel
            out_kernels[center_red_mask, zero_mask] = center_kernels[center_red_mask]
            # set green channel to surround kernel
            out_kernels[center_red_mask, zero_mask+1] = surround_kernels[center_red_mask]

        center_green_mask = ((~by_mask) & (~center_color_mask)).int().nonzero().flatten()
        if center_green_mask.any():
            zero_mask = torch.zeros(len(center_green_mask)).long()
            # set green channel to center kernel
            out_kernels[center_green_mask, zero_mask+1] = center_kernels[center_green_mask]
            # set red channel to surround kernel
            out_kernels[center_green_mask, zero_mask] = surround_kernels[center_green_mask]
    
    def _create_single_opponent_filters(self):
        num_filters = int(self.num_filters * self.single_opponent_pc)

        axis_ratios = torch.empty(num_filters).uniform_(self.single_opponent_axis_ratio_bound, 1/self.single_opponent_axis_ratio_bound)
        stds1 = torch.empty(num_filters).uniform_(
                self.min_std/self.single_opponent_axis_ratio_bound,
                self.max_std*self.single_opponent_axis_ratio_bound
            )
        stds2 = stds1 * axis_ratios
        self.so_stds = torch.stack([stds1, stds2], 1)
        self.so_corrs = torch.empty(num_filters).uniform_(0, self.single_opponent_corr_bound)
        self.so_sc_ratio = torch.empty(num_filters, 1).uniform_(1, self.dog_max_std_ratio)

        num_kernels = self.so_stds.shape[0]
        self.so_opponent_weights = -1 * torch.empty(num_kernels, 1, 1).uniform_(self.min_opponent_weight, 1) # weight assigned to the surround
        center_kernels = bivariate_gaussian_kernels(self.kernel_size, self.so_stds, self.so_corrs)
        surround_kernels = bivariate_gaussian_kernels(self.kernel_size, self.so_stds*self.so_sc_ratio, self.so_corrs)
        surround_kernels *= self.so_opponent_weights

        kernels = torch.zeros(num_kernels, 3, self.kernel_size, self.kernel_size)
        by_mask = torch.empty(num_kernels).uniform_() < self.blue_pc
        center_color_mask = torch.empty(num_kernels).uniform_() < 0.5 # if True center is Red/Blue else center is Green/Yellow        
        
        self._set_kernels_to_channels(center_kernels, surround_kernels, kernels, by_mask, center_color_mask)

        self.so_kernels = nn.parameter.Parameter(kernels, requires_grad=False)

    def _create_double_opponent_filters(self):
        num_filters = int(self.num_filters * (1 - self.single_opponent_pc))

        self.do_stds = torch.empty(num_filters, 2).uniform_(self.min_std, self.max_std)
        if self.max_loc_offset > 0:
            self.do_means = torch.empty(num_filters, 2).normal_(std=self.max_loc_offset/4)
        else:
            self.do_means = torch.zeros(num_filters, 2)
        self.do_corrs = torch.empty(num_filters).uniform_(0, 0.9)
        self.do_sc_ratio = torch.empty(num_filters, 1).uniform_(1, self.dog_max_std_ratio)

        num_kernels = self.do_stds.shape[0]
        self.do_opponent_weights = -1 * torch.empty(num_kernels, 1, 1).uniform_(self.min_opponent_weight, 1) # weight assigned to the surround
        center_kernels = bivariate_gaussian_kernels(self.kernel_size, self.do_stds, self.do_corrs, means=self.do_means)
        surround_kernels = bivariate_gaussian_kernels(self.kernel_size, self.do_stds*self.do_sc_ratio, self.do_corrs, means=self.do_means)
        dog_kernels = center_kernels - surround_kernels
        weighted_dog_kernels = self.do_opponent_weights * dog_kernels

        kernels = torch.zeros(num_kernels, 3, self.kernel_size, self.kernel_size)
        by_mask = torch.empty(num_kernels).uniform_() < self.blue_pc
        center_color_mask = torch.empty(num_kernels).uniform_() < 0.5 # if True center is Red/Blue else center is Green/Yellow        
        
        self._set_kernels_to_channels(dog_kernels, weighted_dog_kernels, kernels, by_mask, center_color_mask)

        self.do_kernels = nn.parameter.Parameter(kernels, requires_grad=False)
    
    def forward(self, x):
        K = torch.cat([self.so_kernels, self.do_kernels])
        x = nn.functional.conv2d(x, K, stride=self.stride, padding=self.padding, dilation=self.dilation)
        return x
    


    # def _create_single_opponent_filters(self):
    #     num_filters = int(self.num_filters * self.single_opponent_pc)
    #     num_filters_cbrt = int(np.cbrt(num_filters))
    #     num_filters_cbrt_half = num_filters_cbrt // 2
        
    #     axis_ratios = torch.linspace(self.single_opponent_axis_ratio_bound, 1., num_filters_cbrt_half)
    #     base_stds = torch.stack(
    #         [torch.zeros(num_filters_cbrt_half)+self.max_std, axis_ratios*self.max_std], 1
    #     )
    #     stds = torch.cat([base_stds, torch.flip(base_stds, [1])], 0)
    #     corrs = torch.linspace(-self.single_opponent_corr_bound, self.single_opponent_corr_bound, num_filters_cbrt)
    #     std_ratios = torch.linspace(1., self.dog_max_std_ratio, num_filters_cbrt_half)
    #     print(base_stds.shape, stds.shape, corrs.shape, std_ratios.shape)

    #     stdsi_x_corrsi = torch.cartesian_prod(torch.arange(stds.shape[0]), torch.arange(corrs.shape[0]), torch.arange(std_ratios.shape[0]))
    #     self.single_opponent_stds = stds = stds[stdsi_x_corrsi[:,0]]
    #     self.single_opponent_corrs = corrs = corrs[stdsi_x_corrsi[:,1]]
    #     self.single_opponent_std_ratios = std_ratios = std_ratios[stdsi_x_corrsi[:,2]]

    #     num_kernels = stds.shape[0]
    #     opponent_weights = -1 * torch.empty(num_kernels).uniform_(self.min_opponent_weight, 1) # weight assigned to the surround
    #     center_kernels = bivariate_gaussian_kernels(self.kernel_size, stds, corrs)
    #     surround_kernels = bivariate_gaussian_kernels(self.kernel_size, stds*std_ratios, corrs) * opponent_weights

    #     kernels = torch.empty(num_kernels, 3, self.kernel_size, self.kernel_size)
    #     by_mask = torch.empty(num_kernels).uniform_() < self.blue_pc
    #     center_color_mask = torch.empty(num_kernels).uniform_() < 0.5 # if True center is Red/Blue else center is Green/Yellow

    #     center_blue_mask = (by_mask & center_color_mask).int().nonzero()
    #     # set blue channel to center kernel
    #     kernels[center_blue_mask, torch.zeros(len(center_blue_mask))+2] = center_kernels[center_blue_mask]
    #     # set green and red channels to surround kernels
    #     kernels[center_blue_mask, torch.zeros(len(center_blue_mask))+1] = surround_kernels[center_blue_mask]
    #     kernels[center_blue_mask, torch.zeros(len(center_blue_mask))] = surround_kernels[center_blue_mask]

    #     center_yellow_mask = (by_mask & (~center_color_mask)).int().nonzero()
    #     # set blue channel to surround kernel
    #     kernels[center_yellow_mask, torch.zeros(len(center_yellow_mask))+2] = surround_kernels[center_yellow_mask]
    #     # set green and red channels to center kernels
    #     kernels[center_yellow_mask, torch.zeros(len(center_yellow_mask))+1] = center_kernels[center_yellow_mask]
    #     kernels[center_yellow_mask, torch.zeros(len(center_yellow_mask))] = center_kernels[center_yellow_mask]

    #     center_red_mask = ((~by_mask) & center_color_mask).int().nonzero()
    #     # set red channel to center kernel
    #     kernels[center_red_mask, torch.zeros(len(center_red_mask))] = center_kernels[center_red_mask]
    #     # set green channel to surround kernel
    #     kernels[center_red_mask, torch.zeros(len(center_red_mask))+1] = surround_kernels[center_red_mask]

    #     center_green_mask = ((~by_mask) & (~center_color_mask)).int().nonzero()
    #     # set green channel to center kernel
    #     kernels[center_green_mask, torch.zeros(len(center_green_mask))+1] = center_kernels[center_green_mask]
    #     # set red channel to surround kernel
    #     kernels[center_green_mask, torch.zeros(len(center_green_mask))] = surround_kernels[center_green_mask]

    #     self.so_kernels = nn.parameter.Parameter(kernels, requires_grad=False)

class LinearColorOpponentRF(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        num_filters:int=None
        kernel_size:int=None
        stride:int=1
        padding:int=0
        dilation:int=1
        max_std:float=None
        min_std:float=None
        single_opponent_pc:float=0.25
        blue_pc:float=0.02
        single_opponent_axis_ratio_bound:float=0.9
        single_opponent_corr_bound:float=0.1
        dog_max_std_ratio:float=np.sqrt(2)
        min_opponent_weight:float=0.1
        max_loc_offset:float=0

    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self._make_network()
    
    def _make_network(self):
        kwargs_to_exclude = set(['cls'])
        kwargs = self.params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        self.filterbank = _LinearColorOpponentRF(**kwargs)
    
    def forward(self, x):
        return self.filterbank(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss