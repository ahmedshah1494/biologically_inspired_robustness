from copy import deepcopy
from enum import Enum, auto
from random import shuffle
from turtle import forward
from types import FunctionType
from typing import Callable, List, Literal, Tuple, Union
import torch
from torch import nn
import torchvision
import numpy as np
from scipy.stats import laplace
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

def local_pixel_shuffle(img, kernel_size):
    b, c, h, w = img.shape
    kk = kernel_size**2
    img_unf = nn.functional.unfold(img, kernel_size, stride=kernel_size)
    b, c_kk, l = img_unf.shape
    img_unf = rearrange(img_unf, 'b (c kk) l -> (b l c kk)', c=c)
    perm_idx = np.concatenate([np.random.permutation(kk)+(i*kk) for i in range(b*c*l)], 0)
    img_unf = img_unf[perm_idx]
    img_unf = rearrange(img_unf, '(b l c kk) -> b (c kk) l', c=c, b=b, kk=kk)
    img = nn.functional.fold(img_unf, (h, w), kernel_size, stride=kernel_size)
    return img

def gaussian_fn(M, std):
    n = torch.cat([torch.arange(M//2,-1,-1), torch.arange(1, M//2+1)])
    # w = torch.exp(-0.5*((n/std)**2))/(np.sqrt(2*np.pi)*std)
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    w /= w.sum()
    return w

def gkern(kernlen=256, std=128):
    assert kernlen%2 == 1
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std) 
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

def gaussian_blur_pytorch(img, kernel):
    padded_img = img#nn.ReflectionPad2d(kernel.shape[0]//2)(img)
    W = torch.repeat_interleave(kernel.unsqueeze(0).unsqueeze(0), 3, 0)
    blurred_img = nn.functional.conv2d(padded_img, W, groups=3)
    # print(img.shape, padded_img.shape, blurred_img.shape)
    return blurred_img

def _get_gaussian_filtered_image_and_density_mat_pytorch(img, isobox_w, avg_bins, loc_idx, kernels, kernel_width, shuffle_pixels=True):
    filtered_img = torch.zeros_like(img)
    density_mat = torch.zeros_like(img)
    padded_img = nn.ReflectionPad2d(kernel_width//2)(img)
    half_ks = kernel_width//2
    loc_idx_ = np.array(loc_idx)+half_ks
    # print(loc_idx_, padded_img.shape)
    for w, p, kern in zip(isobox_w, avg_bins, kernels):
        # fimg = gaussian_blur_pytorch(img, kern)
        # filtered_crop = fimg[:,:, max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w]
        # crop = img[:,:, max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w]
        crop = padded_img[:,:, max(0,loc_idx_[0]-w-half_ks):loc_idx_[0]+w+half_ks][:,:,:,max(0,loc_idx_[1]-w-half_ks):loc_idx_[1]+w+half_ks]
        # if shuffle_pixels and (w==max(isobox_w)):
        #     crop = local_pixel_shuffle(crop, 4)
        # print(w, half_ks, padded_img.shape, crop.shape, max(0,loc_idx_[0]-w-half_ks), loc_idx_[0]+w+half_ks, max(0,loc_idx_[1]-w-half_ks), loc_idx_[1]+w+half_ks)
        filtered_crop = gaussian_blur_pytorch(crop, kern)
        filtered_img[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w] = filtered_crop
        density_mat[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w] = p
    return filtered_img, density_mat

def _merge_small_bins(hist, bins, min_count):
    new_bins = [bins[-1]]
    new_hist = []
    running_count = hist[-1]
    for count, bin_end in zip(hist[::-1][1:], bins[::-1][1:]):
        running_count += count
        if running_count >= min_count:
            new_bins.append(bin_end)
            new_hist.append(running_count)
            running_count = 0
    return new_hist[::-1], new_bins[::-1]

def get_isodensity_box_width(probs, nbins):
    # p = dist_to_prob(domain, scale=scale)
    hist, bins = np.histogram(probs, bins=nbins)
    # print(hist,bins)
    hist, bins = _merge_small_bins(hist, bins, 2)
    # bins_ = deepcopy(bins)
    # # bins_[0] = 0
    # prob_bin_idxs = np.digitize(probs, bins, right=True)
    # avg_bins = []
    # for i in range(len(bins)):
    #     avg_p = probs[prob_bin_idxs == i].mean()
    #     avg_bins.append(avg_p)
    # avg_bins = np.array(avg_bins)
    # print(hist,bins)
    # plt.plot(domain, p)
    # bins[0] = 0
    avg_bins = np.array([np.min(bins[i:i+2]) if np.min(bins[i:i+2]) < 0.5 else np.max(bins[i:i+2]) for i in range(len(bins)-1)])
    # avg_bins = bins[1:]
    # print(hist, bins, avg_bins)
    return np.cumsum(hist[::-1])[::-1], avg_bins, bins

def dist_to_prob(d, scale):
    rv = laplace(scale=scale)
    # # rv = levy_stable(1, 0, scale=scale)
    p_coords = rv.pdf(d)
    scale *= 2.5
    p_coords2 = 1/(np.pi*(1+(d/scale)**2)*scale)
    p_coords = np.maximum(p_coords, p_coords2)
    # p_coords = 1/(d*np.pi*scale*(1+(np.log(d)/scale)**2))
    p_coords /= max(p_coords.max(), 1.)
    return p_coords

class AbstractRetinaFilter(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        input_shape: Union[int, List[int]] = None
        loc_mode: Literal['center', 'random_uniform'] = 'random_uniform'
        batch_size: int = 128
    
    def _get_loc(self):
        if self.params.loc_mode == 'center':
            loc = (self.input_shape[1]//2, self.input_shape[1]//2)
        elif self.params.loc_mode == 'random_uniform':
            loc = (np.random.randint(0, self.input_shape[1]), np.random.randint(0, self.input_shape[2]))
        else:
            raise ValueError('params.loc_mode must be "center" or "random_uniform"')
        return loc
    
    def _forward_batch(self, x, loc_idx):
        pass

    def forward(self, x, loc_idx=None):
        batches = torch.split(x, self.params.batch_size)
        filtered = [self._forward_batch(b, self._get_loc() if loc_idx is None else loc_idx) for b in batches]
        if len(filtered) > 1:
            filtered = torch.cat(filtered, dim=0)
        else:
            filtered = filtered[0]
        return filtered

class RetinaBlurFilter(AbstractRetinaFilter):
    @define(slots=False)
    class ModelParams(AbstractRetinaFilter.ModelParams):
        cone_std: float = None
        rod_std: float = None
        max_rod_density: float = None
        kernel_size: int = None

    def __init__(self, params, eps=1e-5) -> None:
        super().__init__(params)
        self.input_shape = self.params.input_shape
        self.cone_std = self.params.cone_std
        self.rod_std = self.params.rod_std
        self.max_rod_density = self.params.max_rod_density
        self.eps = eps

        max_dim = max(self.input_shape)
        d = np.arange(max_dim)/max_dim
        cone_density = dist_to_prob(d, self.cone_std)
        clr_isobox_w, clr_avg_bins, bins = get_isodensity_box_width(cone_density, 'auto')
        self.clr_isobox_w = clr_isobox_w
        self.clr_avg_bins = clr_avg_bins
        print(clr_isobox_w, clr_avg_bins)

        self.rod_density = (1 - dist_to_prob(d, self.rod_std)) * self.max_rod_density
        gry_isobox_w, gry_avg_bins, _ = get_isodensity_box_width(-self.rod_density, 'auto')
        self.gry_avg_bins = -gry_avg_bins
        self.gry_isobox_w = gry_isobox_w
        print(gry_isobox_w, gry_avg_bins)

        self.kernel_size = self.params.kernel_size
        if self.kernel_size % 2 == 0:
            self.kernel_size -= 1
        self.clr_kernels = nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, self.prob2std(p)), requires_grad=False) for p in self.clr_avg_bins])
        self.gry_kernels = nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, self.prob2std(p)), requires_grad=False) for p in self.gry_avg_bins])

    
    def __repr__(self):
        return f'RetinaBlurFilter(cone_std={self.cone_std}, rod_std={self.rod_std}, max_rod_density={self.max_rod_density}, kernel_size={self.kernel_size})'
    
    def prob2std(self, p):
        s = 1.5*(1-p) + 1e-5
        # print(s, p)
        return s
    
    def _forward_batch(self, img, loc_idx):
        grey_img = torch.repeat_interleave(img.mean(1, keepdims=True), 3, 1)
        gry_filtered_img, rod_density_mat = _get_gaussian_filtered_image_and_density_mat_pytorch(grey_img, self.gry_isobox_w, self.gry_avg_bins, loc_idx, self.gry_kernels, self.kernel_size)
        clr_filtered_img, cone_density_mat = _get_gaussian_filtered_image_and_density_mat_pytorch(img, self.clr_isobox_w, self.clr_avg_bins, loc_idx, self.clr_kernels, self.kernel_size)

        final_img = (rod_density_mat*gry_filtered_img + cone_density_mat*clr_filtered_img) / (rod_density_mat+cone_density_mat)
        final_img = torch.clamp(final_img, 0, 1.)
        # i = final_img[0].transpose(0,1).transpose(1,2).cpu().detach().numpy()
        # plt.figure()
        # ax = plt.subplot(1,2,1)
        # ax.imshow(i)
        # for w in self.clr_isobox_w:
        #     rect = Rectangle(np.array(loc_idx)+np.array([w, -w]), 2*w, 2*w, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # ax = plt.subplot(1,2,2)
        # ax.imshow(img[0].transpose(0,1).transpose(1,2).cpu().detach().numpy())
        # plt.savefig('input.png')
        # plt.close()
        return final_img
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class RetinaSampleFilter(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        input_shape: Union[int, List[int]] = None
        cone_std: float = None
        rod_std: float = None
        max_rod_density: float = None
        kernel_size: int = None
        loc_mode: Literal['center', 'random_uniform'] = 'random_uniform'

    def __init__(self, params, eps=1e-5) -> None:
        super().__init__(params)
        self.input_shape = self.params.input_shape
        self.cone_std = self.params.cone_std
        self.rod_std = self.params.rod_std
        self.max_rod_density = self.params.max_rod_density
        self.eps = eps

        kernel_size = np.array(self.input_shape[1:])
        max_dim = kernel_size.max()
        x, y = np.meshgrid(np.arange(kernel_size[1])/max_dim,
                            np.arange(kernel_size[0])/max_dim)
        self.coords = np.dstack((y,x))

    def get_cone_density_and_sample(self, dist):
        p = dist_to_prob(dist, self.cone_std)
        sample = np.random.binomial(1, p)
        # sample = get_evenly_spaced_sample(p_coords)
        sample = np.repeat(np.expand_dims(sample, 0), 3, 0)
        p = torch.FloatTensor(p)
        sample = torch.FloatTensor(sample)
        return p, sample

    def get_rod_density_and_sample(self, dist, cone_sample):
        rod_density = (1 - dist_to_prob(dist, self.rod_std)) * self.max_rod_density
        sample = np.random.binomial(1, rod_density)
        # sample = get_evenly_spaced_sample(rod_density)
        sample[cone_sample == 1] = 0
        sample = np.repeat(np.expand_dims(sample, 0), 3, 0)
        rod_density = torch.FloatTensor(rod_density)
        sample = torch.FloatTensor(sample)
        return rod_density, sample
        

    def _get_loc(self):
        if self.params.loc_mode == 'center':
            loc = (self.input_shape[1]//2, self.input_shape[1]//2)
        elif self.params.loc_mode == 'random_uniform':
            loc = (np.random.randint(0, self.input_shape[1]), np.random.randint(0, self.input_shape[2]))
        else:
            raise ValueError('params.loc_mode must be "center" or "random_uniform"')
        return loc
    
    def __repr__(self):
        return f'RetinaSampleFilter(cone_std={self.cone_std}, rod_std={self.rod_std}, max_rod_density={self.max_rod_density})'
    
    def forward(self, img, loc_idx=None):
        if loc_idx is None:
            loc_idx = self._get_loc()
        loc_idx = np.array(loc_idx)
        loc = loc_idx / max(self.input_shape[-2:])
        dist = self.coords - loc + 1e-4
        dist = np.max(np.abs(dist), 2)
        cone_density, cone_sample_3d = self.get_cone_density_and_sample(dist)
        rod_density, rod_sample_3d = self.get_rod_density_and_sample(dist, cone_sample_3d[0,:,:])

        grey_img = torch.repeat_interleave(img.mean(1, keepdims=True), 3, 1)
        fimg = (grey_img * rod_sample_3d.to(img.device)) + (img * cone_sample_3d.to(img.device))

        # i = fimg[0].transpose(0,1).transpose(1,2).cpu().detach().numpy()
        # plt.figure()
        # ax = plt.subplot(1,2,1)
        # ax.imshow(i)
        # ax = plt.subplot(1,2,2)
        # ax.imshow(img[0].transpose(0,1).transpose(1,2).cpu().detach().numpy())
        # plt.savefig('input.png')
        # plt.close()
        return fimg
    
    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class RetinaNonUniformPatchEmbedding(AbstractRetinaFilter):
    @define(slots=False)
    class ModelParams(AbstractRetinaFilter.ModelParams):
        hidden_size: int = None
        mask_small_rf_region: bool = False
        isobox_w: List[int] = None
        rec_flds: List[int] = None

    def __init__(self, params: BaseParameters) -> None:
        super().__init__(params)
        self.input_shape = np.array(self.params.input_shape)
        if self.params.isobox_w is None:
            n_isoboxes = int(np.log(min(self.params.input_shape[1:])) + 1 - 1e-5)
            self.isobox_w = np.array([2**(i+1) for i in range(1,n_isoboxes)])
        else:
            self.isobox_w = np.array(self.params.isobox_w)
        if self.params.rec_flds is None:
            self.rec_flds = np.array([2**i for i in range(len(self.isobox_w) + 1)])
        else:
            self.rec_flds = np.array(self.params.rec_flds)
        print(self.isobox_w, self.rec_flds)
        self.convs = nn.ModuleList([nn.Conv2d(self.params.input_shape[0], self.params.hidden_size, rf, rf) for rf in self.rec_flds])
        # self.pos_emb = PositionalEncodingPermute2D(3)
    
    def _get_loc(self):
        if self.params.loc_mode == 'center':
            loc = (self.input_shape[1]//2, self.input_shape[2]//2)
        elif self.params.loc_mode == 'random_uniform':
            offset = max(self.isobox_w) // 2
            loc = (np.random.randint(offset, self.input_shape[1]-offset), np.random.randint(offset, self.input_shape[2]-offset))
        else:
            raise ValueError('params.loc_mode must be "center" or "random_uniform"')
        return loc
    
    def _get_name(self):
        return f'RetinaNonUniformPatchEmbedding[hidden_size={self.params.hidden_size}, loc_mode={self.params.loc_mode}{", masked=True" if self.params.mask_small_rf_region else ""}, isobox_w={(self.isobox_w).tolist()}, rec_flds={self.rec_flds.tolist()}]'
    
    def _forward_batch(self, img, loc_idx):
        # img = img + self.pos_emb(img)
        if loc_idx is None:
            loc_idx = self._get_loc()

        def _get_top_left_coord(w):
            w_l = w // 2
            top_left = np.array([loc_idx[0]-w_l, loc_idx[1]-w_l])
            return top_left

        def _pad_crop(crop):
            top_left = _get_top_left_coord(w)
            top_right = top_left + np.array([0, (w - 1)])
            bot_left = top_left + np.array([(w - 1), 0])
            hpad = (max(0, -top_left[1]), max(0, top_right[1]-(self.input_shape[1] - 1)))
            vpad = (max(0, -top_left[0]), max(0, bot_left[0]-(self.input_shape[2] - 1)))
            padded_crop = nn.functional.pad(crop, hpad+vpad)
            return padded_crop

        def _get_or_set_crop(x:torch.Tensor, w:int, mode:Literal['get','set'], pad:bool=False, set_val:float = 0):
            top_left = _get_top_left_coord(w)
            pos_top_left = np.maximum(0, top_left)
            if mode == 'set':
                x[:,:, pos_top_left[0]:top_left[0]+w][:,:,:,pos_top_left[1]:top_left[1]+w] = set_val
            elif mode == 'get':
                crop = x[:,:, pos_top_left[0]:top_left[0]+w][:,:,:,pos_top_left[1]:top_left[1]+w]
                if pad:
                    crop = _pad_crop(crop)
                return crop
            else:
                raise ValueError('mode={mode} but must be either "get" or "set"')

        def _get_patch_emb(crop, conv, rf):
            grey_crop = torch.repeat_interleave(crop.mean(1, keepdims=True), 3, 1)
            crop = (1/rf)*crop + (1 - 1/rf)*grey_crop
            pemb = conv(crop)
            pemb = rearrange(pemb, 'b c h w -> b (h w) c')
            return pemb
        masked_img = img
        patch_embs = []
        crops = []
        for w, conv, rf in zip(self.isobox_w, self.convs, self.rec_flds):
            crop = _get_or_set_crop(masked_img, w, 'get', pad=True)
            crops.append(crop)
            pemb = _get_patch_emb(crop, conv, rf)
            if self.params.mask_small_rf_region:
                # b = conv.bias.unsqueeze(0) if conv.bias is not None else 0
                # nzidx = torch.arange(0,pemb.shape[1])[(pemb[0] != b).any(-1)]
                # pemb = pemb[:, nzidx]
                _get_or_set_crop(masked_img, w, 'set', set_val=0.)
            patch_embs.append(pemb)
        pemb = _get_patch_emb(masked_img, self.convs[-1], self.rec_flds[-1])
        patch_embs.append(pemb)
        patch_embs = torch.cat(patch_embs, 1)
        # fig, axs = plt.subplots(1,len(crops)+2,figsize=(25, 5))
        # axs[0].scatter([loc_idx[1]], [loc_idx[0]])
        # axs[0].imshow(rearrange(img[0], 'c h w -> h w c').cpu().detach().numpy())
        # for c, ax in zip(crops, axs[1:]):
        #     ax.imshow(rearrange(c[0], 'c h w -> h w c').cpu().detach().numpy())
        # axs[-1].imshow(rearrange(masked_img[0], 'c h w -> h w c').cpu().detach().numpy())
        # plt.savefig('input.png')
        return patch_embs