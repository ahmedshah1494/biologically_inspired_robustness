from copy import deepcopy
from enum import Enum, auto
from random import shuffle
from turtle import forward
from types import FunctionType
from typing import Callable, List, Tuple, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
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
from adversarialML.biologically_inspired_models.retinawarp.retina import retina_pt

def convert_image_tensor_to_ndarray(img):
    return img.cpu().detach().transpose(0,1).transpose(1,2).numpy()

def local_pixel_shuffle(img, kernel_size):
    b, c, h, w = img.shape
    kk = kernel_size**2
    print(img.shape)
    img_unf = nn.functional.unfold(img, kernel_size, stride=kernel_size)
    b, c_kk, l = img_unf.shape
    print(img_unf.shape)
    img_unf = rearrange(img_unf, 'b (c kk) l -> (b l c kk)', c=c)
    print(img_unf.shape)
    perm_idx = np.concatenate([(np.repeat(np.random.permutation(kk).reshape(1,-1), c, axis=0) + kk*np.arange(c).reshape(-1,1)).reshape(-1) +(i*(kk*c)) for i in range(b*l)], 0)
    img_unf = img_unf[perm_idx]
    img_unf = rearrange(img_unf, '(b l c kk) -> b (c kk) l', c=c, b=b, kk=kk)
    print(img_unf.shape)
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

def seperable_gaussian_blur_pytorch(img, kernel):
    W = torch.repeat_interleave(kernel.unsqueeze(0).unsqueeze(0), 3, 0)
    b,c,h,w = img.shape
    img = rearrange(img, 'b c h w -> (b h) c w')
    blurred_img = nn.functional.conv1d(img, W, groups=3)
    blurred_img = rearrange(blurred_img, '(b h) c w -> (b w) c h', b=b)
    blurred_img = nn.functional.conv1d(blurred_img, W, groups=3)
    blurred_img = rearrange(blurred_img, '(b w) c h -> b c h w', b=b)
    return blurred_img

def seperable_DoG_blur_pytorch(img, kernels):
    img1 = seperable_gaussian_blur_pytorch(img, kernels[0])
    img2 = seperable_gaussian_blur_pytorch(img, kernels[1])
    return img1-img2

def _get_gaussian_filtered_image_and_density_mat_pytorch(img, isobox_w, avg_bins, loc_idx, kernels, kernel_width, shuffle_pixels=True, blur=True, gblur_fn=gaussian_blur_pytorch):
    filtered_img = torch.zeros_like(img) if blur else img
    density_mat = torch.zeros_like(img)
    padded_img = nn.ReflectionPad2d(kernel_width//2)(img)
    half_ks = kernel_width//2
    loc_idx_ = np.array(loc_idx)+half_ks
    # print(loc_idx_, padded_img.shape)
    imgsize = max(img.shape[2:])
    if (len(isobox_w) > 0) and (isobox_w >= imgsize).any():
        maxw = isobox_w[np.argmin(isobox_w[isobox_w >= imgsize] - imgsize)]
    elif len(isobox_w) > 0:
        maxw = isobox_w.max()
    for w, p, kern in zip(isobox_w, avg_bins, kernels):
        if w > maxw:
            continue
        if blur and len(kern[kern >= 1e-4]) > 1:
            # fimg = gaussian_blur_pytorch(img, kern)
            # filtered_crop = fimg[:,:, max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w]
            # crop = img[:,:, max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w]
            half_ks = kern.shape[-1] // 2
            crop = padded_img[:,:, max(0,loc_idx_[0]-w-half_ks):loc_idx_[0]+w+half_ks][:,:,:,max(0,loc_idx_[1]-w-half_ks):loc_idx_[1]+w+half_ks]
            # if shuffle_pixels and (w==max(isobox_w)):
            #     crop = local_pixel_shuffle(crop, 4)
            # print(w, half_ks, padded_img.shape, crop.shape, max(0,loc_idx_[0]-w-half_ks), loc_idx_[0]+w+half_ks, max(0,loc_idx_[1]-w-half_ks), loc_idx_[1]+w+half_ks)
            filtered_crop = gblur_fn(crop, kern)
            filtered_img[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w] = filtered_crop
        elif blur:
            filtered_img[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w] = img[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w]
        density_mat[:,:,max(0,loc_idx[0]-w):loc_idx[0]+w][:,:,:,max(0,loc_idx[1]-w):loc_idx[1]+w] = p
    return filtered_img, density_mat

def _merge_small_bins(hist, bins, min_count):
    new_bins = [bins[-1]]
    new_hist = []
    running_count = 0
    for count, bin_end in zip(hist[::-1], bins[::-1][1:]):
        running_count += count
        if running_count >= min_count:
            new_bins.append(bin_end)
            new_hist.append(running_count)
            running_count = 0
    return new_hist[::-1], new_bins[::-1]

def get_isodensity_box_width(probs, nbins, min_bincount=2):
    hist, bins = np.histogram(probs, bins=nbins)
    hist, bins = _merge_small_bins(hist, bins, min_bincount)
    avg_bins = []
    for i in range(len(bins)-1):
        lo = bins[i]
        hi = bins[i+1]
        if i < len(bins) - 2:
            avg_p = probs[(probs >= lo) & (probs < hi)].mean()
        else:
            avg_p = probs[(probs >= lo) & (probs <= hi)].mean()
        avg_bins.append(avg_p)
    avg_bins = np.array(avg_bins)
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

class GaussianBlurLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        kernel_size: int = None
        std: float = None
        prob: float = 1.

    def __init__(self, params) -> None:
        super().__init__(params)
        self.std = self.params.std
        if self.params.kernel_size is None:
            self.kernel_size = 4*int(self.std+1) + 1
        else:
            self.kernel_size = self.params.kernel_size
        self.gblur = torchvision.transforms.GaussianBlur(self.kernel_size, self.std)
    
    def forward(self, img):
        return self.gblur(img)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class GreyscaleLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        clr_wt: float = 0
        ndim: int = 1
    
    def forward(self, x: torch.Tensor):
        x = x.mean(1, keepdim=True)
        x = torch.repeat_interleave(x, self.params.ndim, 1)
        return x

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class GaussianNoiseLayer(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        std: float = None
        add_noise_during_inference: bool = False
        add_deterministic_noise_during_inference: bool = False
        max_input_size: List[int] = [3, 224, 224]
        neuronal_noise: bool = False

    def __init__(self, params) -> None:
        super().__init__(params)
        self.std = self.params.std
        if self.params.add_deterministic_noise_during_inference:
            self.register_buffer('noise_patch', 
                                 torch.empty(1,*(params.max_input_size)).normal_(0, self.std, generator=torch.Generator().manual_seed(51972691))
                                 )
    
    def __repr__(self):
        return f"GaussianNoiseLayer(std={self.std}, neuronal={self.params.neuronal_noise})"
    
    def forward(self, img):
        if self.training or self.params.add_noise_during_inference:
            d = torch.empty_like(img).normal_(0, self.std)
        else:
            if self.params.add_deterministic_noise_during_inference:
                b, c, h ,w = img.shape
                # d = self.noise_patch[:c, :h, :w].unsqueeze(0)
                d = self.noise_patch
                # print(d.min(), d.mean(), d.median(), d.max(), d.sum(), d.abs().sum())
            else:
                d = 0
        if self.params.neuronal_noise:
            d = d * torch.sqrt(torch.relu(img.clone()) + 1e-5)
        return img + d

    def compute_loss(self, x, y, return_logits=True, **kwargs):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class AbstractRetinaFilter(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        input_shape: Union[int, List[int]] = None
        loc_mode: Literal['center', 'random_uniform', 'random_five_fixations', 'five_fixations', 'const'] = 'random_uniform'
        loc: Tuple[int, int] = None
        batch_size: int = 128
        straight_through: bool = False

    def _get_five_fixations(self, img, randomize=False):
        _, _, h, w = img.shape
        if randomize:
            pts = [(p // w, p % w) for p in np.random.choice(h*w, 5)]
        else:
            pts = [
                (h//6, w//6),
                (h//6, w-w//6),
                (h//2, w//2),
                (h-h//6, w//6),
                (h-h//6, w-w//6),
            ]
        return pts
    
    def _get_loc(self, batch, batch_idx):
        input_shape = batch.shape[1:]
        if self.params.loc_mode == 'center':
            loc = (input_shape[1]//2, input_shape[1]//2)
        elif self.params.loc_mode == 'random_uniform':
            loc = (np.random.randint(0, self.params.input_shape[1]), np.random.randint(0, self.params.input_shape[2]))
        elif self.params.loc_mode == 'const':
            if isinstance(self.params.loc, tuple):
                loc = self.params.loc
            elif np.iterable(self.params.loc):
                loc = self.params.loc[batch_idx]
            else:
                raise ValueError(f'params.loc must be tuple or (nested) iterable of tuples, but got {self.params.loc}')
        else:
            raise ValueError('params.loc_mode must be "center" or "random_uniform"')
        return loc
    
    def _forward_batch(self, x, loc_idx):
        pass

    def forward(self, x, loc_idx=None):
        if self.params.loc_mode in ['five_fixations', 'random_five_fixations']:
            locs = self._get_five_fixations(x, self.params.loc_mode.startswith('random'))
            filtered = [self._forward_batch(x, loc) for loc in locs]
            filtered = torch.stack(filtered, dim=1)
            filtered = filtered.reshape(-1, *(filtered.shape[2:]))
        else:
            batches = torch.split(x, self.params.batch_size)
            # filtered = [self._forward_batch(b, self._get_loc(b,i) if loc_idx is None else loc_idx) for i,b in enumerate(batches)]
            filtered = []
            for i,b in enumerate(batches):
                _loc_idx = self._get_loc(b,i) if loc_idx is None else loc_idx
                if isinstance(_loc_idx, tuple):
                    filtered.append(self._forward_batch(b, _loc_idx))
                if isinstance(_loc_idx, list) and isinstance(_loc_idx[0], tuple):
                    fimg = torch.cat([self._forward_batch(b, li) for li in _loc_idx], 0)
                    filtered.append(fimg)
            if len(filtered) > 1:
                filtered = torch.cat(filtered, dim=0)
            else:
                filtered = filtered[0]
        if self.params.straight_through:
            return filtered.detach() + x - x.detach()
        return filtered

class RetinaBlurFilter(AbstractRetinaFilter):
    @define(slots=False)
    class ModelParams(AbstractRetinaFilter.ModelParams):
        cone_std: float = None
        rod_std: float = None
        max_rod_density: float = None
        max_kernel_size: int = np.inf
        view_scale: Union[int, str] = None
        only_color: bool = False
        no_blur: bool = False
        scale: float = 0.05
        use_1d_gkernels: bool = False
        min_bincount: int = 224//16
        set_min_bin_to_1: bool = False

    def __init__(self, params, eps=1e-5) -> None:
        super().__init__(params)
        self.input_shape = self.params.input_shape
        self.cone_std = self.params.cone_std
        self.rod_std = self.params.rod_std
        self.max_rod_density = self.params.max_rod_density
        self.eps = eps
        self.scale = self.params.scale
        self.include_gry_img = not self.params.only_color
        self.apply_blur = not self.params.no_blur

        img_width = max(self.input_shape[1:])
        max_kernel_size = min(self.params.max_kernel_size, img_width)

        max_dim = max(self.input_shape)
        d = np.arange(max_dim)/max_dim
        cone_density = dist_to_prob(d, self.cone_std)
        clr_isobox_w, clr_avg_bins, bins = get_isodensity_box_width(cone_density, 'auto', min_bincount=params.min_bincount)
        if params.set_min_bin_to_1:
            clr_avg_bins[-1] = 1.
        self.clr_isobox_w = clr_isobox_w
        self.clr_avg_bins = clr_avg_bins

        self.rod_density = (1 - dist_to_prob(d, self.rod_std)) * self.max_rod_density
        gry_isobox_w, gry_avg_bins, _ = get_isodensity_box_width(-self.rod_density, 'auto', min_bincount=params.min_bincount)
        self.gry_avg_bins = -gry_avg_bins
        self.gry_isobox_w = gry_isobox_w

        clr_stds = [self.prob2std(p) for p in self.clr_avg_bins]
        gry_stds = [self.prob2std(p) for p in self.gry_avg_bins]
        print(clr_isobox_w, clr_stds)
        print(gry_isobox_w, gry_stds)

        if isinstance(self.params.view_scale, int):
            self.view_scale = min(self.params.view_scale, len(clr_stds), len(gry_stds))
        else:
            self.view_scale = self.params.view_scale

        max_std = max(clr_stds + gry_stds)
        self.kernel_size = min(4*int(np.ceil(max_std))+1, max_kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size -= 1

        if self.apply_blur:
            # self.clr_kernels = nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, self.prob2std(p)), requires_grad=False) for p in self.clr_avg_bins])
            self.clr_kernels = self.create_kernels(clr_stds)
            if self.include_gry_img:
                # self.gry_kernels = nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, self.prob2std(p)), requires_grad=False) for p in self.gry_avg_bins])
                self.gry_kernels = self.create_kernels(gry_stds)
        else:
            self.clr_kernels = [None]*len(self.clr_avg_bins)
            self.gry_kernels = [None]*len(self.gry_avg_bins)
    
    def create_kernels(self, std_list):
        if self.params.use_1d_gkernels:
            return [gaussian_fn(self.kernel_size, std=s) for s in std_list]
        else:
            return [gkern(self.kernel_size, s) for s in std_list]
        # return nn.ParameterList([nn.parameter.Parameter(gaussian_fn(int(np.ceil(4*s)), std=s), requires_grad=False) for s in std_list])
    
    def apply_kernel(self, img, isobox_w, avg_bins, loc_idx, kernels):
        gfn = seperable_gaussian_blur_pytorch if self.params.use_1d_gkernels else gaussian_blur_pytorch
        return _get_gaussian_filtered_image_and_density_mat_pytorch(img, isobox_w, avg_bins, loc_idx, 
                                                            kernels, self.kernel_size, blur=self.apply_blur,
                                                            gblur_fn=gfn
                                                            )
    
    def __repr__(self):
        return f'RetinaBlurFilter(loc_mode={self.params.loc_mode}, cone_std={self.cone_std}, rod_std={self.rod_std}, max_rod_density={self.max_rod_density}, kernel_size={self.kernel_size}, view_scale={self.view_scale}, beta={self.scale})'
    
    def prob2std(self, p):
        s = self.scale*max(self.input_shape[1:])*(1-p) + 1e-5
        return s

    def _get_view_scale(self):
        if isinstance(self.view_scale, int):
            return self.view_scale
        else:
            if self.view_scale is None:
                s = 0
            elif self.view_scale == 'random_uniform':
                max_s = min(len(self.clr_isobox_w), len(self.gry_isobox_w)) - 1
                s = np.random.randint(0, max_s)
            else:
                raise ValueError(f'view_scale must be either None or random_uniform but got {self.view_scale}')
            return s

    def _forward_batch(self, img, loc_idx):
        s = self._get_view_scale()
        # print(f'view_scale={s}')
        assert not ((self.params.view_scale is None) and (s > 0))
        if s > 0:
            if self.include_gry_img:
                gry_isobox_w = self.gry_isobox_w[:-s]
                gry_avg_bins = self.gry_avg_bins[s:]
                gry_kernels = self.gry_kernels[s:]
            clr_isobox_w = self.clr_isobox_w[:-s]
            clr_avg_bins = self.clr_avg_bins[s:]
            clr_kernels = self.clr_kernels[s:]
        else:
            if self.include_gry_img:
                gry_isobox_w = self.gry_isobox_w
                gry_avg_bins = self.gry_avg_bins
                gry_kernels = self.gry_kernels
            clr_isobox_w = self.clr_isobox_w
            clr_avg_bins = self.clr_avg_bins
            clr_kernels = self.clr_kernels
        clr_kernels = [k.type(img.dtype).to(img.device) if k is not None else k for k in clr_kernels]
        # print(clr_isobox_w, self.clr_isobox_w)
        # print([self.prob2std(p) for p in clr_avg_bins], [self.prob2std(p) for p in self.clr_avg_bins])
        # print(gry_isobox_w, self.gry_isobox_w)
        # print([self.prob2std(p) for p in gry_avg_bins], [self.prob2std(p) for p in self.gry_avg_bins])
        clr_filtered_img, cone_density_mat = self.apply_kernel(img, clr_isobox_w, clr_avg_bins, loc_idx, clr_kernels)
        if self.include_gry_img:
            gry_kernels = [k.type(img.dtype).to(img.device) if k is not None else k for k in gry_kernels]
            grey_img = torch.repeat_interleave(img.mean(1, keepdims=True), 3, 1)
            gry_filtered_img, rod_density_mat = self.apply_kernel(grey_img, gry_isobox_w, gry_avg_bins, loc_idx, gry_kernels)

            final_img = (rod_density_mat*gry_filtered_img + cone_density_mat*clr_filtered_img) / (rod_density_mat+cone_density_mat)
        else:
            final_img = clr_filtered_img
        # final_img = torch.clamp(final_img, 0, 1.)
        # nplots = 5
        # # print(img[0].min(), img[0].max(), clr_filtered_img[0].min(), clr_filtered_img[0].max(), gry_filtered_img[0].min(), gry_filtered_img[0].max(), final_img[0].min(), final_img[0].max())
        # f = plt.figure(figsize=(20,nplots))
        # plt.subplot(1,nplots,1)
        # plt.title('original')
        # plt.imshow(convert_image_tensor_to_ndarray(img[0]))
        # plt.subplot(1,nplots,2)
        # plt.title('Cone Output')
        # plt.imshow(convert_image_tensor_to_ndarray(clr_filtered_img[0]))
        # # plt.subplot(1,nplots,3)
        # # plt.title('Rod Output')
        # # plt.imshow(convert_image_tensor_to_ndarray(gry_filtered_img[0]))
        # plt.subplot(1,nplots,4)
        # plt.title('Combined Output')
        # plt.imshow(convert_image_tensor_to_ndarray(final_img[0]))
        # plt.subplot(1,nplots,5)
        # # plt.imshow(cone_density_mat[0,0])
        # plt.plot(np.arange(img.shape[3]), cone_density_mat[0,0, loc_idx[0]].cpu().detach().numpy())
        # # plt.plot(np.arange(img.shape[3]), rod_density_mat[0,0, loc_idx[0]].cpu().detach().numpy())
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
        
class RetinDoGBlurFilter(RetinaBlurFilter):
    @define(slots=False)
    class ModelParams(RetinaBlurFilter.ModelParams):
        DoG_factor: int = 5

    # def prob2std(self, p):
    #     s = self.params.DoG_factor*self.scale*max(self.input_shape[1:])*(1-p) + 1e-5
    #     return s

    def create_kernels(self, std_list):
        if self.params.use_1d_gkernels:
            return nn.ParameterList([nn.parameter.Parameter(torch.stack([gaussian_fn(self.kernel_size, std=s), gaussian_fn(self.kernel_size, std=s*self.params.DoG_factor)], 0), requires_grad=False) for s in std_list])
        else:
            return nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, s) - gkern(self.kernel_size, s*self.params.DoG_factor), requires_grad=False) for s in std_list])
    
    def apply_kernel(self, img, isobox_w, avg_bins, loc_idx, kernels):
        gfn = seperable_DoG_blur_pytorch if self.params.use_1d_gkernels else gaussian_blur_pytorch
        return _get_gaussian_filtered_image_and_density_mat_pytorch(img, isobox_w, avg_bins, loc_idx, 
                                                            kernels, self.kernel_size, blur=self.apply_blur,
                                                            gblur_fn=gfn
                                                            )

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
        conv_stride: int = None
        conv_padding: Union[int, str] = 0
        place_crop_features_in_grid: bool = False
        normalization_layer_params: BaseParameters = None
        visualize:bool = False

    def __init__(self, params) -> None:
        super().__init__(params)
        if self.params.place_crop_features_in_grid and (self.params.conv_stride != 1) and (self.params.conv_padding != 0):
            raise ValueError(f'if place_crop_features_in_grid=True then set conv_stride=1 and conv_padding=0, but got conv_stride={self.params.conv_stride} and conv_padding={self.params.conv_padding}')
            
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
        s = self.params.conv_stride
        p = self.params.conv_padding
        self.convs = nn.ModuleList([nn.Conv2d(self.params.input_shape[0], self.params.hidden_size, rf, rf if s is None else s, p) for rf in self.rec_flds])
        self.visualize = self.params.visualize
        if self.params.normalization_layer_params is not None:
            self.normalization_layer = self.params.normalization_layer_params.cls(self.params.normalization_layer_params)
        else:
            self.normalization_layer = nn.Identity()
    
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
    
    def _get_top_left_coord(self, w, loc_idx):
        w_l = w // 2
        top_left = np.array([loc_idx[0]-w_l, loc_idx[1]-w_l])
        return top_left

    def _pad_crop(self, crop, w, loc_idx):
        top_left = self._get_top_left_coord(w, loc_idx)
        top_right = top_left + np.array([0, (w - 1)])
        bot_left = top_left + np.array([(w - 1), 0])
        hpad = (max(0, -top_left[1]), max(0, top_right[1]-(self.input_shape[1] - 1)))
        vpad = (max(0, -top_left[0]), max(0, bot_left[0]-(self.input_shape[2] - 1)))
        padded_crop = nn.functional.pad(crop, hpad+vpad)
        return padded_crop

    def _get_or_set_crop(self, x:torch.Tensor, w:int, loc_idx:Tuple[int], mode:Literal['get','set'], pad:bool=False, set_val:float = 0):
        top_left = self._get_top_left_coord(w, loc_idx)
        pos_top_left = np.maximum(0, top_left)
        if mode == 'set':
            x[:,:, pos_top_left[0]:top_left[0]+w][:,:,:,pos_top_left[1]:top_left[1]+w] = set_val
        elif mode == 'get':
            crop = x[:,:, pos_top_left[0]:top_left[0]+w][:,:,:,pos_top_left[1]:top_left[1]+w]
            if pad:
                crop = self._pad_crop(crop, w, loc_idx)
            return crop
        else:
            raise ValueError('mode={mode} but must be either "get" or "set"')

    def _get_patch_emb(self, crop, conv, rf, ax):
        grey_crop = torch.repeat_interleave(crop.mean(1, keepdims=True), 3, 1)
        crop = (1/rf)*crop + (1 - 1/rf)*grey_crop
        crop = self.normalization_layer(crop)
        if self.visualize:
            ax.imshow(convert_image_tensor_to_ndarray(crop[0]))
            ax.set_title(f'cropw={crop.shape[2]}\nkernw={rf}')
        pemb = conv(crop)
        if not self.params.place_crop_features_in_grid:
            pemb = rearrange(pemb, 'b c h w -> b (h w) c')
        return pemb
    
    def _place_crop_features_in_grid(self, pembs, loc_idx):
        pembs = pembs[::-1]
        g = pembs[0].clone()
        for pe, w in zip(pembs[1:], self.isobox_w[::-1]):
            self._get_or_set_crop(g, w, loc_idx, 'set', set_val=pe)
        return g

    def _forward_batch(self, img, loc_idx):
        if loc_idx is None:
            loc_idx = self._get_loc()

        if self.visualize:
            fig, axs = plt.subplots(1, len(self.isobox_w)+2, figsize=(20,5))
            ax0 = axs[0]
            img_arr = convert_image_tensor_to_ndarray(img[0])
            ax0.imshow(img_arr)
            ax0.scatter([loc_idx[1]], [loc_idx[0]])
        else:
            axs = [None]*(len(self.isobox_w)+2)

        masked_img = img
        patch_embs = []
        crops = []
        for w, conv, rf, ax in zip(self.isobox_w, self.convs, self.rec_flds, axs[1:]):
            crop = self._get_or_set_crop(masked_img, w, loc_idx, 'get', pad=True)
            if self.visualize:
                top_left = self._get_top_left_coord(w, loc_idx)
                rect = Rectangle((top_left[1]-0.5, top_left[0]-0.5), w, w, linewidth=1, edgecolor='r', facecolor='none')
                ax0.add_patch(rect)
            crops.append(crop)
            pemb = self._get_patch_emb(crop, conv, rf, ax)
            if self.params.mask_small_rf_region:
                # b = conv.bias.unsqueeze(0) if conv.bias is not None else 0
                # nzidx = torch.arange(0,pemb.shape[1])[(pemb[0] != b).any(-1)]
                # pemb = pemb[:, nzidx]
                self._get_or_set_crop(masked_img, w, loc_idx, 'set', set_val=0.)
            patch_embs.append(pemb)
        pemb = self._get_patch_emb(masked_img, self.convs[-1], self.rec_flds[-1], axs[-1])
        patch_embs.append(pemb)
        if not self.params.place_crop_features_in_grid:
            patch_embs = torch.cat(patch_embs, 1)
        else:
            patch_embs = self._place_crop_features_in_grid(patch_embs, loc_idx)
        # fig, axs = plt.subplots(1,len(crops)+2,figsize=(25, 5))
        # axs[0].scatter([loc_idx[1]], [loc_idx[0]])
        # axs[0].imshow(rearrange(img[0], 'c h w -> h w c').cpu().detach().numpy())
        # for c, ax in zip(crops, axs[1:]):
        #     ax.imshow(rearrange(c[0], 'c h w -> h w c').cpu().detach().numpy())
        # axs[-1].imshow(rearrange(masked_img[0], 'c h w -> h w c').cpu().detach().numpy())
        if self.visualize:
            plt.savefig('input.png')
        return patch_embs

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

class RetinaWarp(AbstractRetinaFilter):
    def __repr__(self):
        return f'RetinaWarp(loc_mode={self.params.loc_mode})'
    def _forward_batch(self, x, loc_idx):
        loc_idx = loc_idx - np.array(x.shape[2:]) / 2
        warped = retina_pt.warp_image(x, max(x.shape[2:]), loc_idx)
        # i = warped[0].transpose(0,1).transpose(1,2).cpu().detach().numpy()
        # plt.figure()
        # ax = plt.subplot(1,2,1)
        # ax.imshow(i)
        # ax = plt.subplot(1,2,2)
        # ax.imshow(x[0].transpose(0,1).transpose(1,2).cpu().detach().numpy())
        # plt.savefig('input.png')
        # plt.close()
        return warped

class VOneBlock(AbstractModel):
    @define(slots=False)
    class ModelParams(BaseParameters):
        sf_corr: float = 0.75
        sf_max: int = 9
        sf_min: int = 0
        rand_param: bool = False
        gabor_seed: int = 0
        simple_channels: int = 256
        complex_channels: int = 256
        noise_mode: str = 'neuronal'
        noise_scale: float = 0.35
        noise_level: float = 0.07
        k_exc: int = 25
        image_size: int = 224
        visual_degrees: int = 8
        ksize: int = 25
        stride: int = 4
        model_arch: None = None
        dropout_p: float = 0.
        add_noise_during_inference: bool = False
        add_deterministic_noise_during_inference: bool = False

    
    def __init__(self, params: ModelParams) -> None:
        super().__init__(params)
        self.params = params
        kwargs_to_exclude = set(['cls', 'add_noise_during_inference', 'add_deterministic_noise_during_inference', 'dropout_p'])
        kwargs = params.asdict(filter=lambda a,v: a.name not in kwargs_to_exclude)
        print(kwargs)
        from adversarialML.biologically_inspired_models.vonenet.vonenet.vonenet import VOneNet
        voneblock = VOneNet(**kwargs)
        bottleneck = nn.Conv2d(params.simple_channels+params.complex_channels, 64, kernel_size=1, stride=1, bias=False)
        self.voneblock = nn.Sequential(
            voneblock,
            bottleneck
        )
        if self.params.dropout_p > 0:
            self.voneblock.add_module('dropout_0',nn.Dropout(self.params.dropout_p))
        v1block = self.voneblock[0]
        noise_sz = (1, v1block.out_channels, int(v1block.input_size/v1block.stride),
                                 int(v1block.input_size/v1block.stride))
        self.detnoise = torch.empty(*noise_sz,).normal_(generator=torch.Generator().manual_seed(51972691))

    
    def forward(self, x):
        if (not self.training) and (not self.params.add_noise_during_inference):
                if self.params.add_deterministic_noise_during_inference:
                    noise = self.detnoise.repeat_interleave(x.shape[0], dim=0).to(x.device)
                    self.voneblock[0].fixed_noise = noise
                else:
                    self.voneblock[0].noise_mode = None
        else:
            self.voneblock[0].noise_mode = self.params.noise_mode
            self.voneblock[0].fixed_noise = None
        return self.voneblock(x)

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

        