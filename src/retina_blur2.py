from copy import deepcopy
from time import time
from typing import List, Union
import torch
from torch import nn
import torchvision
import numpy as np
from scipy.stats import laplace
from mllib.models.base_models import AbstractModel
from mllib.param import BaseParameters
from attrs import define
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from adversarialML.biologically_inspired_models.src.retina_preproc import AbstractRetinaFilter, gaussian_fn, seperable_gaussian_blur_pytorch, dist_to_prob, get_isodensity_box_width, convert_image_tensor_to_ndarray

class Rectangle:
    def __init__(self, x1, y1, x2, y2) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self) -> float:
        return abs(self.x2-self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2-self.y1)

    def translate(self, x, y):
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y
        return self
        
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1<x2 and y1<y2:
            return Rectangle(x1, y1, x2, y2)
    
    def draw(self, ax):
        w = abs(self.x1-self.x2)
        h = abs(self.y1-self.y2)
        rect = patches.Rectangle((self.x1-0.5, self.y1-0.5), w, h, linewidth=0.5, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    
    def __repr__(self) -> str:
        return f'Rectangle({self.x1}, {self.y1}, {self.x2}, {self.y2})'
    
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, Rectangle) and (all([self.x1 == __o.x1, self.x2 == __o.x2, self.y1 == __o.y1, self.y2 == __o.y2]))

def embed_and_foveate(vfwidth, loc_idx, img, isobox_w, avg_bins, kernels, gblur_fn=None):
    center = vfwidth//2
    # Cocentric squares defining regions of equal visual acuity (isoboxes). 
    # isobox_w contains the width of the squares.
    eboxes = [Rectangle(-w+center,w+center,w+center,-w+center) for w in isobox_w]
    # Rectangular region of the same size as the image placed at loc_idx in the visual field
    img_rect = Rectangle(loc_idx[1], loc_idx[0], loc_idx[1]+img.shape[3], loc_idx[0]+img.shape[2])
    # Intersecting sub-rectangles the image region and the isoboxes.
    intersections = [img_rect.intersection(r) for r in eboxes]
    # Remove superfluous sub-rectangles that arise if the image is smaller than the second largest
    # isobox. In this case the intersections of all the isoboxes that completely enclose the image
    # will be the same and we only need to blur the image based on the smallest such isobox.
    for i, r in enumerate(intersections[:][:-1]):
        if r == intersections[i+1]:
            intersections[i] = None

    bimg = torch.zeros_like(img)
    density_mat = torch.zeros_like(img)
    # Pad image to maintain spatial dimensions after blur
    i = 0
    while (intersections[i] is None):
        i += 1
    maxks = min(kernels[i].shape[-1], min(img.shape[2:])-1)
    pimg = torch.nn.functional.pad(img, (maxks,)*4, mode='reflect')
    for i, (r, K, p, w) in enumerate(zip(intersections, kernels, avg_bins, isobox_w)):
        if (r is not None):            
            ks = min(kernels[i].shape[-1], min(img.shape[2:])-1)
            hks = ks//2
            offset = (len(K) - ks)//2
            K = K[offset:offset+ks]
            # Translate the image rectangle so that it corresponds to 
            # corrdinates on the image instead of on the visual field.
            r = r.translate(-(loc_idx[1]-maxks), -(loc_idx[0]-maxks))
            if (torch.numel(K[K >= 1e-4]) > 1):
                crop = pimg[..., r.y1-hks:r.y2+hks, r.x1-hks:r.x2+hks]
                bcrop = gblur_fn(crop, K)
            else:
                bcrop = pimg[..., r.y1:r.y2, r.x1:r.x2]
            # print(w, kernels[i].shape, K.shape, img.shape, pimg.shape, crop.shape, bcrop.shape)
            r = r.translate(-maxks, -maxks)
            bimg[..., r.y1:r.y1+bcrop.shape[2], r.x1:r.x1+bcrop.shape[3]] = bcrop
            density_mat[..., r.y1:r.y1+bcrop.shape[2], r.x1:r.x1+bcrop.shape[3]] = p
    return bimg, density_mat

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
    
    def forward(self, x: torch.Tensor):
        return x.mean(1, keepdim=True)

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

    def __init__(self, params) -> None:
        super().__init__(params)
        self.std = self.params.std
        if self.params.add_deterministic_noise_during_inference:
            self.register_buffer('noise_patch', torch.empty(self.params.max_input_size).normal_(0, self.std))
    
    def __repr__(self):
        return f"GaussianNoiseLayer(std={self.std})"
    
    def forward(self, img):
        if self.training:
            d = torch.empty_like(img).normal_(0, self.std)
        else:
            if self.params.add_deterministic_noise_during_inference:
                b, c, h ,w = img.shape
                d = self.noise_patch[:c, :h, :w].unsqueeze(0)
            else:
                d = 0
        return img + d

    def compute_loss(self, x, y, return_logits=True):
        out = self.forward(x)
        logits = out
        loss = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        if return_logits:
            return logits, loss
        else:
            return loss

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
        max_res: int = np.inf
        min_res: int = -np.inf

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
        self.kernel_size = min(int(np.ceil(4*self.prob2std(0))), max_kernel_size)
        half_fovea_width = img_width//100

        d = np.arange(img_width/2)/(img_width/2)
        self.cone_density = dist_to_prob(d, self.cone_std)
        self.cone_density[-half_fovea_width:] = 1.
        clr_isobox_w, clr_avg_bins, bins = get_isodensity_box_width(self.cone_density, 'auto', min_bincount=half_fovea_width)
        self.clr_isobox_w = clr_isobox_w
        self.clr_avg_bins = clr_avg_bins
        self.clr_avg_bins[-1] = 1.

        self.rod_density = (1 - dist_to_prob(d, self.rod_std)) * self.max_rod_density
        gry_isobox_w, gry_avg_bins, _ = get_isodensity_box_width(-self.rod_density, 'auto', min_bincount=half_fovea_width)
        self.gry_avg_bins = -gry_avg_bins
        self.gry_avg_bins[-1] = 0.
        self.gry_isobox_w = gry_isobox_w

        clr_stds = [self.prob2std(p) for p in self.clr_avg_bins]
        gry_stds = [self.prob2std(p) for p in self.gry_avg_bins]
        print(clr_isobox_w, clr_stds)
        print(gry_isobox_w, gry_stds)

        if isinstance(self.params.view_scale, int):
            self.view_scale = min(self.params.view_scale, len(self.clr_isobox_w), len(self.gry_isobox_w))
        else:
            self.view_scale = self.params.view_scale


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
        # return nn.ParameterList([nn.parameter.Parameter(gkern(self.kernel_size, s), requires_grad=False) for s in std_list])
        return nn.ParameterList([nn.parameter.Parameter(gaussian_fn(int(np.ceil(4*s)), std=s), requires_grad=False) for s in std_list])
        # return nn.Parameter(torch.stack([gaussian_fn(int(np.ceil(4*s)), std=s) for s in std_list], 0), requires_grad=False)
    
    def apply_kernel(self, img, isobox_w, avg_bins, loc_idx, kernels):
        # return _get_gaussian_filtered_image_and_density_mat_pytorch(img, isobox_w, avg_bins, loc_idx, 
        #                                                     kernels, self.kernel_size, blur=self.apply_blur,
        #                                                     gblur_fn=seperable_gaussian_blur_pytorch
        #                                                     )
        if self.apply_blur:
            return embed_and_foveate(self.input_shape[1], loc_idx, img, isobox_w, avg_bins, kernels, seperable_gaussian_blur_pytorch)
        else:
            return img, None
    def __repr__(self):
        return f'RetinaBlurFilter2(loc_mode={self.params.loc_mode}, cone_std={self.cone_std}, rod_std={self.rod_std}, max_rod_density={self.max_rod_density}, kernel_size={self.kernel_size}, view_scale={self.view_scale}, beta={self.scale}, blur={self.apply_blur}, desaturate={self.include_gry_img})'
    
    def prob2std(self, p):
        s = self.scale*(1-p) + 1e-5
        return s

    # def _get_view_scale(self, img):
    #     hwidths = [w for w in self.clr_isobox_w if ((2*w >= self.params.min_res) and (2*w <= self.params.max_res))]
    #     # print(hwidths)
    #     scales = 2*np.array(hwidths) / min(img.shape[2:])
    #     if 1. not in scales:
    #         scales = np.sort(np.append(scales, [1]))[::-1]
    #     if isinstance(self.view_scale, int):
    #         return scales[self.view_scale]
    #     else:
    #         if (self.view_scale is None):
    #             s = 1
    #         elif self.view_scale == 'random_uniform':                
    #             s = np.random.choice(scales)
    #         else:
    #             raise ValueError(f'view_scale must be either None or random_uniform but got {self.view_scale}')
    #         return s
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

    def _get_five_fixations(self, img):
        locs = np.array(super()._get_five_fixations(img))
        center = np.array(self.input_shape[1:]) // 2
        locs = center - locs
        return locs

    def _get_hscan_fixations(self, img):
        center = np.array(self.input_shape[1:]) // 2
        y = center[0]
        max_x = self.input_shape[2] - img.shape[3]
        xs = np.linspace(0, max_x, 5, dtype=int)
        locs = np.stack(([y]*5, xs), 1)
        return locs
    
    def _get_center_loc(self, img):
        center = np.array(self.input_shape[1:]) // 2
        img_shape = img.shape[1:]
        loc = (center[0] - img_shape[1]//2, center[1] - img_shape[2]//2)
        return loc

    def _get_random_loc(self, img):
        img_shape = img.shape[1:]
        loc = (np.random.randint(0, self.input_shape[1]-img_shape[1]), np.random.randint(0, self.input_shape[2]-img_shape[2]))
        return loc

    def _get_random2_loc(self, img):
        img_shape = img.shape[1:]
        center = np.array(self.input_shape[1:]) // 2
        flip = np.random.binomial(1,0.5)
        if flip == 1:
            loc = (center[0] - np.random.randint(0, img_shape[1]), center[1] - np.random.randint(0, img_shape[2]))
        else:
            loc = self._get_random_loc(img)
        return loc
    
    def forward(self, x, loc_idx=None):
        if self.params.loc_mode == 'hscan_fixations':
            locs = self._get_hscan_fixations(x)
            filtered = [self._forward_batch(x, loc) for loc in locs]
            filtered = torch.stack(filtered, dim=1)
            filtered = filtered.reshape(-1, *(filtered.shape[2:]))
        else:
            filtered = super().forward(x, loc_idx=loc_idx)
        return filtered

    def _get_loc(self, img):
        img_shape = img.shape[1:]
        if self.params.loc_mode == 'center':
            loc = self._get_center_loc(img)
        elif self.params.loc_mode == 'const':
            loc = self.params.loc
        elif (self.params.loc_mode in ['random_uniform', 'random_uniform_2']):
            if (self.input_shape[2]>img_shape[2]) and (self.input_shape[1] > img_shape[1]):
                if (self.params.loc_mode == 'random_uniform'):
                    loc = self._get_random_loc(img)
                elif (self.params.loc_mode == 'random_uniform_2'):
                    loc = self._get_random2_loc(img)
            else:
                loc = (0,0)            
        else:
            raise ValueError('params.loc_mode must be "center" or "random_uniform" or "random_uniform_2" or "const"')
        return loc

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

        # print(clr_isobox_w, self.clr_isobox_w)
        # print([self.prob2std(p) for p in clr_avg_bins], [self.prob2std(p) for p in self.clr_avg_bins])
        # print(gry_isobox_w, self.gry_isobox_w)
        # print([self.prob2std(p) for p in gry_avg_bins], [self.prob2std(p) for p in self.gry_avg_bins])
        # t0 = time()
        clr_filtered_img, cone_density_mat = self.apply_kernel(img, clr_isobox_w, clr_avg_bins, loc_idx, clr_kernels)
        if self.include_gry_img:
            grey_img = torch.repeat_interleave(img.mean(1, keepdims=True), 3, 1)
            gry_filtered_img, rod_density_mat = self.apply_kernel(grey_img, gry_isobox_w, gry_avg_bins, loc_idx, gry_kernels)

            final_img = (rod_density_mat*gry_filtered_img + cone_density_mat*clr_filtered_img) / (rod_density_mat+cone_density_mat)
        else:
            final_img = clr_filtered_img
        # print('time:',time() - t0)
        final_img = torch.clamp(final_img, 0, 1.)
        # nplots = 8
        # print(img[0].min(), img[0].max(), clr_filtered_img[0].min(), clr_filtered_img[0].max(), gry_filtered_img[0].min(), gry_filtered_img[0].max(), final_img[0].min(), final_img[0].max())
        # f = plt.figure(figsize=(20,nplots))
        # plt.subplot(1,nplots,1)
        # plt.title('original')
        # plt.imshow(convert_image_tensor_to_ndarray(img[0]))
        # ax = plt.subplot(1,nplots,2)
        # vfwidth = self.input_shape[1]
        # vf = torch.zeros((img.shape[1], vfwidth, vfwidth))
        # # print(loc_idx, vf.shape)
        # w = min(img.shape[3], vfwidth - loc_idx[1])
        # h = min(img.shape[2], vfwidth - loc_idx[0])
        # vf[..., loc_idx[0]:loc_idx[0]+img.shape[2], loc_idx[1]:loc_idx[1]+img.shape[3]] = img[0,:,:h,:w]
        # plt.imshow(convert_image_tensor_to_ndarray(vf))
        # center = vfwidth//2
        # for w in self.clr_isobox_w:        
        #     rect = patches.Rectangle((max(0,center-w)-0.5, max(0,center-w)-0.5), 2*w, 2*w, linewidth=0.5, edgecolor='red', facecolor='none')
        #     ax.add_patch(rect)
        # ax = plt.subplot(1,nplots,3)
        # plt.title('Cone Output')
        # plt.imshow(convert_image_tensor_to_ndarray(clr_filtered_img[0]))
        # plt.subplot(1,nplots,4)
        # plt.title('Rod Output')
        # plt.imshow(convert_image_tensor_to_ndarray(gry_filtered_img[0]))
        # plt.subplot(1,nplots,5)
        # plt.title('Combined Output')
        # plt.imshow(convert_image_tensor_to_ndarray(final_img[0]))
        # ax = plt.subplot(1,nplots,6)
        # # plt.imshow(cone_density_mat[0,0].cpu().detach().numpy())
        # CS = plt.contour(cone_density_mat[0,0].cpu().detach().numpy().T, levels=len(self.clr_avg_bins))
        # ax.clabel(CS, CS.levels, inline=True, fmt='%.2f')
        # # plt.plot(np.arange(img.shape[3]), cone_density_mat[0,0, loc_idx[0]].cpu().detach().numpy())
        # # plt.plot(np.arange(img.shape[3]), rod_density_mat[0,0, loc_idx[0]].cpu().detach().numpy())
        # ax = plt.subplot(1,nplots,7)
        # # plt.imshow(cone_density_mat[0,0].cpu().detach().numpy())
        # CS = plt.contour(rod_density_mat[0,0].cpu().detach().numpy().T, levels=len(self.gry_avg_bins))
        # ax.clabel(CS, CS.levels, inline=True, fmt='%.2f')
        # plt.subplot(1,nplots,8)
        # plt.plot(np.arange(img.shape[3]), cone_density_mat[0,0, img.shape[3]//2].cpu().detach().numpy())
        # plt.plot(np.arange(img.shape[3]), rod_density_mat[0,0, img.shape[3]//2].cpu().detach().numpy())
        # plt.tight_layout()
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