from itertools import product
import math
import time
from typing import Any, Callable, Literal
import torch
from torch import nn
import numpy as np
from copy import deepcopy

# from mllib.adversarial.lib.autoattack.autopgd_base import APGDAttack
from rblur.fixation_prediction.models import RetinaFilterWithFixationPrediction, unnormalized_gkern, _update_mask
from rblur.models import IdentityLayer
from torchattacks.attack import Attack
from torchattacks.attacks.apgd import APGD

class mAPGDAttack(APGD):
    def __init__(self, model, norm='Linf', eps=8 / 255, n_iter=100, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=0.75, verbose=False):
        super().__init__(model, norm, eps, n_iter, n_restarts, seed, loss, eot_iter, rho, verbose)

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        loss[ind_to_fool] = loss_curr[ind_to_fool]
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
            
            return adv, loss, acc
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best
        
    # def __init__(self, predict, n_iter=100, norm='Linf', n_restarts=1, eps=None, seed=0, loss='ce', eot_iter=1, rho=0.75, topk=None, verbose=False, device=None, use_largereps=False, is_tf_model=False, logger=None):
    #     super().__init__(predict, n_iter, norm, n_restarts, eps, seed, loss, eot_iter, rho, topk, verbose, device, use_largereps, is_tf_model, logger)
    #     self.targeted = (self.loss == 'ce-targeted')
    
    # def perturb(self, x, y=None, best_loss=False, x_init=None, return_loss=False):
    #     """
    #     :param x:           clean images
    #     :param y:           clean labels, if None we use the predicted labels
    #     :param best_loss:   if True the points attaining highest loss
    #                         are returned, otherwise adversarial examples
    #     """

    #     assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
    #     if not y is None and len(y.shape) == 0:
    #         x.unsqueeze_(0)
    #         y.unsqueeze_(0)
    #     self.init_hyperparam(x)

    #     x = x.detach().clone().float().to(self.device)
    #     if not self.is_tf_model:
    #         y_pred = self.model(x).max(1)[1]
    #     else:
    #         y_pred = self.model.predict(x).max(1)[1]
    #     if y is None:
    #         #y_pred = self.predict(x).max(1)[1]
    #         y = y_pred.detach().clone().long().to(self.device)
    #     else:
    #         y = y.detach().clone().long().to(self.device)

    #     adv = x.clone()
    #     if self.loss != 'ce-targeted':
    #         acc = y_pred == y
    #     else:
    #         acc = y_pred != y
    #     loss = -1e10 * torch.ones_like(acc).float()
    #     if self.verbose:
    #         print('-------------------------- ',
    #             'running {}-attack with epsilon {:.5f}'.format(
    #             self.norm, self.eps),
    #             '--------------------------')
    #         print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        
        
    #     if self.use_largereps:
    #         epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
    #         iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
    #             .4 * self.n_iter_orig]
    #         iters = [math.ceil(c) for c in iters]
    #         iters[-1] = self.n_iter_orig - sum(iters[:-1]) # make sure to use the given iterations
    #         if self.verbose:
    #             print('using schedule [{}x{}]'.format('+'.join([str(c
    #                 ) for c in epss]), '+'.join([str(c) for c in iters])))
        
    #     startt = time.time()
    #     if not best_loss:
    #         torch.random.manual_seed(self.seed)
    #         torch.cuda.random.manual_seed(self.seed)
            
    #         for counter in range(self.n_restarts):
    #             ind_to_fool = acc.nonzero().squeeze()
    #             if len(ind_to_fool.shape) == 0:
    #                 ind_to_fool = ind_to_fool.unsqueeze(0)
    #             if ind_to_fool.numel() != 0:
    #                 x_to_fool = x[ind_to_fool].clone()
    #                 y_to_fool = y[ind_to_fool].clone()
                    
                    
    #                 if not self.use_largereps:
    #                     res_curr = self.attack_single_run(x_to_fool, y_to_fool)
    #                 else:
    #                     res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
    #                 best_curr, acc_curr, loss_curr, adv_curr = res_curr
    #                 ind_curr = (acc_curr == 0).nonzero().squeeze()

    #                 acc[ind_to_fool[ind_curr]] = 0
    #                 adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
    #                 loss[ind_to_fool] = loss_curr[ind_to_fool]
                    
    #                 if self.verbose:
    #                     print('restart {} - robust accuracy: {:.2%}'.format(
    #                         counter, acc.float().mean()),
    #                         '- cum. time: {:.1f} s'.format(
    #                         time.time() - startt))
            
    #         return adv, loss, acc

    #     else:
    #         adv_best = x.detach().clone()
    #         loss_best = torch.ones([x.shape[0]]).to(
    #             self.device) * (-float('inf'))
    #         for counter in range(self.n_restarts):
    #             best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
    #             ind_curr = (loss_curr > loss_best).nonzero().squeeze()
    #             adv_best[ind_curr] = best_curr[ind_curr] + 0.
    #             loss_best[ind_curr] = loss_curr[ind_curr] + 0.

    #             if self.verbose:
    #                 print('restart {} - loss: {:.5f}'.format(
    #                     counter, loss_best.sum()))
            
    #         return adv_best, loss_best

class PGDFixationScanpathMapAttack(Attack):
    r"""
    CODE TAKEN FROM TORCHATTACKS: https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master/torchattacks
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True,
                 loss_fn: Callable = nn.CrossEntropyLoss,
                 targeted=True, retina = None):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.loss_fn = loss_fn
        self._targeted = targeted
        self.retina = retina

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        images = self.preprocess_fn(images)
        # labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -self.loss_fn(outputs, labels)
            else:
                cost = self.loss_fn(outputs, labels)
            
            print(f'step={i}\t loss={cost.cpu().detach().numpy()}')

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
def fixation_map_loss(fmaps, scanpaths):
    loss = 0
    for fmap, scanpath in zip(fmaps, scanpaths):
        for loc in scanpath:
            loss += fmap[..., loc[0], loc[1]]
    loss /= len(fmaps)
    return -loss

class FixationAwareAPGDAttack(APGD):
    def __init__(self, model: nn.Module, 
                 fixation_sampling_strategy: Literal['grid', 'random'] = 'random',
                 num_fixation_samples: int = 100,
                 fixation_selection_attack_num_steps: int = 10,
                 classification_attack_num_steps: int = 100,
                 norm='Linf', eps=8 / 255, steps=100, n_restarts=1,
                 seed=0, loss='ce', eot_iter=1, rho=0.75, verbose=True):
        super().__init__(model, norm, eps, steps, n_restarts, seed, loss, eot_iter, rho, verbose)
    # def __init__(self, model: nn.Module, 
    #              fixation_sampling_strategy: Literal['grid', 'random'] = 'random',
    #              num_fixation_samples: int = 100,
    #              fixation_selection_attack_num_steps: int = 10,
    #              fixation_map_attack_num_steps: int = 100,
    #              classification_attack_num_steps: int = 100,
    #              norm='Linf', eps=8/255, steps=100, n_restarts=1, 
    #              seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False
    #              ) -> None:
    
        # super().__init__()
        self.fixation_sampling_strategy = fixation_sampling_strategy
        self.num_fixation_samples = num_fixation_samples
        self.model = (model)
        self.rfmodule = self._pop_retina_fixation_module()
        self.retina = self.rfmodule.retina
        self.scanpath_length = self.rfmodule.params.num_eval_fixation_points
        self.fixation_selection_attack_num_steps = fixation_selection_attack_num_steps
        self.fixation_selection_attack = mAPGDAttack(self.model, n_iter=fixation_selection_attack_num_steps, norm=norm, eps=eps, verbose=False)

    def _pop_retina_fixation_module(self) -> RetinaFilterWithFixationPrediction:
        rfmodule = None
        for name,m in self.model.named_modules():
            if isinstance(m, RetinaFilterWithFixationPrediction):                
                rfmodule = m
                break
        assert rfmodule is not None
        # module_path = name.split('.')
        # submodule = self.model
        # for i, submodname in enumerate(module_path):
        #     if submodname.isnumeric():
        #         if i < (len(module_path)-1):
        #             submodule = submodule[int(submodname)]
        #         else:
        #             submodule[int(submodname)] = rfmodule.retina
        #     else:
        #         if i < (len(module_path)-1):
        #             submodule = getattr(submodule, submodname)
        #         else:
        #             setattr(submodule, submodname, rfmodule.retina)
        return rfmodule
    
    def apgd_attack_single_run(self, x_in, y_in, scanpath_in, criterion_indiv, steps=None):
        if steps is None:
            steps = self.steps
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        self.steps_2, self.steps_min, self.size_decr = max(int(0.22 * steps), 1), max(int(0.06 * steps), 1), max(int(0.03 * steps), 1)
        if self.verbose:
            print('parameters: ', steps, self.steps_2, self.steps_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([steps, x.shape[0]])
        loss_best_steps = torch.zeros([steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        # criterion_indiv = self._compute_scanpath_loss
        # if self.loss == 'ce':
        #     criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        # elif self.loss == 'dlr':
        #     criterion_indiv = self.dlr_loss
        # else:
        #     raise ValueError('unknowkn loss')
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv, logits = criterion_indiv(x_adv, scanpath_in, y)
                loss = loss_indiv.sum()
                    
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(steps):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    # print(step_size.detach().cpu().numpy(), torch.norm(grad.reshape(grad.shape[0], -1), dim=1).detach().cpu().numpy())
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv, logits = criterion_indiv(x_adv, scanpath_in, y)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            # x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            x_best_adv[loss_indiv > loss_best] = x_adv[loss_indiv > loss_best] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f} - Accuracy: {:.6f}'.format(i, loss_best.sum(), acc.float().mean()))
            
            ### check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0
                
                counter3 += 1
            
                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()
                    
                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()
                        
                        fl_oscillation = np.where(fl_oscillation)
                        
                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                        
                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)
              
        return x_best, acc, loss_best, x_best_adv
    
    def pgd_attack_single_run(self, images, labels, scanpath):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        adv_images = images.clone().detach()

        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Calculate loss
            cost = self._compute_scanpath_loss(adv_images, scanpath, labels)[0]

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            print('grad_norm =', torch.norm(grad))
            adv_images = adv_images.detach() + (self.eps // 40)*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return cost, adv_images
    
    def _initialize_mask_and_kernel(self, x, shape_std_ratio=10):
        h,w = x.shape[-2:]
        mask = np.ones((1,1,h,w))
        std = max(h,w) / shape_std_ratio
        ks = int(4*std) 
        ks += int((ks % 2) == 0)
        mask_kernel = 1-unnormalized_gkern(ks, std) + 1e-8
        mask_kernel = mask_kernel.cpu().detach().numpy()
        return mask, mask_kernel
    
    def _sample_fixation_locations(self, x):
        h,w = x.shape[-2:]
        if self.fixation_sampling_strategy == 'random':
            mask, mask_kernel = self._initialize_mask_and_kernel(x)
            mask *= 1/mask.size
            rng = np.random.Generator(np.random.PCG64())
            loc_idxs = []            
            for i in range(self.scanpath_length):
                flat_mask = mask.reshape(-1)
                idx = rng.choice(np.arange(flat_mask.shape[0]), p=flat_mask/flat_mask.sum())
                loc_idxs.append(idx)
                _update_mask(idx, [0], mask_kernel.shape[-1]//2, mask, mask_kernel)
            loc_idxs = np.array(loc_idxs)
            row_locs = loc_idxs // w
            col_locs = loc_idxs % w
            locs = list(zip(row_locs, col_locs))
        return locs
            
        # elif self.fixation_sampling_strategy == 'grid':
        #     n = np.sqrt(self.num_fixation_samples)
        #     assert n == int(n)

        #     col_locs = np.linspace(0, w, n, dtype=np.int32)
        #     row_locs = np.linspace(0, h, n, dtype=np.int32)
        #     locs = list(product(row_locs, col_locs))
        # return locs
    
    # def _select_fixation_locs(self, images, labels):
    #     all_scanpaths = []
    #     for i in range(len(images)):
    #         sp_loss = []
    #         scanpaths = []
    #         for j in range(self.num_fixation_samples):
    #             scanpath = self._sample_fixation_locations(images[i])
    #             scanpaths.append(scanpath)
                
    #             self.retina.params.loc_mode='const'
    #             self.retina.params.loc = [scanpath]
    #             blurred_images = self.retina(images[[i]])
    #             adv_images, loss, acc = self.fixation_selection_attack.perturb(blurred_images, labels[[i]])
    #             sp_loss.append(loss)
    #             if acc[0] == 0:
    #                 break
    #         # print(i,j,list(zip(scanpaths, sp_loss)))
    #         # if self.fixation_selection_attack.targeted:
    #         #     best_idx = np.argmin(sp_loss)
    #         # else:
    #         best_idx = np.argmax(sp_loss)
    #         all_scanpaths.append(scanpaths[best_idx])
    #     return all_scanpaths
    
    # def _select_fixation_locs(self, images, labels):
    #     all_scanpaths = []
    #     for i in range(len(images)):
    #         sp_loss = torch.zeros(self.num_fixation_samples, dtype=images.dtype, device=images.device)
    #         scanpaths = []
    #         for j in range(self.num_fixation_samples):
    #             scanpath = self._sample_fixation_locations(images[i])
    #             scanpaths.append(scanpath)
    #             x_best, acc, loss, x_best_adv = self.apgd_attack_single_run(images[[i]], labels[[i]], [scanpath],
    #                                 self._compute_scanpath_selection_loss, self.fixation_selection_attack_num_steps)
    #             sp_loss[j] = loss.sum()
    #             if acc[0] == 0:
    #                 break
    #         best_idx = sp_loss.argmax()
    #         all_scanpaths.append(scanpaths[best_idx])
    #     return all_scanpaths

    def _select_fixation_locs(self, images, labels):
        ind_to_fool = torch.ones_like(labels).nonzero().squeeze()
        acc = torch.ones_like(labels)
        loss = torch.ones_like(labels) * 1e-10
        scanpaths = [None]*len(labels)

        j = 0
        while (ind_to_fool.numel() > 0) and (j < self.num_fixation_samples):

            x_to_fool = images[ind_to_fool]
            y_to_fool = labels[ind_to_fool]

            curr_scanpaths = [self._sample_fixation_locations(x) for x in x_to_fool]
            print(curr_scanpaths)
            x_best, acc_curr, loss_curr, x_best_adv = self.apgd_attack_single_run(x_to_fool, y_to_fool, curr_scanpaths,
                                        self._compute_scanpath_selection_loss, self.fixation_selection_attack_num_steps)
            
            successful_idx = (acc_curr == 0).nonzero().squeeze()
            if len(successful_idx.shape) == 0: successful_idx = successful_idx.unsqueeze(0)

            loss_improved_idx = (loss_curr > loss[ind_to_fool]).nonzero().squeeze()
            if len(loss_improved_idx.shape) == 0: loss_improved_idx = loss_improved_idx.unsqueeze(0)

            acc[ind_to_fool[successful_idx]] = 0
            loss[ind_to_fool[loss_improved_idx]] = loss_curr[loss_improved_idx].clone()
            for i in loss_improved_idx:
                scanpaths[ind_to_fool[i]] = curr_scanpaths[i]
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            j += 1

        return scanpaths


    
    def _compute_scanpath_selection_loss(self, images, scanpaths, labels):
        self.retina.params.loc_mode='const'
        self.retina.params.loc = scanpaths
        blurred_images = self.retina(images)

        self.rfmodule.params.disable = True
        logits = self.model(blurred_images)
        loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        self.rfmodule.params.disable = False
        return loss, logits

    def _compute_scanpath_loss(self, images, scanpaths, labels):
        h,w = images.shape[-2:]
        scanpaths_ = [[(0,0)] + sp for sp in scanpaths]

        self.rfmodule.params.disable = False
        self.retina.params.batch_size = 1
        self.retina.params.loc_mode='const'

        blurry_images = []
        fixation_loss = torch.zeros(images.shape[0], dtype=images.dtype, device=images.device)
        for i in range(self.scanpath_length+1):
            curr_locs = [sp[i] for sp in scanpaths_]
            self.retina.params.loc = curr_locs
            x_blur = self.retina(images)
            if i > 0:
                blurry_images.append(x_blur)

            if i < self.scanpath_length:
                next_locs = [sp[i+1] for sp in scanpaths_]
                fixation_maps = self.rfmodule.fixation_model(x_blur)
                fixation_maps = torch.log_softmax(torch.flatten(fixation_maps, 2), -1).reshape(fixation_maps.shape)
                # print(fixation_maps.shape, torch.flatten(fixation_maps, 2).exp().sum(-1))
                for j,(fmap, loc) in enumerate(zip(fixation_maps, next_locs)):
                    mask, mask_kernel = self._initialize_mask_and_kernel(x_blur, 100)
                    _update_mask(loc[0]*w+loc[1], [0], mask_kernel.shape[-1]//2, mask, mask_kernel)
                    mask = torch.tensor(mask[0], dtype=fmap.dtype, device=fmap.device)
                    loc_loss = (fmap * (1-mask)).sum()
                    fixation_loss[j] += loc_loss / self.scanpath_length
                    # print(i,j,curr_locs[j], loc, loc_loss, fixation_loss[j])
                    # print(fixation_loss.shape, fmap.shape, mask.shape)
                    # print(i, j, fixation_maps.shape, fmap.shape, loc, curr_locs[j])
                    # fixation_loss[j] += fmap[..., loc[0], loc[1]].squeeze()

        blurry_images = torch.stack(blurry_images, 1)
        blurry_images = blurry_images.reshape(-1, *(blurry_images.shape[2:]))

        self.rfmodule.params.disable = False
        logits = self.model(images)
        classification_loss = nn.functional.cross_entropy(logits, labels)
        loss = (fixation_loss + classification_loss)
        # print(fixation_loss.cpu().detach().numpy(), classification_loss.cpu().detach().numpy(), loss.cpu().detach().numpy())
        return loss, logits


    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        self.rfmodule.params.disable = False

        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        target_scanpaths = torch.zeros((x_in.shape[0], self.scanpath_length, 2)).long()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()

                        self.rfmodule.params.disable = True
                        scanpaths_in = self._select_fixation_locs(x_to_fool, y_to_fool)
                        best_curr, acc_curr, loss_curr, adv_curr = self.apgd_attack_single_run(x_to_fool, y_to_fool, scanpaths_in, self._compute_scanpath_loss)

                        loss_improved_idx = (loss_curr > loss[ind_to_fool]).nonzero().squeeze() 
                        successful_idx = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[successful_idx]] = 0
                        adv[ind_to_fool[loss_improved_idx]] = adv_curr[loss_improved_idx].clone()
                        loss[ind_to_fool[loss_improved_idx]] = loss_curr[loss_improved_idx].clone()

                        scanpath_array = torch.LongTensor([[list(loc) for loc in sp] for sp in scanpaths_in])
                        target_scanpaths[ind_to_fool[loss_improved_idx]] = scanpath_array[loss_improved_idx]
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
                            print(loss, loss_curr, target_scanpaths, scanpath_array)
            
            return acc, loss, target_scanpaths, adv
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.apgd_attack_single_run(x, y, scanpath_in, criterion_indiv, steps=steps)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels, cheap=True)[-1]
        self.rfmodule.params.disable = False
        return adv_images


    # def perturb(self, images, labels):
    #     images = images.clone().detach().to(images.device)
    #     labels = labels.clone().detach().to(images.device)

    #     self.rfmodule.params.disable = True
    #     scanpaths = self._select_fixation_locs(images, labels)

    #     adv_images = self.apgd_attack_single_run(images, labels, scanpaths, self._compute_scanpath_loss)[-1]
    #     # adv_images, loss, acc = self.classification_attack.perturb(images, labels)
        
        
    #     return adv_images