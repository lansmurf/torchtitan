# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig
import gc
import math
import string
from typing import List
import random
from torch.optim import Optimizer

# consider split between PP and non-PP
def build_optimizers(model_parts, job_config: JobConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = job_config.optimizer.fused

        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }

        if name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        elif name == "PrecondScheduleSFPaLMSOAP":
            optimizer = PrecondScheduleSFPaLMSOAP(
                model.parameters(),
                lr=lr,
                beta=optimizer_kwargs["betas"][0],  # beta1
                beta2_scale=0.8,  # From PaLM paper
                eps=1e-8,
                weight_decay=optimizer_kwargs["weight_decay"],
                precondition_frequency=2,  # How often to update preconditioner
                max_precond_dim=2048,  # Max dimension for preconditioner matrices
                merge_dims=True,  # Merge dimensions for efficiency
                precondition_1d=False,  # Whether to precondition 1D tensors
                normalize_grads=False,
                data_format="channels_first",
                correct_bias=True,
                warmup_steps=1000,
                gradient_clip_val=0.1,
                precond_scheduler=(1/3, 9)  # Schedule parameters for preconditioner updates
            )
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])


def linear_warmup_linear_decay(
    warmup_steps: int, decay_steps: int, current_step: int
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def warmup_stable_decay(
    total_steps: int,
    warmup_fraction: float,
    stable_fraction: float,
    current_step: int
) -> float:
    """Modified WSD with +1 trick to avoid zero LR on first step"""
    warmup_steps = int(total_steps * warmup_fraction)
    stable_steps = int(total_steps * stable_fraction)
    decay_steps = total_steps - warmup_steps - stable_steps
    
    if current_step < warmup_steps:
        # Linear warmup with +1 trick
        current_step += 1  # Add 1 to avoid zero LR
        return float(current_step) / (warmup_steps + 1)
    elif current_step < (warmup_steps + stable_steps):
        return 1.0
    else:
        decay_step = current_step - warmup_steps - stable_steps
        return max(0.0, 1 - (decay_step / decay_steps))

def build_lr_schedulers(optimizers, job_config: JobConfig):
    def _build_lr_scheduler(optimizer):
        """Build WSD scheduler with fractional parameters"""
        total_steps = int(job_config.training.steps)
        warmup_fraction = float(job_config.training.warmup_fraction)
        stable_fraction = float(job_config.training.stable_fraction)
        
        # Validate fractions
        if warmup_fraction + stable_fraction >= 1.0:
            raise ValueError(
                f"warmup_fraction ({warmup_fraction}) + stable_fraction ({stable_fraction}) "
                "must be less than 1.0 to leave room for decay"
            )
        
        lr_lambda = functools.partial(
            warmup_stable_decay,
            total_steps,
            warmup_fraction,
            stable_fraction
        )
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    class SchedulersContainer:
        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for scheduler in self.schedulers:
                scheduler.step()

    return SchedulersContainer(
        [_build_lr_scheduler(optimizer) for optimizer in optimizers]
    )

#-------------------------
# OPTIMIZER UTILS (credits to https://github.com/ClashLuke/HeavyBall/blob/main/heavyball/utils.py)
#-------------------------

_mode = None

if _mode is None:
    def decorator(func):
        return func
else:
    decorator = torch.compile(fullgraph=False, dynamic=True, mode=_mode)

_einsum_base = string.ascii_lowercase + string.ascii_uppercase


def warmup(lr: float, step: int, warmup_steps: int):
    return lr * min(step / warmup_steps, 1)


def schedule_free_(lr: float, weight_lr_power: float, weight_sum: float, beta1: float, parameters: List[torch.Tensor],
                   z: List[torch.Tensor], grad: list[torch.Tensor]):
    weight = lr ** weight_lr_power
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    # These operations update y in-place,
    # without computing x explicitly.
    p32 = [promote(p) for p in parameters]
    z32 = [promote(z_) for z_ in z]
    torch._foreach_lerp_(p32, z32, weight=ckp1)
    torch._foreach_add_(p32, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))
    copy_stochastic_list_(parameters, p32)

    # z step
    torch._foreach_sub_(z, grad, alpha=lr)
    copy_stochastic_list_(z, z32)
    return weight_sum


def dim_merger(grad, max_precond_dim):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape >= max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)
    return new_grad


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta ** step)


def exp_avg_sq_(state, grad, beta2, eps):
    torch._foreach_mul_(state, beta2)
    torch._foreach_addcmul_(state, grad, grad, value=1 - beta2)
    denom = torch._foreach_sqrt(state)
    torch._foreach_maximum_(denom, eps)
    return denom


def adaptive_gradient_clipping_(parameters: List[torch.Tensor], gradients: List[torch.Tensor], clip_val: float,
                                minimum: float = 1e-3, eps: float = 1e-8):
    if clip_val <= 0:
        return
    p_norm = torch._foreach_norm(parameters)
    g_norm = torch._foreach_norm(gradients)
    torch._foreach_maximum_(p_norm, minimum)
    torch._foreach_maximum_(g_norm, eps)
    torch._foreach_div_(p_norm, g_norm)
    torch._foreach_mul_(p_norm, clip_val)
    torch._foreach_minimum_(p_norm, 1)
    torch._foreach_mul_(gradients, p_norm)


def set_(dst: torch.Tensor, src: torch.Tensor):
    if src.is_contiguous() and dst.is_contiguous() and src.dtype == dst.dtype:
        dst.set_(src)
    else:
        dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


@decorator
def get_orthogonal_matrix_QR(GG, Q, exp_avg_sq, max_precond_dim=10000, merge_dims=False):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    indices = []

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        del m
        est_eig = torch.einsum('ij,ij->j', o, tmp)
        del o
        sort_idx = torch.argsort(est_eig, descending=True)
        del est_eig
        indices.append(sort_idx)
        power_iter = tmp[:, sort_idx]
        set_(q, torch.linalg.qr(power_iter)[0])
        del power_iter

    exp_avg_sq_new = exp_avg_sq
    if merge_dims:
        exp_avg_sq_new = dim_merger(exp_avg_sq, max_precond_dim)

    indices = tuple(slice(None) if ind is None else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq_new.dim() - i - 1))  #
                    for i, ind in enumerate(indices))
    exp_avg_sq_new = exp_avg_sq_new[indices]

    if merge_dims:
        exp_avg_sq_new = exp_avg_sq_new.reshape(exp_avg_sq.shape)
    set_(exp_avg_sq, exp_avg_sq_new)


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, 'cpu'):
            if modifier is not None:
                m = m.to(modifier)
            try:
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device,
                                                                                                dtype=dtype)
                break
            except torch.OutOfMemoryError:
                pass
            except RuntimeError:  # failed to compute eigenvalues
                continue
            clean()
        else:
            raise RuntimeError("Failed to compute eigenvalues.")

        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    return final


@decorator
def compute_ggt(grad, GG, max_precond_dim, merge_dims, precondition_1d, beta):
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return

    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    for idx, sh in enumerate(grad.shape):
        if sh > max_precond_dim:
            continue
        b = _einsum_base[idx]
        g0 = _einsum_base[:grad.dim()]
        g1 = g0.replace(b, b.upper())
        outer_product = torch.einsum(f'{g0},{g1}->{b + b.upper()}', grad, grad)
        GG[idx].lerp_(promote(outer_product), 1 - beta)


def promote(x):
    if x is (torch.bfloat16, torch.float16):
        return torch.float32
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


def update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    compute_ggt(grad, state['GG'], max_precond_dim, merge_dims, precondition_1d, beta)
    if state['Q'] is None:
        state['Q'] = get_orthogonal_matrix(state['GG'])
    if update_precond:
        get_orthogonal_matrix_QR(state['GG'], state['Q'], state['exp_avg_sq'], max_precond_dim, merge_dims)


def init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['Q'] = None  # Will hold all the eigenbases of the preconditioner.
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append([])
            return
        state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype))
        return

    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    for sh in grad.shape:
        if sh > max_precond_dim:
            state['GG'].append([])
        else:
            state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))



@decorator
def project(grad, Q, merge_dims, max_precond_dim, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    original_shape = grad.shape
    if merge_dims:
        grad = dim_merger(grad, max_precond_dim)

    param = _einsum_base[:grad.dim()]
    preconditioners = ",".join(
        (g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if len(m) > 0)
    if preconditioners:
        out = ''.join(c.upper() if c.upper() in preconditioners else c for c in param)
        grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if len(q) > 0])
    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


class ScheduleFree(torch.optim.Optimizer):
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        z = promote(state['z'])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta'] if 'beta' in group else group['betas'][0]
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        z = promote(state['z'])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        copy_stochastic_(p.data, p32)
                group['train_mode'] = True

    def _step(self):
        raise NotImplementedError


def copy_stochastic_list_(target: List[torch.Tensor], source: List[torch.Tensor]):
    for t, s in zip(target, source):
        if t.dtype == torch.bfloat16:
            copy_stochastic_(t, s)
        elif t.data_ptr() != s.data_ptr():
            t.copy_(s)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    if target.data_ptr() == source.data_ptr():
        return

    """Taken as-is from https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905"""
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def update_param_(param: List[torch.Tensor], update: List[torch.Tensor], lr: float, decay: float,
                  add_fn: callable = None):
    param32 = [promote(p) for p in param]
    update32 = [promote(u) for u in update]
    if decay > 0:
        torch._foreach_mul_(param32, 1 - decay * lr)
    if add_fn is None:
        torch._foreach_add_(param32, update32, alpha=lr)
    else:
        add_fn(param32, update32, lr)
    copy_stochastic_list_(param, param32)


def precond_schedule(step, precond_scheduler, rng):
    precond_prob = max(step, 1) ** precond_scheduler[0]
    precond_prob = math.log10(precond_prob)
    precond_prob = precond_prob ** precond_scheduler[1] + 1
    precond_prob = 1 / precond_prob
    update_precond = rng.random() < precond_prob
    return update_precond

#-----------------
# OPTIMIZER DEFINITIONS (credits to heavyball)
#-----------------

class PrecondScheduleSFPaLMSOAP(Optimizer):
    """
    Implements PrecondScheduleSFPaLMSOAP algorithm.
    
    This combines:
    - SOAP: Improving and Stabilizing Shampoo using Adam
    - Schedule-Free optimization from "The Road Less Scheduled"
    - PaLM's beta2 scheduling
    - Preconditioner scheduling from "Preconditioned Stochastic Gradient Descent"
    """
    
    def __init__(self, params, lr=3e-3, beta=0.9, beta2_scale=0.8, eps=1e-8,
                 weight_decay=0.01, precondition_frequency=2, max_precond_dim=2048,
                 merge_dims=True, precondition_1d=False, normalize_grads=False,
                 data_format="channels_first", correct_bias=True, warmup_steps=1,
                 r=0.0, weight_lr_power=2.0, gradient_clip_val=0.1,
                 precond_scheduler=(1/3, 9), betas=(None, None)):
                 
        if betas[0] is not None:
            beta = betas[0]
            
        defaults = dict(
            lr=lr,
            beta=beta,
            beta2_scale=beta2_scale,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            normalize_grads=normalize_grads,
            correct_bias=correct_bias,
            warmup_steps=warmup_steps,
            r=r,
            weight_lr_power=weight_lr_power,
            train_mode=True,
            step=-1,
            weight_sum=0,
            gradient_clip_val=gradient_clip_val,
            precond_scheduler=precond_scheduler
        )
        
        super().__init__(params, defaults)
        self._data_format = data_format
        self.rng = random.Random(0x120983109)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            vals = []
            merge_dims = group['merge_dims']
            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']

            step = group['step'] = group.get("step", -1) + 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                p.grad = None
                vals.append((p, grad))

            if not vals:
                continue

            p_list, grad = zip(*vals)
            vals = []

            # Adaptive gradient clipping
            adaptive_gradient_clipping_(p_list, grad, group["gradient_clip_val"], eps=group["eps"])

            for p, grad in zip(p_list, grad):
                state = self.state[p]

                if "z" not in state:
                    state["z"] = torch.clone(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
                    init_preconditioner(grad, state, max_precond_dim, precondition_1d, merge_dims)
                    update_preconditioner(grad, state, max_precond_dim, merge_dims, 
                                       precondition_1d, 0, True)
                    continue

                # Project gradients to eigenbases
                grad_projected = project(grad, state['Q'], merge_dims, max_precond_dim, False)
                z, exp_avg_sq = state["z"], state["exp_avg_sq"]
                vals.append((p, grad, grad_projected, z, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, z, exp_avg_sq = zip(*vals)

            # PaLM's beta2 schedule
            beta2 = 1 - max(step, 1) ** -group['beta2_scale']
            old_debiased2 = beta_debias(beta2, step)

            # Update momentum and compute denom
            denom = exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, group['eps'])
            torch._foreach_div_(grad_projected, denom)

            # Preconditioner scheduling
            update_precond = precond_schedule(step, group['precond_scheduler'], self.rng)

            for p, g, gp in zip(p_list, grad, grad_projected):
                state = self.state[p]
                # Project back to original space
                set_(gp, project(gp, state['Q'], merge_dims, max_precond_dim, back=True))
                update_preconditioner(g, state, max_precond_dim, merge_dims, precondition_1d,
                                   old_debiased2, update_precond)

            # Weight decay
            if group["weight_decay"] > 0:
                torch._foreach_add_(grad, p_list, alpha=group["weight_decay"])

            # Learning rate warmup
            lr = warmup(group['lr'], step, group['warmup_steps'])
            
            # Schedule-free update
            group['weight_sum'] = schedule_free_(lr, group['weight_lr_power'],
                                              group['weight_sum'], group['beta'],
                                              p_list, z, grad_projected)

        return loss