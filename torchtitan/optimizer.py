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