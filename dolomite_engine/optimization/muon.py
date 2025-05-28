import os
import math
import torch
from .matmul_muon_triton import matmul_transpose_assign
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor import DTensor
from ..utils import print_rank_0


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T


    # buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    # buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        # matmul_transpose_assign(X, buf1)
        # matmul_transpose_assign(buf1, buf2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng

        # B = b * buf1 + c * buf2
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

# @torch.compile(mode="max-autotune")

@torch.compile
def zeropower_via_newtonschulz5_batched(G: torch.Tensor, steps: int):
    a, b, c = (3.4445, -4.7750, 2.0315)
    transpose = G.size(1) > G.size(2)
    if transpose:
        G = G.transpose(1, 2)
    norms = G.norm(dim=(1, 2), keepdim=True) + 1e-7
    X = G / norms
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))
        AA = torch.bmm(A, A)
        B = b * A + c * AA
        X = a * X + torch.bmm(B, X)
    if transpose:
        X = X.transpose(1, 2)
    return X
 
# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape,is_expert=False):
    
    if is_expert:
        A,B = param_shape[1],param_shape[2]
    else:
        if len(param_shape)==3: # Conv1d
            A,B = param_shape[0], param_shape[1]*param_shape[2]
        else:
            A, B = param_shape[:2]
    
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        betas: The betas for the internal AdamW.
        eps: The epsilon for the internal AdamW.
        wd: The weight decay for all params.
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
    """

    def __init__(
        self,
        lr=1e-3,
        weight_decay=0.1,
        muon_params=None,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=5,
        adamw_params=None,
        betas=[0.9, 0.95],
        eps=1e-8,
        muon_matched_adamw_rms=0.2,
    ):

        defaults = dict(
            lr=lr,
            wd=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            adamw_betas=betas,
            adamw_eps=eps,
            matched_adamw_rms=muon_matched_adamw_rms,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is = 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2 or p._is_expert_weight, p.ndim
            # print_rank_0(p.ndim,p._is_expert_weight)
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False


    @torch.no_grad()    
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            matched_adamw_rms = group["matched_adamw_rms"]
            # if grad_clip_norm is not None:
        
            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                mup_scale = getattr(p, "_mup_scale", None)

                if mup_scale is not None:
                    lr = lr /mup_scale # For Mup 

                if getattr(p, "_is_expert_weight", False):
                    # Calculate update for each expert
                    state = self.state[p] #TODO: Check if p is also correct here if we are using  to_local

                    if "momentum_buffer" not in state:
                        # state["momentum_buffer"] = [torch.zeros_like(expert_grads_[i]) for i in range(num_experts)]                    
                        state["momentum_buffer"] = torch.zeros_like(g)        
                    
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum) 
                    else:
                        g = buf

                    met_data_dtensor = None
                    if isinstance(g, DTensor):
                        met_data_dtensor = dict(
                            placements=g.placements,
                            device_mesh=g.device_mesh,
                        )
                        # g = g.full_tensor() # 64E 256 GPU
                        g = g.to_local() # NOTE: Already gradients are reduced before .step
                    
                    u = zeropower_via_newtonschulz5_batched(g, steps=group["ns_steps"])

                    if met_data_dtensor is not None:
                        u = DTensor.from_local(u, device_mesh=met_data_dtensor["device_mesh"], placements=met_data_dtensor["placements"])
                        

                    # scale update
                    adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms,p.shape,is_expert=True)
                    # Apply weight decay
                    p.data.mul_(1 - lr * wd)

                    # Apply update for the expert weight
                    p.data.add_(u, alpha=-adjusted_lr)

                else:
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None

                    # calc update
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf


                    met_data_dtensor = None
                    if isinstance(g, DTensor):
                        met_data_dtensor = dict(
                            placements=g.placements,
                            device_mesh=g.device_mesh,
                        )
                        g = g.full_tensor() # TODO: Scat-Gather on one rank isntead of doing on all ranks 
                    
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    if met_data_dtensor is not None:
                        u=distribute_tensor(u, device_mesh=met_data_dtensor["device_mesh"], placements=met_data_dtensor["placements"])

                    # scale update
                    adjusted_lr = adjust_lr_wd_for_muon(lr,matched_adamw_rms, p.shape)

                    u = u.view(p.shape)
                    # apply weight decay
                    p.data.mul_(1 - lr * wd)

                    # apply update
                    p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]

            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                
                mup_scale = getattr(p, "_mup_scale", None)
                if mup_scale is not None:
                    lr = lr /mup_scale # For Mup 
                
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss