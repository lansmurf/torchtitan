import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# Setting limit as 65536 as in LayerNorm tutorial for better performance
MAX_FUSED_SIZE = 65536 // 2

def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="none",  # Default to no reduction for manual accumulation
    softcap=None,
):
    dtype = _input.dtype
    device = _input.device

    BT, H = _input.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    
    # Use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

    # Get total non-ignore tokens but avoid .item() for CUDA sync
    total_n_non_ignore = (target != ignore_index).sum()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
            
        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]
        loss_1d_slice = loss_1d[start_idx:end_idx]
        n_non_ignore = (target_chunk != ignore_index).sum()

        logits_chunk = logits_chunk.float()
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # Calculate losses and gradients with kernel
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            loss_ptr=loss_1d_slice,
            z_loss_ptr=loss_1d_slice,  # Dummy ptr
            loss_stride=loss_1d_slice.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction="none",  # Always use none in kernel for manual reduction
            softcap=softcap if softcap is not None else 0.0,
            RETURN_Z_LOSS=0,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # Convert back to original dtype for gradient computation
        logits_chunk = logits_chunk.to(dtype)
        
        # For accumulating gradients, don't scale by n_non_ignore/total_n_non_ignore here
        # Let the training loop handle the scaling after accumulation
        grad_logits_chunk = logits_chunk
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=grad_logits_chunk.t(),
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,  # No scaling here
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=grad_logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,  # No scaling here
            )

    # Return based on reduction mode
    if reduction == "none":
        return loss_1d, total_n_non_ignore, grad_input, grad_weight, grad_bias
    elif reduction == "sum":
        return loss_1d.sum(), total_n_non_ignore, grad_input, grad_weight, grad_bias
    elif reduction == "mean":
        return loss_1d.sum() / total_n_non_ignore, total_n_non_ignore, grad_input, grad_weight, grad_bias
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")

def fused_linear_cross_entropy_backward(
    grad_output, grad_input, grad_weight, grad_bias
):
    # If cross entropy is the last layer, grad_output is 1.0
    if torch.ne(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V
            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V
            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    
    return grad_input, grad_weight, grad_bias

class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="none",  # Default to none for manual accumulation
        softcap=None,
    ):
        output = fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            bias,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            reduction,
            softcap,
        )
        
        if reduction == "none":
            loss, n_tokens, grad_input, grad_weight, grad_bias = output
        else:
            loss, n_tokens, grad_input, grad_weight, grad_bias = output
            
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        
        if reduction == "none":
            return loss, n_tokens
        else:
            return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) == 2:
            grad_output, _ = grad_outputs  # When reduction="none"
        else:
            grad_output = grad_outputs[0]
            
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (grad_input, grad_weight, None, grad_bias, None, None, None, None, None)