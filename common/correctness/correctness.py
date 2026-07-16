import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def _count_signed_zero_pairs(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return 0
    mask_a = (a == 0.0) & (a.sign() < 0)
    mask_b = (b == 0.0) & (b.sign() < 0)
    diff = mask_a ^ mask_b
    return int(diff.sum().item())


def check_correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 0.0,
    atol: float = 0.0,
    require_bitwise: bool = True,
    label: str = "",
) -> Dict:
    """Compare output vs reference tensor.

    Performs a multi-level correctness check:
      1. Bitwise equality (exact match)
      2. Signed-zero bitwise differences
      3. Numeric equality within tolerance (rtol / atol)
      4. NaN / Inf presence accounting

    Parameters
    ----------
    output : torch.Tensor
        Tensor produced by the implementation under test.
    reference : torch.Tensor
        Reference tensor (usually from torch_npu).
    rtol : float
        Relative tolerance for numeric comparison.
    atol : float
        Absolute tolerance for numeric comparison.
    require_bitwise : bool
        If True, PASS requires exact bitwise equality.
    label : str
        Optional label for reporting context.

    Returns
    -------
    Dict
        - status (str): ``"PASS"`` or ``"FAIL"``.
        - bitwise_equal (bool): exact bitwise match including sign of zero.
        - bitwise_mismatch_count (int): number of positions that differ.
        - signed_zero_mismatch_count (int): positions where sign of zero
          is the only difference.
        - numeric_mismatch_count (int): positions that differ beyond a
          signed-zero discrepancy.
        - max_abs_diff (float): maximum absolute difference.
        - max_rel_diff (float): maximum relative difference.
        - nan_count (int): number of NaN values in output.
        - inf_count (int): number of Inf values in output.

    Raises
    ------
    TypeError
        If *output* or *reference* are not ``torch.Tensor``.
    ValueError
        If shapes are mismatched.
    """
    if not isinstance(output, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("Both output and reference must be torch.Tensor")
    if output.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: output {tuple(output.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    if output.dtype != reference.dtype:
        raise ValueError(
            f"dtype mismatch: output {output.dtype} vs "
            f"reference {reference.dtype}"
        )
    if output.device != reference.device:
        raise ValueError(
            f"device mismatch: output {output.device} vs "
            f"reference {reference.device}"
        )

    if output.dtype == torch.float16:
        output_cpu = output.cpu()
        ref_cpu = reference.cpu()
        bitwise_equal = torch.equal(output_cpu.view(torch.uint16), ref_cpu.view(torch.uint16))
    else:
        bitwise_equal = torch.equal(output, reference)

    if output.dtype in (torch.bool, torch.uint8, torch.int8):
        bitwise_mismatch_cpu = (output.cpu() != reference.cpu())
        bitwise_mismatch_count = int(bitwise_mismatch_cpu.sum().item())
    elif output.dtype == torch.float16:
        bitwise_mismatch_cpu = (output_cpu.view(torch.uint16) != ref_cpu.view(torch.uint16))
        bitwise_mismatch_count = int(bitwise_mismatch_cpu.sum().item())
    else:
        bitwise_mismatch = output != reference
        bitwise_mismatch_count = int(bitwise_mismatch.sum().item())

    signed_zero_mismatch_count = _count_signed_zero_pairs(output, reference)

    if output.dtype in (torch.bool, torch.uint8, torch.int8):
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        nan_count = 0
        inf_count = 0
        numeric_mismatch_count = bitwise_mismatch_count
    else:
        abs_diff = (output - reference).abs()
        max_abs_diff = 0.0 if bitwise_equal else float(abs_diff.max().item())
        rel_diff = abs_diff / (reference.abs() + 1e-12)
        max_rel_diff = 0.0 if bitwise_equal else float(rel_diff.max().item())
        nan_count = int(torch.isnan(output).sum().item())
        inf_count = int(torch.isinf(output).sum().item())
        numeric_mismatch_count = bitwise_mismatch_count - signed_zero_mismatch_count

    if require_bitwise:
        passed = bitwise_equal
    else:
        numerically_close = torch.allclose(
            output.to(torch.float64),
            reference.to(torch.float64),
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        passed = numeric_mismatch_count == 0 or numerically_close

    return {
        "status": "PASS" if passed else "FAIL",
        "bitwise_equal": bitwise_equal,
        "bitwise_mismatch_count": bitwise_mismatch_count,
        "signed_zero_mismatch_count": signed_zero_mismatch_count,
        "numeric_mismatch_count": numeric_mismatch_count,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "label": label,
    }
