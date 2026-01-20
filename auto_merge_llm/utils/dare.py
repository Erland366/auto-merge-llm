"""
DARE (Drop And REscale) preprocessing for model merging.

Based on: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"
Paper: https://arxiv.org/abs/2311.03099

DARE reduces task vector interference by randomly dropping delta parameters and rescaling.
"""

import torch
from typing import Dict, Optional
import copy


def apply_dare_to_tensor(
    delta: torch.Tensor,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply DARE to a single delta tensor.

    Args:
        delta: The delta tensor (finetuned - pretrained)
        drop_rate: Probability of dropping each parameter (p), typically 0.9
        rescale: Whether to rescale remaining values by 1/(1-p)
        seed: Optional random seed for reproducibility

    Returns:
        Sparsified delta tensor
    """
    if drop_rate <= 0.0:
        return delta

    if drop_rate >= 1.0:
        return torch.zeros_like(delta)

    if seed is not None:
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(seed)
    else:
        generator = None

    # Create random mask: 1 = keep, 0 = drop
    # Probability of keeping = (1 - drop_rate)
    if generator is not None:
        mask = torch.bernoulli(
            torch.ones_like(delta) * (1 - drop_rate),
            generator=generator
        )
    else:
        mask = torch.bernoulli(torch.ones_like(delta) * (1 - drop_rate))

    # Apply mask
    sparse_delta = delta * mask

    # Rescale to preserve expected value: E[sparse_delta] = E[delta]
    if rescale:
        sparse_delta = sparse_delta / (1 - drop_rate)

    return sparse_delta


def apply_dare_to_task_vector(
    task_vector_param_dict: Dict[str, torch.Tensor],
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None,
    inplace: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Apply DARE to all parameters in a task vector dictionary.

    Args:
        task_vector_param_dict: Dict mapping param names to delta tensors
        drop_rate: Probability of dropping each parameter (p)
        rescale: Whether to rescale by 1/(1-p)
        seed: Optional random seed (each param gets seed + index for reproducibility)
        inplace: If True, modify dict in place; if False, create copy

    Returns:
        Dict with DARE-sparsified delta tensors
    """
    if drop_rate <= 0.0:
        return task_vector_param_dict if inplace else copy.deepcopy(task_vector_param_dict)

    if inplace:
        result = task_vector_param_dict
    else:
        result = {}

    for idx, (name, delta) in enumerate(task_vector_param_dict.items()):
        param_seed = (seed + idx) if seed is not None else None
        sparse_delta = apply_dare_to_tensor(
            delta=delta,
            drop_rate=drop_rate,
            rescale=rescale,
            seed=param_seed
        )
        result[name] = sparse_delta

    return result


def apply_dare_to_flattened_params(
    flattened_params: torch.Tensor,
    drop_rate: float = 0.9,
    rescale: bool = True,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply DARE to flattened parameter tensor(s).

    Args:
        flattened_params: Tensor of shape (num_models, num_params) or (num_params,)
        drop_rate: Probability of dropping each parameter
        rescale: Whether to rescale by 1/(1-p)
        seed: Optional random seed

    Returns:
        DARE-sparsified flattened params
    """
    if drop_rate <= 0.0:
        return flattened_params

    if drop_rate >= 1.0:
        return torch.zeros_like(flattened_params)

    if seed is not None:
        generator = torch.Generator(device=flattened_params.device)
        generator.manual_seed(seed)
    else:
        generator = None

    if generator is not None:
        mask = torch.bernoulli(
            torch.ones_like(flattened_params) * (1 - drop_rate),
            generator=generator
        )
    else:
        mask = torch.bernoulli(torch.ones_like(flattened_params) * (1 - drop_rate))

    sparse_params = flattened_params * mask

    if rescale:
        sparse_params = sparse_params / (1 - drop_rate)

    return sparse_params


class DAREConfig:
    """Configuration for DARE preprocessing."""

    def __init__(
        self,
        enabled: bool = False,
        drop_rate: float = 0.9,
        rescale: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            enabled: Whether DARE is enabled
            drop_rate: Probability of dropping (0.0-1.0), paper recommends 0.9
            rescale: Whether to rescale remaining params by 1/(1-p)
            seed: Random seed for reproducibility
        """
        self.enabled = enabled
        self.drop_rate = drop_rate
        self.rescale = rescale
        self.seed = seed

    def __repr__(self):
        return (
            f"DAREConfig(enabled={self.enabled}, drop_rate={self.drop_rate}, "
            f"rescale={self.rescale}, seed={self.seed})"
        )

    @classmethod
    def from_method_params(cls, method_params: dict) -> "DAREConfig":
        """Create DAREConfig from method_params dict."""
        return cls(
            enabled=method_params.get("dare_enabled", False),
            drop_rate=method_params.get("dare_drop_rate", 0.9),
            rescale=method_params.get("dare_rescale", True),
            seed=method_params.get("dare_seed", None)
        )
