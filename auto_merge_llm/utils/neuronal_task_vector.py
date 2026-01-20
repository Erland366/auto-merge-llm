"""
Neuronal Task Vector utilities for NeuroMerging.

Implements subspace decomposition of task vectors into parallel and orthogonal
components relative to pretrained neuron weights, as described in the
NeuroMerging paper (Fang et al., 2025).

Key concepts:
- Parallel subspace (P): captures input sensitivity adjustments
- Orthogonal subspace (O): captures novel task-specific adaptations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import re


# Patterns for identifying neuronal parameters (dense layer weights)
# These are the parameters where we apply neuronal decomposition
NEURONAL_PARAM_PATTERNS = [
    r'.*\.attention\..*\.(query|key|value)\.weight$',
    r'.*\.attention\.output\.dense\.weight$',
    r'.*\.intermediate\.dense\.weight$',
    r'.*\.output\.dense\.weight$',
    # For BERT/RoBERTa-style models
    r'.*self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$',
    r'.*mlp\.(fc1|fc2|gate_proj|up_proj|down_proj)\.weight$',
]


def is_neuronal_param(param_name: str) -> bool:
    """
    Check if a parameter is a neuronal parameter (dense layer weight).

    Neuronal parameters are weight matrices where each row represents
    the incoming connections to a neuron. These are decomposed into
    parallel and orthogonal subspaces in NeuroMerging.

    Non-neuronal parameters (biases, LayerNorm, embeddings) use standard merging.
    """
    for pattern in NEURONAL_PARAM_PATTERNS:
        if re.match(pattern, param_name):
            return True
    return False


def decompose_neuron_row(
    pretrained_row: torch.Tensor,
    task_vector_row: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a single neuron's task vector into parallel and orthogonal components.

    Given:
    - w_0: pretrained neuron weight row (shape: [in_features])
    - tau: task vector row = w_finetuned - w_pretrained (shape: [in_features])

    Computes:
    - tau_parallel = (w_0 · tau) / ||w_0||^2 * w_0  (projection onto w_0)
    - tau_orthogonal = tau - tau_parallel  (orthogonal complement)

    Args:
        pretrained_row: Pretrained neuron weight row w_0
        task_vector_row: Task vector tau = w_t - w_0
        eps: Small constant for numerical stability

    Returns:
        Tuple of (tau_parallel, tau_orthogonal)
    """
    # Compute ||w_0||^2
    norm_sq = torch.sum(pretrained_row * pretrained_row) + eps

    # Compute projection coefficient: (w_0 · tau) / ||w_0||^2
    proj_coef = torch.sum(pretrained_row * task_vector_row) / norm_sq

    # tau_parallel = projection coefficient * w_0
    tau_parallel = proj_coef * pretrained_row

    # tau_orthogonal = tau - tau_parallel
    tau_orthogonal = task_vector_row - tau_parallel

    return tau_parallel, tau_orthogonal


def decompose_weight_matrix(
    pretrained_weight: torch.Tensor,
    task_vector_weight: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a weight matrix's task vector into parallel and orthogonal components.

    For a weight matrix W of shape [out_features, in_features], each row k
    represents a neuron's incoming connections. We decompose each row's
    task vector independently.

    Args:
        pretrained_weight: Pretrained weight matrix W_0 [out_features, in_features]
        task_vector_weight: Task vector tau = W_t - W_0 [out_features, in_features]
        eps: Small constant for numerical stability

    Returns:
        Tuple of (tau_parallel, tau_orthogonal), each [out_features, in_features]
    """
    # Handle different tensor shapes
    if pretrained_weight.dim() == 1:
        # Single neuron (bias-like), though biases shouldn't be decomposed
        return decompose_neuron_row(pretrained_weight, task_vector_weight, eps)

    # For 2D weight matrices [out_features, in_features]
    # Process all rows at once using broadcasting

    # Compute ||w_0||^2 for each row [out_features]
    norm_sq = torch.sum(pretrained_weight * pretrained_weight, dim=-1, keepdim=True) + eps

    # Compute (w_0 · tau) for each row [out_features]
    dot_product = torch.sum(pretrained_weight * task_vector_weight, dim=-1, keepdim=True)

    # Projection coefficient for each row
    proj_coef = dot_product / norm_sq

    # tau_parallel = projection coefficient * w_0
    tau_parallel = proj_coef * pretrained_weight

    # tau_orthogonal = tau - tau_parallel
    tau_orthogonal = task_vector_weight - tau_parallel

    return tau_parallel, tau_orthogonal


class NeuronalTaskVector:
    """
    Neuronal task vector with subspace decomposition.

    Decomposes task-specific weight changes into:
    - Parallel subspace: adjustments in the direction of pretrained weights
    - Orthogonal subspace: novel task-specific adaptations

    For non-neuronal parameters (biases, LayerNorm, embeddings), the full
    task vector is stored without decomposition.
    """

    def __init__(
        self,
        pretrained_model=None,
        finetuned_model=None,
        exclude_param_names_regex=None,
        task_vector_param_dict: Optional[Dict[str, torch.Tensor]] = None,
        pretrained_param_dict: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize neuronal task vector.

        Args:
            pretrained_model: Base pretrained model
            finetuned_model: Fine-tuned model
            exclude_param_names_regex: Patterns to exclude from merging
            task_vector_param_dict: Pre-computed task vector dict (optional)
            pretrained_param_dict: Pre-computed pretrained params (optional)
        """
        # Store decomposed components
        self.parallel_param_dict: Dict[str, torch.Tensor] = {}
        self.orthogonal_param_dict: Dict[str, torch.Tensor] = {}
        self.non_neuronal_param_dict: Dict[str, torch.Tensor] = {}

        # Store pretrained weights for reconstruction
        self.pretrained_param_dict: Dict[str, torch.Tensor] = {}

        # Track which params are neuronal
        self.neuronal_params: List[str] = []
        self.non_neuronal_params: List[str] = []

        if task_vector_param_dict is not None and pretrained_param_dict is not None:
            # Use pre-computed dicts
            self._decompose_from_dicts(task_vector_param_dict, pretrained_param_dict)
        elif pretrained_model is not None and finetuned_model is not None:
            # Compute from models
            self._compute_from_models(
                pretrained_model, finetuned_model, exclude_param_names_regex
            )

    def _decompose_from_dicts(
        self,
        task_vector_dict: Dict[str, torch.Tensor],
        pretrained_dict: Dict[str, torch.Tensor]
    ):
        """Decompose task vectors using pre-computed dicts."""
        for param_name, task_vec in task_vector_dict.items():
            pretrained_weight = pretrained_dict.get(param_name)
            if pretrained_weight is None:
                # No pretrained weight, store as non-neuronal
                self.non_neuronal_param_dict[param_name] = task_vec
                self.non_neuronal_params.append(param_name)
                continue

            self.pretrained_param_dict[param_name] = pretrained_weight

            if is_neuronal_param(param_name) and task_vec.dim() >= 2:
                # Neuronal parameter: decompose
                tau_parallel, tau_orthogonal = decompose_weight_matrix(
                    pretrained_weight, task_vec
                )
                self.parallel_param_dict[param_name] = tau_parallel
                self.orthogonal_param_dict[param_name] = tau_orthogonal
                self.neuronal_params.append(param_name)
            else:
                # Non-neuronal: store without decomposition
                self.non_neuronal_param_dict[param_name] = task_vec
                self.non_neuronal_params.append(param_name)

    def _compute_from_models(
        self,
        pretrained_model,
        finetuned_model,
        exclude_param_names_regex
    ):
        """Compute neuronal task vectors from models."""
        from .utils import get_param_names_to_merge

        pretrained_params = {
            name: param.data.clone()
            for name, param in pretrained_model.named_parameters()
        }
        finetuned_params = {
            name: param.data.clone()
            for name, param in finetuned_model.named_parameters()
        }

        param_names = get_param_names_to_merge(
            list(pretrained_params.keys()),
            exclude_param_names_regex
        )

        with torch.no_grad():
            for param_name in param_names:
                pretrained_weight = pretrained_params[param_name]
                finetuned_weight = finetuned_params[param_name]
                task_vec = finetuned_weight - pretrained_weight

                self.pretrained_param_dict[param_name] = pretrained_weight

                if is_neuronal_param(param_name) and task_vec.dim() >= 2:
                    # Neuronal parameter: decompose
                    tau_parallel, tau_orthogonal = decompose_weight_matrix(
                        pretrained_weight, task_vec
                    )
                    self.parallel_param_dict[param_name] = tau_parallel
                    self.orthogonal_param_dict[param_name] = tau_orthogonal
                    self.neuronal_params.append(param_name)
                else:
                    # Non-neuronal: store without decomposition
                    self.non_neuronal_param_dict[param_name] = task_vec
                    self.non_neuronal_params.append(param_name)

    def get_full_task_vector(self, param_name: str) -> torch.Tensor:
        """Get the full task vector for a parameter (parallel + orthogonal)."""
        if param_name in self.non_neuronal_param_dict:
            return self.non_neuronal_param_dict[param_name]
        elif param_name in self.parallel_param_dict:
            return self.parallel_param_dict[param_name] + self.orthogonal_param_dict[param_name]
        else:
            raise KeyError(f"Parameter {param_name} not found")

    def compute_l1_norm_ratio(self, mask_rate: float = 0.15) -> float:
        """
        Compute the L1-norm ratio of masked elements for lambda_2 estimation.

        sigma = ||tau_masked||_1 / ||tau||_1

        Args:
            mask_rate: Fraction of smallest-magnitude elements to mask

        Returns:
            L1-norm ratio sigma
        """
        # Collect all task vector elements
        all_elements = []
        for param_name in self.neuronal_params:
            full_tv = self.parallel_param_dict[param_name] + self.orthogonal_param_dict[param_name]
            all_elements.append(full_tv.flatten())
        for param_name in self.non_neuronal_params:
            all_elements.append(self.non_neuronal_param_dict[param_name].flatten())

        if not all_elements:
            return 0.0

        all_params = torch.cat(all_elements)
        total_l1 = all_params.abs().sum().item()

        if total_l1 < 1e-10:
            return 0.0

        # Find threshold for masking
        num_elements = all_params.numel()
        num_mask = int(num_elements * mask_rate)

        if num_mask == 0:
            return 0.0

        # Find k-th smallest magnitude
        kth_value = torch.kthvalue(all_params.abs().flatten(), num_mask).values.item()

        # Compute L1 norm of masked elements
        masked_l1 = all_params.abs()[all_params.abs() < kth_value].sum().item()

        return masked_l1 / total_l1

    @property
    def all_param_names(self) -> List[str]:
        """Get all parameter names."""
        return self.neuronal_params + self.non_neuronal_params
