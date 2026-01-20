"""
NeuroMerging: Neuron-level model merging with subspace decomposition.

Implements the NeuroMerging algorithm from "To See a World in a Spark of Neuron:
Disentangling Multi-task Interference for Training-free Model Merging"
(Fang et al., 2025).

Key innovation: Decomposes task vectors into parallel and orthogonal subspaces
relative to pretrained neuron weights, then merges primarily in the orthogonal
subspace which preserves ~88% of task-specific capabilities.
"""

from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from auto_merge_llm.utils import TaskVector, logger
from auto_merge_llm.utils.neuronal_task_vector import (
    NeuronalTaskVector,
    is_neuronal_param,
    decompose_weight_matrix,
)
from .base_method import MergeMethod


class NeuroMerging(MergeMethod):
    """
    NeuroMerging: Neuron-level selective model merging.

    Algorithm:
    1. Create neuronal task vectors for each model
    2. Decompose into parallel (input sensitivity) and orthogonal (task adaptability) subspaces
    3. Apply magnitude masking (keep top k% by magnitude)
    4. Merge subspaces using elect+mean (TIES-style disjoint merge)
    5. Scale by lambda_1 (parallel) and lambda_2 (orthogonal)
    6. Combine with pretrained model

    Default configuration (no validation):
    - lambda_1 = 0 (ignore parallel subspace - has little impact)
    - lambda_2 = 1/(1-sigma) where sigma is L1-norm ratio of masked elements
    - mask_ratio = 0.15 (mask smallest 15% by magnitude)
    """

    def _mask_by_magnitude(
        self,
        param_dict: dict,
        mask_rate: float = 0.15
    ) -> dict:
        """
        Mask smallest-magnitude parameter values by setting them to zero.

        Args:
            param_dict: Dictionary of parameter tensors
            mask_rate: Fraction of smallest elements to mask (set to zero)

        Returns:
            Masked parameter dictionary
        """
        if mask_rate <= 0:
            return param_dict

        masked_dict = {}
        for name, tensor in param_dict.items():
            if tensor.numel() == 0:
                masked_dict[name] = tensor
                continue

            num_mask = int(tensor.numel() * mask_rate)
            if num_mask == 0:
                masked_dict[name] = tensor
                continue

            try:
                # Find threshold (k-th smallest magnitude)
                kth_value = torch.kthvalue(tensor.abs().flatten(), num_mask).values
                # Create mask: True for values to keep
                mask = tensor.abs() >= kth_value
                masked_dict[name] = tensor * mask
            except RuntimeError:
                # Fallback if kthvalue fails
                masked_dict[name] = tensor

        return masked_dict

    def _disjoint_merge(
        self,
        tensors: list,
        dim: int = 0
    ) -> torch.Tensor:
        """
        TIES-style disjoint merge (elect+mean).

        1. Compute majority sign across all tensors
        2. Keep only values whose sign matches the majority
        3. Average the kept values

        Args:
            tensors: List of tensors to merge
            dim: Dimension along which tensors are stacked

        Returns:
            Merged tensor
        """
        if len(tensors) == 1:
            return tensors[0]

        # Stack tensors [num_models, ...]
        stacked = torch.stack(tensors, dim=dim)

        # Compute sign of summed tensor to get majority sign
        sum_sign = torch.sign(stacked.sum(dim=dim))

        # Replace zeros with overall majority sign
        majority_sign = torch.sign(sum_sign.sum())
        if majority_sign == 0:
            majority_sign = 1.0
        sum_sign = torch.where(sum_sign == 0, majority_sign, sum_sign)

        # Create mask: keep values that match majority sign
        mask = (
            ((sum_sign.unsqueeze(dim) > 0) & (stacked > 0)) |
            ((sum_sign.unsqueeze(dim) < 0) & (stacked < 0))
        )

        # Apply mask and compute mean of non-zero preserved values
        preserved = stacked * mask
        num_preserved = (preserved != 0).sum(dim=dim).float().clamp(min=1.0)
        merged = preserved.sum(dim=dim) / num_preserved

        return merged

    def _merge_orthogonal_with_svd(
        self,
        orthogonal_tensors: list,
        num_tasks: int
    ) -> torch.Tensor:
        """
        Merge orthogonal subspace components using SVD for coordinate system.

        For multi-task merging, we use SVD to find principal directions in the
        orthogonal subspace, then apply disjoint merge along each direction.

        Args:
            orthogonal_tensors: List of orthogonal component tensors
            num_tasks: Number of tasks being merged

        Returns:
            Merged orthogonal component
        """
        if len(orthogonal_tensors) == 1:
            return orthogonal_tensors[0]

        # For 2D tensors [out_features, in_features]
        # Stack along new dimension [num_tasks, out_features, in_features]
        stacked = torch.stack(orthogonal_tensors, dim=0)

        if stacked.dim() < 3:
            # For 1D or simple cases, use direct disjoint merge
            return self._disjoint_merge(orthogonal_tensors, dim=0)

        num_tasks_actual, out_features, in_features = stacked.shape

        # For each neuron row, apply disjoint merge across tasks
        # This is more efficient than full SVD for large matrices
        merged = self._disjoint_merge(orthogonal_tensors, dim=0)

        return merged

    def _compute_lambda2(
        self,
        neuronal_tvs: list,
        mask_rate: float
    ) -> float:
        """
        Compute lambda_2 scaling factor based on L1-norm ratio.

        lambda_2 = 1 / (1 - sigma)
        where sigma = max over tasks of (||masked||_1 / ||full||_1)

        Args:
            neuronal_tvs: List of NeuronalTaskVector objects
            mask_rate: Masking rate used

        Returns:
            Scaling factor lambda_2
        """
        max_sigma = 0.0
        for ntv in neuronal_tvs:
            sigma = ntv.compute_l1_norm_ratio(mask_rate)
            max_sigma = max(max_sigma, sigma)

        # Clamp to avoid division by zero or negative values
        max_sigma = min(max_sigma, 0.95)

        lambda_2 = 1.0 / (1.0 - max_sigma + 1e-8)
        return lambda_2

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        """
        Merge models using NeuroMerging algorithm.

        Args:
            base_model: Base pretrained model path
            models_to_merge: List of fine-tuned model paths
            method_params: Dictionary containing:
                - scaling_coefficient: Overall scaling (default 1.0)
                - param_value_mask_rate: Fraction to mask (default 0.15)
                - lambda_1: Parallel subspace weight (default 0.0)
                - lambda_2: Orthogonal subspace weight (default: auto-computed)
            mask_merging: Optional additional masking config
            exclude_param_names_regex: Patterns to exclude from merging

        Returns:
            Dictionary with merged model, tokenizers
        """
        scaling_coefficient = method_params.get("scaling_coefficient", 1.0)
        mask_rate = method_params.get("param_value_mask_rate", 0.15)
        lambda_1 = method_params.get("lambda_1", 0.0)  # Default: ignore parallel
        lambda_2 = method_params.get("lambda_2", None)  # Auto-compute if None

        logger.info(f"NeuroMerging with mask_rate={mask_rate}, lambda_1={lambda_1}")

        # Load models
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, exclude_param_names_regex
        )
        base_model_obj = base_model_dict['model']

        # Get pretrained parameters
        pretrained_params = {
            name: param.data.clone().cpu()
            for name, param in base_model_obj.named_parameters()
        }

        # Create neuronal task vectors for each model
        neuronal_tvs = []
        for model_dict in merging_model_list:
            model = model_dict['model']
            task_vec_dict = {}
            for name, param in model.named_parameters():
                if name in pretrained_params:
                    task_vec_dict[name] = (param.data.clone().cpu() -
                                           pretrained_params[name])

            ntv = NeuronalTaskVector(
                task_vector_param_dict=task_vec_dict,
                pretrained_param_dict=pretrained_params
            )
            neuronal_tvs.append(ntv)

        # Auto-compute lambda_2 if not provided
        if lambda_2 is None:
            lambda_2 = self._compute_lambda2(neuronal_tvs, mask_rate)
            logger.info(f"Auto-computed lambda_2={lambda_2:.4f}")

        # Merge each parameter
        merged_params = {}

        # Get all parameter names
        all_param_names = set()
        for ntv in neuronal_tvs:
            all_param_names.update(ntv.all_param_names)

        with torch.no_grad():
            for param_name in all_param_names:
                # Check if neuronal or non-neuronal
                is_neuronal = param_name in neuronal_tvs[0].neuronal_params

                if is_neuronal:
                    # Collect parallel and orthogonal components from all models
                    parallel_list = []
                    orthogonal_list = []

                    for ntv in neuronal_tvs:
                        if param_name in ntv.parallel_param_dict:
                            parallel_list.append(ntv.parallel_param_dict[param_name])
                            orthogonal_list.append(ntv.orthogonal_param_dict[param_name])

                    if not parallel_list:
                        continue

                    # Apply magnitude masking to full task vectors
                    # (This matches TIES behavior of masking before merge)
                    full_tvs = [p + o for p, o in zip(parallel_list, orthogonal_list)]
                    masked_full = self._mask_by_magnitude(
                        {str(i): tv for i, tv in enumerate(full_tvs)},
                        mask_rate
                    )

                    # Re-decompose masked task vectors
                    pretrained_w = pretrained_params[param_name]
                    masked_parallel = []
                    masked_orthogonal = []
                    for i, tv in enumerate(full_tvs):
                        masked_tv = masked_full[str(i)]
                        p, o = decompose_weight_matrix(pretrained_w, masked_tv)
                        masked_parallel.append(p)
                        masked_orthogonal.append(o)

                    # Merge parallel subspace (if lambda_1 > 0)
                    if lambda_1 > 0 and masked_parallel:
                        merged_parallel = self._disjoint_merge(masked_parallel, dim=0)
                    else:
                        merged_parallel = torch.zeros_like(masked_parallel[0])

                    # Merge orthogonal subspace
                    if masked_orthogonal:
                        merged_orthogonal = self._merge_orthogonal_with_svd(
                            masked_orthogonal, len(neuronal_tvs)
                        )
                    else:
                        merged_orthogonal = torch.zeros_like(masked_parallel[0])

                    # Combine: tau = lambda_1 * tau_parallel + lambda_2 * tau_orthogonal
                    merged_tv = lambda_1 * merged_parallel + lambda_2 * merged_orthogonal

                    # Add to pretrained: theta = theta_0 + scaling * tau
                    merged_params[param_name] = (
                        pretrained_params[param_name] + scaling_coefficient * merged_tv
                    )

                else:
                    # Non-neuronal: use standard TIES-style merge
                    non_neuronal_list = []
                    for ntv in neuronal_tvs:
                        if param_name in ntv.non_neuronal_param_dict:
                            non_neuronal_list.append(ntv.non_neuronal_param_dict[param_name])

                    if not non_neuronal_list:
                        continue

                    # Mask and merge
                    masked = self._mask_by_magnitude(
                        {str(i): tv for i, tv in enumerate(non_neuronal_list)},
                        mask_rate
                    )
                    masked_list = [masked[str(i)] for i in range(len(non_neuronal_list))]

                    merged_tv = self._disjoint_merge(masked_list, dim=0)

                    # Scale by lambda_2 for consistency
                    merged_params[param_name] = (
                        pretrained_params[param_name] + scaling_coefficient * lambda_2 * merged_tv
                    )

        return self.finalize_merge(
            base_model_obj, base_model_dict, merging_model_list, merged_params
        )

    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        """
        Merge individual tensors using NeuroMerging algorithm.

        This is a simplified version for single tensor merging, used when
        merging at the tensor level rather than full model level.
        """
        scaling_coefficient = method_params.get("scaling_coefficient", 1.0)
        mask_rate = method_params.get("param_value_mask_rate", 0.15)
        lambda_1 = method_params.get("lambda_1", 0.0)
        lambda_2 = method_params.get("lambda_2", 1.2)  # Default for single tensor

        # Compute task vectors
        task_vectors = [
            t.to("cpu") - base_tensor.to("cpu")
            for t in tensors_to_merge
        ]

        # Check if this is a neuronal parameter
        if is_neuronal_param(tensor_name) and base_tensor.dim() >= 2:
            # Neuronal: decompose and merge in subspaces
            parallel_list = []
            orthogonal_list = []

            for tv in task_vectors:
                p, o = decompose_weight_matrix(base_tensor.cpu(), tv)
                parallel_list.append(p)
                orthogonal_list.append(o)

            # Mask full task vectors
            masked_tvs = self._mask_by_magnitude(
                {str(i): tv for i, tv in enumerate(task_vectors)},
                mask_rate
            )

            # Re-decompose masked
            masked_parallel = []
            masked_orthogonal = []
            for i, _ in enumerate(task_vectors):
                p, o = decompose_weight_matrix(base_tensor.cpu(), masked_tvs[str(i)])
                masked_parallel.append(p)
                masked_orthogonal.append(o)

            # Merge
            if lambda_1 > 0:
                merged_p = self._disjoint_merge(masked_parallel, dim=0)
            else:
                merged_p = torch.zeros_like(masked_parallel[0])

            merged_o = self._disjoint_merge(masked_orthogonal, dim=0)

            merged_tv = lambda_1 * merged_p + lambda_2 * merged_o

        else:
            # Non-neuronal: standard TIES merge
            masked_tvs = self._mask_by_magnitude(
                {str(i): tv for i, tv in enumerate(task_vectors)},
                mask_rate
            )
            masked_list = [masked_tvs[str(i)] for i in range(len(task_vectors))]
            merged_tv = self._disjoint_merge(masked_list, dim=0)
            merged_tv = lambda_2 * merged_tv

        # Combine with base
        return base_tensor.to("cpu") + scaling_coefficient * merged_tv
