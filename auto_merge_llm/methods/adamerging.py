"""
AdaMerging: Adaptive Model Merging for Multi-Task Learning

Based on: "AdaMerging: Adaptive Model Merging for Multi-Task Learning" (ICLR 2024)
Paper: https://arxiv.org/abs/2310.02575

Key idea: Learn merging coefficients by minimizing entropy of predictions on
unlabeled test data. Entropy serves as a proxy for prediction loss.

This implementation adapts AdaMerging for cross-lingual model merging where:
- Original MTL tasks → source language models
- Each source language model is a "task expert"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import copy

from auto_merge_llm.utils import TaskVector, logger
from .base_method import MergeMethod


class AdaMergingMethod(MergeMethod):
    """
    AdaMerging method for adaptive model merging.

    Supports:
    - Task-wise AdaMerging: one coefficient per source model
    - Layer-wise AdaMerging: one coefficient per layer per source model
    """

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        """
        AdaMerging method.

        Parameters from method_params:
        - adamerging_mode: 'task_wise' or 'layer_wise'
        - initial_coefficients: initial merging coefficients (list of floats)
        - learning_rate: lr for coefficient optimization (default: 1e-3)
        - num_iterations: number of optimization steps (default: 100)
        - target_dataloader: dataloader for target language data
        - use_ties: whether to apply TIES preprocessing before AdaMerging
        """
        adamerging_mode = method_params.get("adamerging_mode", "task_wise")
        initial_coefficients = method_params.get("initial_coefficients", None)
        learning_rate = method_params.get("learning_rate", 1e-3)
        num_iterations = method_params.get("num_iterations", 100)
        target_dataloader = method_params.get("target_dataloader", None)
        use_ties = method_params.get("use_ties", False)

        logger.info(f"AdaMerging mode: {adamerging_mode}")
        logger.info(f"Using TIES preprocessing: {use_ties}")

        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, exclude_param_names_regex
        )
        base_model_obj = base_model_dict['model']

        # Compute task vectors
        task_vectors = [
            TaskVector(
                pretrained_model=base_model_obj,
                finetuned_model=model_to_merge['model'],
                exclude_param_names_regex=exclude_param_names_regex
            )
            for model_to_merge in merging_model_list
        ]

        num_models = len(task_vectors)
        logger.info(f"Number of models to merge: {num_models}")

        # Apply TIES preprocessing if requested
        if use_ties:
            logger.info("Applying TIES preprocessing to task vectors")
            task_vectors = self._apply_ties_preprocessing(
                task_vectors,
                method_params.get("param_value_mask_rate", 0.8)
            )

        # Initialize coefficients
        if initial_coefficients is not None:
            coefficients = torch.tensor(initial_coefficients, dtype=torch.float32)
        else:
            # Default: equal weights normalized
            coefficients = torch.ones(num_models, dtype=torch.float32) / num_models

        if adamerging_mode == "layer_wise":
            # Get layer names
            layer_names = list(task_vectors[0].task_vector_param_dict.keys())
            num_layers = len(layer_names)
            # Coefficients: (num_layers, num_models)
            coefficients = coefficients.unsqueeze(0).expand(num_layers, -1).clone()
            logger.info(f"Layer-wise coefficients shape: {coefficients.shape}")

        # If no dataloader provided, skip optimization and use initial coefficients
        if target_dataloader is None:
            logger.warning("No target_dataloader provided, using initial coefficients without optimization")
            optimized_coefficients = coefficients
        else:
            # Optimize coefficients via entropy minimization
            optimized_coefficients = self._optimize_coefficients(
                base_model_obj=base_model_obj,
                task_vectors=task_vectors,
                initial_coefficients=coefficients,
                target_dataloader=target_dataloader,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                mode=adamerging_mode
            )

        # Merge with optimized coefficients
        with torch.no_grad():
            if adamerging_mode == "task_wise":
                merged_params = self._merge_task_wise(
                    base_model_obj, task_vectors, optimized_coefficients
                )
            else:
                merged_params = self._merge_layer_wise(
                    base_model_obj, task_vectors, optimized_coefficients
                )

        return self.finalize_merge(
            base_model_obj, base_model_dict, merging_model_list, merged_params
        )

    def _optimize_coefficients(
        self,
        base_model_obj,
        task_vectors: List[TaskVector],
        initial_coefficients: torch.Tensor,
        target_dataloader,
        learning_rate: float,
        num_iterations: int,
        mode: str
    ) -> torch.Tensor:
        """
        Optimize merging coefficients via entropy minimization.

        Args:
            base_model_obj: The pretrained base model
            task_vectors: List of task vectors
            initial_coefficients: Starting coefficients
            target_dataloader: DataLoader for target language data
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            mode: 'task_wise' or 'layer_wise'

        Returns:
            Optimized coefficients tensor
        """
        # Make coefficients learnable
        coefficients = nn.Parameter(initial_coefficients.clone())
        optimizer = torch.optim.Adam([coefficients], lr=learning_rate)

        # Move model to GPU if available for faster optimization
        if torch.cuda.is_available():
            base_model_obj = base_model_obj.to("cuda")
        device = next(base_model_obj.parameters()).device
        logger.info(f"Optimizing coefficients on device: {device}")

        # Get base model state dict for reference
        base_state_dict = {
            name: param.clone()
            for name, param in base_model_obj.named_parameters()
        }

        for iteration in range(num_iterations):
            total_entropy = 0.0
            num_batches = 0

            for batch in target_dataloader:
                # Get inputs
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                else:
                    inputs = batch[0].to(device)

                # Apply merged weights to model
                self._apply_merged_weights(
                    base_model_obj, base_state_dict, task_vectors, coefficients, mode
                )

                # Forward pass
                with torch.enable_grad():
                    outputs = base_model_obj(**inputs) if isinstance(inputs, dict) else base_model_obj(inputs)

                    # Get logits (handle different output formats)
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif hasattr(outputs, 'last_hidden_state'):
                        # For encoder models, use CLS token
                        logits = outputs.last_hidden_state[:, 0, :]
                    else:
                        logits = outputs

                    # Compute entropy
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

                    total_entropy += entropy
                    num_batches += 1

                # Backprop
                optimizer.zero_grad()
                entropy.backward()
                optimizer.step()

                # Clamp coefficients to [0, 1]
                with torch.no_grad():
                    coefficients.data.clamp_(0.0, 1.0)

            if (iteration + 1) % 10 == 0:
                avg_entropy = total_entropy / max(num_batches, 1)
                logger.info(f"Iteration {iteration + 1}/{num_iterations}, Avg Entropy: {avg_entropy:.4f}")

        # Restore base model weights
        with torch.no_grad():
            for name, param in base_model_obj.named_parameters():
                param.copy_(base_state_dict[name])

        return coefficients.detach()

    def _apply_merged_weights(
        self,
        model,
        base_state_dict: Dict[str, torch.Tensor],
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor,
        mode: str
    ):
        """Apply merged weights to model for forward pass."""
        if mode == "task_wise":
            # coefficients: (num_models,)
            for name, param in model.named_parameters():
                merged_delta = torch.zeros_like(param)
                for idx, tv in enumerate(task_vectors):
                    if name in tv.task_vector_param_dict:
                        merged_delta += coefficients[idx] * tv.task_vector_param_dict[name].to(param.device)
                param.data = base_state_dict[name].to(param.device) + merged_delta
        else:
            # Layer-wise: coefficients (num_layers, num_models)
            layer_names = list(task_vectors[0].task_vector_param_dict.keys())
            name_to_layer_idx = {name: idx for idx, name in enumerate(layer_names)}

            for name, param in model.named_parameters():
                if name in name_to_layer_idx:
                    layer_idx = name_to_layer_idx[name]
                    merged_delta = torch.zeros_like(param)
                    for model_idx, tv in enumerate(task_vectors):
                        if name in tv.task_vector_param_dict:
                            merged_delta += coefficients[layer_idx, model_idx] * tv.task_vector_param_dict[name].to(param.device)
                    param.data = base_state_dict[name].to(param.device) + merged_delta

    def _merge_task_wise(
        self,
        base_model,
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Merge using task-wise coefficients."""
        merged_params = {}
        base_state_dict = dict(base_model.named_parameters())

        for name, base_param in base_state_dict.items():
            merged_delta = torch.zeros_like(base_param)
            for idx, tv in enumerate(task_vectors):
                if name in tv.task_vector_param_dict:
                    merged_delta += coefficients[idx].item() * tv.task_vector_param_dict[name].to(base_param.device)
            merged_params[name] = base_param + merged_delta

        logger.info(f"Task-wise merged with coefficients: {coefficients.tolist()}")
        return merged_params

    def _merge_layer_wise(
        self,
        base_model,
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Merge using layer-wise coefficients."""
        merged_params = {}
        base_state_dict = dict(base_model.named_parameters())
        layer_names = list(task_vectors[0].task_vector_param_dict.keys())
        name_to_layer_idx = {name: idx for idx, name in enumerate(layer_names)}

        for name, base_param in base_state_dict.items():
            if name in name_to_layer_idx:
                layer_idx = name_to_layer_idx[name]
                merged_delta = torch.zeros_like(base_param)
                for model_idx, tv in enumerate(task_vectors):
                    if name in tv.task_vector_param_dict:
                        coeff = coefficients[layer_idx, model_idx].item()
                        merged_delta += coeff * tv.task_vector_param_dict[name].to(base_param.device)
                merged_params[name] = base_param + merged_delta
            else:
                merged_params[name] = base_param

        logger.info(f"Layer-wise merged with coefficients shape: {coefficients.shape}")
        return merged_params

    def _apply_ties_preprocessing(
        self,
        task_vectors: List[TaskVector],
        param_value_mask_rate: float = 0.8
    ) -> List[TaskVector]:
        """Apply TIES preprocessing to task vectors."""
        from .ties import TiesMerging
        ties_method = TiesMerging()

        # Get flattened params
        flattened_list = [
            ties_method.task_vector_param_dict_to_single_vector(tv)
            for tv in task_vectors
        ]
        flattened_params = torch.vstack(flattened_list)

        # Apply TIES: mask smallest magnitude
        masked_params = ties_method.mask_smallest_magnitude_param_values(
            flattened_params, param_value_mask_rate
        )

        # Get signs and apply disjoint merge per-model
        param_signs = ties_method.get_param_signs(masked_params)

        # Create new task vectors with TIES-processed params
        processed_vectors = []
        for idx, tv in enumerate(task_vectors):
            # Apply sign consistency mask
            single_vector = masked_params[idx]
            sign_mask = (
                ((param_signs > 0) & (single_vector > 0)) |
                ((param_signs < 0) & (single_vector < 0))
            )
            processed_vector = single_vector * sign_mask.float()

            # Convert back to param dict
            processed_dict = ties_method.single_vector_to_task_vector_param_dict(
                processed_vector, tv
            )
            processed_vectors.append(TaskVector(task_vector_param_dict=processed_dict))

        return processed_vectors

    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        """
        AdaMerging for tensor merging (simplified without optimization).
        Uses initial coefficients for static merging.
        """
        initial_coefficients = method_params.get("initial_coefficients", None)

        if initial_coefficients is None:
            num_models = len(tensors_to_merge)
            coefficients = [1.0 / num_models] * num_models
        else:
            coefficients = initial_coefficients

        # Compute task vectors
        task_vector_dicts = [
            {tensor_name: t.to("cpu") - base_tensor.to("cpu")}
            for t in tensors_to_merge
        ]

        # Weighted sum
        merged_delta = torch.zeros_like(base_tensor)
        for idx, tv_dict in enumerate(task_vector_dicts):
            merged_delta += coefficients[idx] * tv_dict[tensor_name]

        return base_tensor + merged_delta
