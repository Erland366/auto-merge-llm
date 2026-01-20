"""
Directional Consensus Merging: Project task vectors onto consensus direction.

Hypothesis: Task vectors from different languages point in different directions.
By aligning them to a consensus direction per layer, we reduce interference.

Algorithm (per-parameter):
1. Compute mean direction: d = normalize(sum(tau_i))
2. Project each task vector: tau_aligned_i = (tau_i . d) * d
3. Soft projection (optional): tau_mixed = alpha * tau_aligned + (1-alpha) * tau_original
4. Aggregate: merged_tau = sum(w_i * tau_mixed_i)
5. Combine: merged = base + scaling * merged_tau

Soft projection (alpha parameter):
- alpha=1.0: Full projection onto consensus (maximum variance reduction)
- alpha=0.0: No projection, use original task vectors (equivalent to similarity method)
- 0 < alpha < 1: Interpolate between aligned and original (preserve some orthogonal signal)
"""

import torch

from auto_merge_llm.utils import TaskVector, logger
from auto_merge_llm.utils import apply_dare_to_task_vector, DAREConfig
from .base_method import MergeMethod


class DirectionalConsensusMerging(MergeMethod):
    """
    Directional Consensus: Project task vectors onto mean direction before merging.

    This method computes a consensus direction for each parameter/layer separately,
    then projects all task vectors onto this direction to reduce interference from
    conflicting gradient directions.
    """

    def _compute_consensus_direction(self, param_tensors: list) -> torch.Tensor:
        """
        Compute normalized consensus direction from stacked parameter tensors.

        Args:
            param_tensors: List of tensors, each shape [*param_shape]

        Returns:
            Normalized direction tensor, shape [*param_shape]
        """
        # Stack tensors: shape [num_models, *param_shape]
        stacked = torch.stack(param_tensors, dim=0)

        # Sum across models
        sum_vector = stacked.sum(dim=0)

        # Compute L2 norm
        norm = torch.norm(sum_vector)

        # Normalize (handle zero case)
        if norm > 1e-8:
            return sum_vector / norm
        else:
            return sum_vector

    def _project_onto_direction(
        self,
        tau: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Project a task vector onto the consensus direction.

        Args:
            tau: Task vector tensor, shape [*param_shape]
            direction: Normalized direction tensor, shape [*param_shape]

        Returns:
            Aligned task vector, shape [*param_shape]
        """
        # Compute scalar projection (dot product)
        projection_scalar = (tau * direction).sum()

        # Return projected vector
        return projection_scalar * direction

    def _apply_soft_projection(
        self,
        tau: torch.Tensor,
        tau_aligned: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply soft projection: interpolate between aligned and original.

        Args:
            tau: Original task vector tensor
            tau_aligned: Projected task vector tensor
            alpha: Projection strength (0.0 = original, 1.0 = fully projected)

        Returns:
            Interpolated task vector: alpha * tau_aligned + (1 - alpha) * tau
        """
        if alpha >= 1.0:
            return tau_aligned
        elif alpha <= 0.0:
            return tau
        else:
            return alpha * tau_aligned + (1.0 - alpha) * tau

    def _aggregate_aligned_vectors(
        self,
        aligned_tensors: list,
        weights: list,
        aggregation_mode: str
    ) -> torch.Tensor:
        """
        Aggregate aligned task vectors.

        Args:
            aligned_tensors: List of aligned task vector tensors
            weights: Similarity weights for each model
            aggregation_mode: 'weighted', 'sum', or 'average'

        Returns:
            Merged task vector tensor
        """
        if aggregation_mode == 'weighted':
            # Normalize weights to sum to 1
            weight_sum = sum(weights)
            if weight_sum > 1e-8:
                normalized_weights = [w / weight_sum for w in weights]
            else:
                normalized_weights = [1.0 / len(weights)] * len(weights)

            return sum(w * t for w, t in zip(normalized_weights, aligned_tensors))

        elif aggregation_mode == 'sum':
            return sum(aligned_tensors)

        elif aggregation_mode == 'average':
            return sum(aligned_tensors) / len(aligned_tensors)

        else:
            raise ValueError(f"Unknown aggregation_mode: {aggregation_mode}")

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        """
        Directional Consensus merging.

        Args:
            base_model: Path to base/pretrained model
            models_to_merge: List of paths to fine-tuned models
            method_params:
                - scaling_coefficient: float, final scaling (default 1.0)
                - aggregation_mode: str, 'weighted'|'sum'|'average' (default 'weighted')
                - weights: list, similarity weights for each model (optional)
                - alpha: float, soft projection strength (default 1.0)
                    - 1.0 = full projection (maximum variance reduction)
                    - 0.0 = no projection (use original task vectors)
                    - 0 < alpha < 1 = interpolate (preserve some orthogonal signal)
            mask_merging: Optional mask merging config
            exclude_param_names_regex: Regex patterns for params to exclude

        Returns:
            Dict with 'merged_model', 'base_tokenizer', 'merged_model_tokenizers'
        """
        scaling_coefficient = method_params.get("scaling_coefficient", 1.0)
        aggregation_mode = method_params.get("aggregation_mode", "weighted")
        weights = method_params.get("weights", None)
        alpha = method_params.get("alpha", 1.0)  # Soft projection strength

        dare_config = DAREConfig.from_method_params(method_params)

        assert isinstance(scaling_coefficient, (int, float)), \
            "scaling_coefficient must be a number!"

        # Load models
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, exclude_param_names_regex
        )
        base_model_obj = base_model_dict['model']

        # Apply mask merging if specified
        if mask_merging is not None:
            self.mask_params(
                base_model_obj,
                merging_model_list,
                exclude_param_names_regex,
                mask_merging
            )

        # Create task vectors for each model
        task_vectors = [
            TaskVector(
                pretrained_model=base_model_obj,
                finetuned_model=model_to_merge['model'],
                exclude_param_names_regex=exclude_param_names_regex
            )
            for model_to_merge in merging_model_list
        ]

        # Apply DARE if configured
        if dare_config.enabled:
            logger.info(f"Applying DARE with drop_rate={dare_config.drop_rate}")
            for idx, tv in enumerate(task_vectors):
                seed = (dare_config.seed + idx) if dare_config.seed is not None else None
                tv.task_vector_param_dict = apply_dare_to_task_vector(
                    tv.task_vector_param_dict,
                    drop_rate=dare_config.drop_rate,
                    rescale=dare_config.rescale,
                    seed=seed,
                    inplace=False
                )

        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(task_vectors)] * len(task_vectors)

        # Get parameter names from first task vector
        param_names = list(task_vectors[0].task_vector_param_dict.keys())

        logger.info(f"Directional Consensus merging with {len(task_vectors)} models, "
                   f"aggregation_mode={aggregation_mode}, scaling={scaling_coefficient}, "
                   f"alpha={alpha}")

        # Per-parameter directional consensus
        merged_param_dict = {}

        with torch.no_grad():
            for param_name in param_names:
                # Extract this parameter from all task vectors
                param_tensors = [
                    tv.task_vector_param_dict[param_name]
                    for tv in task_vectors
                ]

                # Compute consensus direction for this parameter
                direction = self._compute_consensus_direction(param_tensors)

                # Project each task vector onto the consensus direction
                aligned_tensors = [
                    self._project_onto_direction(tau, direction)
                    for tau in param_tensors
                ]

                # Apply soft projection (interpolate between aligned and original)
                mixed_tensors = [
                    self._apply_soft_projection(tau, tau_aligned, alpha)
                    for tau, tau_aligned in zip(param_tensors, aligned_tensors)
                ]

                # Aggregate mixed task vectors
                merged_param_dict[param_name] = self._aggregate_aligned_vectors(
                    mixed_tensors, weights, aggregation_mode
                )

        # Create merged task vector and combine with base model
        merged_task_vector = TaskVector(task_vector_param_dict=merged_param_dict)
        merged_params = merged_task_vector.combine_with_pretrained_model(
            pretrained_model=base_model_obj,
            scaling_coefficient=scaling_coefficient
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
        Single tensor version of directional consensus merge.

        Args:
            base_tensor: Base tensor
            tensors_to_merge: List of tensors to merge
            method_params: Method parameters
            mask_merging: Optional mask config
            tensor_name: Name of tensor (for logging)

        Returns:
            Merged tensor
        """
        scaling_coefficient = method_params.get("scaling_coefficient", 1.0)
        aggregation_mode = method_params.get("aggregation_mode", "weighted")
        weights = method_params.get("weights", None)
        alpha = method_params.get("alpha", 1.0)  # Soft projection strength

        # Compute task tensors (difference from base)
        task_tensors = [
            t.to("cpu") - base_tensor.to("cpu")
            for t in tensors_to_merge
        ]

        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(task_tensors)] * len(task_tensors)

        with torch.no_grad():
            # Compute consensus direction
            direction = self._compute_consensus_direction(task_tensors)

            # Project each task tensor onto direction
            aligned_tensors = [
                self._project_onto_direction(tau, direction)
                for tau in task_tensors
            ]

            # Apply soft projection
            mixed_tensors = [
                self._apply_soft_projection(tau, tau_aligned, alpha)
                for tau, tau_aligned in zip(task_tensors, aligned_tensors)
            ]

            # Aggregate
            merged_tau = self._aggregate_aligned_vectors(
                mixed_tensors, weights, aggregation_mode
            )

        # Combine with base
        return base_tensor.to("cpu") + scaling_coefficient * merged_tau
