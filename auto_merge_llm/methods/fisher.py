from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

from auto_merge_llm.utils import get_param_names_to_merge, logger
from .base_method import MergeMethod


class FisherMerging(MergeMethod):        
    def get_param_squared_gradients(self, model, param_names_to_merge):
        """
        get the squared gradients of parameters
        :param model: nn.Module, model
        :param param_names_to_merge: list, list of parameter names that need to be merged
        :return:
        """
        param_squared_gradients = {param_name: param_value.grad.detach(
        ) ** 2 for param_name, param_value in model.named_parameters() if param_name in param_names_to_merge}
        return param_squared_gradients

    def get_models_fisher_norm(
        self, 
        models_to_merge_param_dict, 
        models_to_merge_fisher_weights_list
    ):
        """
        get normalization of fisher weights of all the models that need to be merged
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :return:
        """
        # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
        models_fisher_norm_dict = {}
        # compute L2 norm over models for each parameter
        for param_name, _ in models_to_merge_param_dict.items():
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
            models_fisher = torch.stack(
                [
                    model_to_merge_fisher_weights[param_name]
                    for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
                ], 
                dim=0
            )
            dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
            # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter
            models_fisher_norm = torch.norm(models_fisher, dim=dims)
            models_fisher_norm_dict[param_name] = models_fisher_norm

        # Tensor, shape (num_models_to_merge, num_parameters)
        models_fisher_norm = torch.stack(
            [
                models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()
            ], 
            dim=1
        )
        # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
        models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
        return models_fisher_norm

    def merging_with_fisher_weights(
        self, 
        models_to_merge_param_dict,
        models_to_merge_fisher_weights_list, 
        fisher_scaling_coefficients,
        normalize_fisher_weight, 
        minimal_fisher_weight=1e-6
    ):
        """
        merge parameters of different models with computed fisher weights
        :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        value is a list of the corresponding parameters of all the models that need to be merged
        :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
        each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        :param fisher_scaling_coefficients: torch.Tensor, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :return:
        """
        # dict, dictionary of model parameters
        merged_params = {}

        if normalize_fisher_weight:
            # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
            models_fisher_norm = self.get_models_fisher_norm(
                models_to_merge_param_dict=models_to_merge_param_dict,
                models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list
            )

        for param_name, param_value_list in models_to_merge_param_dict.items():
            # shape (num_models_to_merge, *parameter_shape)
            param_values = torch.stack(param_value_list, dim=0)
            # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
            models_to_merge_fisher_weights = torch.stack(
                [
                    model_to_merge_fisher_weights[param_name]
                    for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
                ], dim=0
            ) + minimal_fisher_weight

            # Tensor, shape (num_models_to_merge, 1, 1, ...)
            reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(
                -1, 
                *[1 for _ in range(param_values.dim() - 1)]
            ).to(param_values.device)

            if normalize_fisher_weight:
                # Tensor, shape (num_models_to_merge, )
                _models_fisher_norm = 1.0 / \
                    (models_fisher_norm + minimal_fisher_weight)
                normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(
                    -1, *[1 for _ in range(param_values.dim() - 1)])
                reshaped_scaling_coefficients = reshaped_scaling_coefficients * \
                    normalized_models_fisher_norm

            # shape (*parameter_shape)
            numerator = (reshaped_scaling_coefficients *
                            models_to_merge_fisher_weights * param_values).sum(dim=0)

            # shape (*parameter_shape)
            denominator = (reshaped_scaling_coefficients *
                            models_to_merge_fisher_weights).sum(dim=0)

            merged_param = numerator / denominator
            merged_params[param_name] = merged_param
        return merged_params

    def merge(
        self, 
        base_model, 
        models_to_merge, 
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        trainers = method_params["trainers"]
        nums_fisher_examples = method_params["nums_fisher_examples"]
        normalize_fisher_weight = method_params["normalize_fisher_weight"]
        minimal_fisher_weight = method_params["minimal_fisher_weight"]
        fisher_scaling_coefficients = method_params["fisher_scaling_coefficients"]
        
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        models_to_merge_fisher_weights_list = []

        assert len(models_to_merge) == len(trainers) == len(
            nums_fisher_examples), "sizes of lists are not identical!"
        
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, 
            exclude_param_names_regex
        ) 
        
        for model_idx, (model_to_merge_dict, trainer, num_fisher_examples) in enumerate(
            zip(merging_model_list, trainers, nums_fisher_examples)
        ):
            model_to_merge = model_to_merge_dict['model']
            param_dict = {
                param_name: param_value 
                for param_name, param_value in model_to_merge.named_parameters()
            }
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex
            )

            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name]
                )

            # list of dictionaries with length (num_fisher_examples // batch_size) or (num_fisher_examples // batch_size) + 1,
            # each dictionary records the fisher weights of parameters for model_to_merge computed by examples in a batch
            batches_fisher_weights_list = []

            num_computed_examples = 0
            train_dataloader = trainer.get_train_dataloader()
            if num_fisher_examples % trainer._train_batch_size != 0:
                print(
                    f"warning: the number of examples for computing fisher cannot be fully "
                    f"divided by the batch size for model {model_idx}, which may lead to a "
                    f"slightly different number of the actually used examples."
                )
            for step, inputs in tqdm(
                enumerate(train_dataloader), 
                desc=f"computing fisher weights for model {model_idx}"
            ):
                if num_computed_examples >= num_fisher_examples:
                    break
                # Prepare and, if needed, trim the last batch to avoid overrun
                inputs = trainer._prepare_inputs(inputs)
                # Infer actual batch size from inputs
                try:
                    any_tensor = next(iter(inputs.values()))
                    actual_bs = int(any_tensor.shape[0])
                except Exception:
                    actual_bs = trainer._train_batch_size
                remaining = num_fisher_examples - num_computed_examples
                if actual_bs > remaining:
                    # Trim tensors in the batch to 'remaining' examples
                    inputs = {k: (v[:remaining] if hasattr(v, 'shape') and v.shape[0] >= remaining else v)
                              for k, v in inputs.items()}
                    actual_bs = remaining
                outputs = model_to_merge(**inputs)
                # Tensor, shape (batch_size, num_label_classes)
                logits = outputs.logits
                # compute fisher weights for regression task
                if logits.shape[-1] == 1:
                    # use the label information to compute loss and obtain gradients
                    mse_loss = outputs.loss
                    model_to_merge.zero_grad()
                    mse_loss.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = self.get_param_squared_gradients(
                        model=model_to_merge, 
                        param_names_to_merge=param_names_to_merge
                    )
                # compute fisher weights for classification task
                else:
                    # use detach() to detach from the computation graph
                    # Tensor, shape (batch_size, num_label_classes)
                    labels_probabilities = torch.softmax(
                        logits, dim=-1
                    ).detach()
                    labels_log_probabilities = torch.log_softmax(
                        logits, 
                        dim=-1
                    )
                    # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                    labels_expectations = torch.sqrt(
                        labels_probabilities
                    ) * labels_log_probabilities
                    # sum over label classes and batch dimension
                    sum_labels_expectations = labels_expectations.sum(
                        dim=-1).sum(dim=0)
                    model_to_merge.zero_grad()
                    sum_labels_expectations.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = self.get_param_squared_gradients(
                        model=model_to_merge, 
                        param_names_to_merge=param_names_to_merge
                    )

                batches_fisher_weights_list.append(batch_fisher_weights)
                num_computed_examples += actual_bs

            model_to_merge_fisher_weights = {}
            for batch_fisher_weights in batches_fisher_weights_list:
                for key in batch_fisher_weights:
                    if key not in model_to_merge_fisher_weights:
                        model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                    else:
                        model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

            # mean over batches
            for key in model_to_merge_fisher_weights:
                model_to_merge_fisher_weights[key] /= num_computed_examples
            models_to_merge_fisher_weights_list.append(
                model_to_merge_fisher_weights)

        # merging with fisher weights
        # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
        if fisher_scaling_coefficients is None:
            fisher_scaling_coefficients = torch.ones(
                len(models_to_merge)) / len(models_to_merge)
        else:
            assert isinstance(
                fisher_scaling_coefficients,
                list
            ), "wrong type of fisher_scaling_coefficients, should be list!"
            assert len(fisher_scaling_coefficients) == len(
                models_to_merge
            ), "mismatched length of fisher_scaling_coefficients!"
            
            fisher_scaling_coefficients = torch.Tensor(
                fisher_scaling_coefficients
            )
        # merging with fisher weights
        merged_params = self.merging_with_fisher_weights(
            models_to_merge_param_dict=models_to_merge_param_dict, 
            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
            fisher_scaling_coefficients=fisher_scaling_coefficients, 
            normalize_fisher_weight=normalize_fisher_weight, 
            minimal_fisher_weight=minimal_fisher_weight
        )
        
        # Use the instantiated base model object (not the string path)
        base_model_instance = base_model_dict['model']
        return self.finalize_merge(
            base_model_instance, 
            base_model_dict,
            merging_model_list, 
            merged_params
        )

    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        # Simplified tensor merging for fisher - falls back to average
        # since we don't have gradient information
        merged_tensor = torch.stack(
            [
                merging_tensor
                for merging_tensor in tensors_to_merge
            ],
            dim=0
        ).mean(dim=0)
        return merged_tensor

class SimplifiedFisherMerging(MergeMethod):
    """
    Simplified Fisher merging that works with the existing pipeline.
    Uses parameter magnitude as a proxy for Fisher information when actual
    training data is not available.
    """

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge,
            exclude_param_names_regex
        )
        self.mask_params(
            base_model_dict['model'],
            [model_to_merge['model'] for model_to_merge in merging_model_list],
            exclude_param_names_regex,
            mask_merging
        )
        models_to_merge_param_dict = defaultdict(list)
        base_model = base_model_dict['model']

        # Extract URIEL weights if provided
        uriel_weights = method_params.get('weights', None)
        if uriel_weights is not None:
            # Convert to tensor and normalize
            uriel_weights = torch.tensor(uriel_weights, dtype=torch.float32)
            uriel_weights = uriel_weights / (uriel_weights.sum() + 1e-8)
            print(f"Using URIEL weights for Fisher merging: {uriel_weights.tolist()}")
        else:
            print("No URIEL weights provided, using pure Fisher merging")

        # iterate each individual model that needs to be merged
        for model_to_merge_dict in merging_model_list:
            model_to_merge = model_to_merge_dict['model']
            param_dict = {
                param_name: param_value
                for param_name, param_value
                in model_to_merge.named_parameters()
            }
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex
            )
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name]
                )

        with torch.no_grad():
            # Simplified Fisher merging using parameter magnitudes as proxy
            merged_params = {}
            for param_name, model_to_merge_param in models_to_merge_param_dict.items():
                # Stack parameters from all models
                param_values = torch.stack(model_to_merge_param, dim=0)

                # Use parameter magnitudes as Fisher information proxy
                # Larger magnitudes indicate more important parameters
                fisher_weights = torch.norm(param_values, dim=tuple(range(1, param_values.dim())))

                # Normalize Fisher weights
                fisher_weights = fisher_weights / (fisher_weights.sum() + 1e-8)

                # Combine Fisher weights with URIEL weights if provided
                if uriel_weights is not None:
                    # Ensure uriel_weights matches the number of models
                    if len(uriel_weights) != len(model_to_merge_param):
                        print(f"Warning: URIEL weights count ({len(uriel_weights)}) != models count ({len(model_to_merge_param)})")
                        # Use equal weights as fallback
                        combined_weights = fisher_weights
                    else:
                        # Combine Fisher and URIEL weights (element-wise multiplication)
                        # This gives us parameter-level importance weighted by model-level similarity
                        uriel_reshaped = uriel_weights.reshape(-1, *[1 for _ in range(param_values.dim() - 1)])
                        combined_weights = fisher_weights.unsqueeze(-1) * uriel_reshaped.squeeze(-1)
                        # Normalize combined weights
                        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
                else:
                    combined_weights = fisher_weights

                # Reshape for broadcasting
                reshaped_weights = combined_weights.reshape(
                    -1, *[1 for _ in range(param_values.dim() - 1)]
                )

                # Weighted average using combined Fisher+URIEL weights
                merged_param = (reshaped_weights * param_values).sum(dim=0)
                merged_params[param_name] = merged_param

        return self.finalize_merge(
            base_model,
            base_model_dict,
            merging_model_list,
            merged_params
        )

    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        stacked_tensors = torch.stack([
            merging_tensor
            for merging_tensor in tensors_to_merge
        ], dim=0)

        fisher_weights = torch.norm(stacked_tensors, dim=tuple(range(1, stacked_tensors.dim())))
        fisher_weights = fisher_weights / (fisher_weights.sum() + 1e-8)

        reshaped_weights = fisher_weights.reshape(
            -1, *[1 for _ in range(stacked_tensors.dim() - 1)]
        )

        merged_tensor = (reshaped_weights * stacked_tensors).sum(dim=0)
        return merged_tensor
