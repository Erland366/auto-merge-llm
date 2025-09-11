import os
import copy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

from auto_merge_llm.tokenizer import align_tokenizers_and_embeddings
from auto_merge_llm.utils import get_model_storage_path, logger, get_param_names_to_merge
from auto_merge_llm.utils.model_registry import ModelRegistry

CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"))


class MergeMethod(ABC):
    def __init__(self):
        pass  
    
    def prepare_merge(
        self,
        base_model,
        models_to_merge,
        exclude_param_names_regex
    ):
        base_model_dict, merging_model_list = self._load_checkpoints(base_model, models_to_merge)
        align_tokenizers_and_embeddings(
            pretrained_model=base_model_dict['model'], 
            pretrained_tokenizer=base_model_dict['tokenizer'],
            pretrained_config=base_model_dict['config'], 
            finetuned_models=[merging_model['model'] for merging_model in merging_model_list],
            finetuned_tokenizers=[merging_model['tokenizer'] for merging_model in merging_model_list], 
            finetuned_configs=[merging_model['config'] for merging_model in merging_model_list]
        )
        return base_model_dict, merging_model_list
    
    def finalize_merge(
        self,
        base_model,
        base_model_dict,
        merging_model_list,
        averaged_params
    ):
        self.copy_params_to_model(params=averaged_params, model=base_model)
        merged_res = {
            'merged_model': base_model,
            'base_tokenizer': base_model_dict['tokenizer'],
            'merged_model_tokenizers': [merging_model['tokenizer']
                                        for merging_model
                                        in merging_model_list]
        }
        return merged_res
    
    def copy_params_to_model(
        self,
        params,
        model
    ):
        size_mismatches = []
        
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                # Handle size mismatches due to tokenizer alignment
                if param_value.shape != params[param_name].shape:
                    # Instead of silently handling, collect all mismatches
                    size_mismatches.append({
                        'param_name': param_name,
                        'model_shape': param_value.shape,
                        'param_shape': params[param_name].shape
                    })
                else:
                    param_value.data.copy_(params[param_name])
        
        # If there are any size mismatches, raise an error with detailed information
        if size_mismatches:
            error_msg = "Model merging failed due to parameter size mismatches:\n\n"
            for mismatch in size_mismatches:
                error_msg += f"Parameter: {mismatch['param_name']}\n"
                error_msg += f"  Model shape: {mismatch['model_shape']}\n"
                error_msg += f"  Param shape: {mismatch['param_shape']}\n\n"
            
            error_msg += "This indicates that the models have incompatible architectures or tokenizer configurations.\n"
            error_msg += "Possible solutions:\n"
            error_msg += "1. Ensure all models use the same base architecture\n"
            error_msg += "2. Check tokenizer compatibility across models\n"
            error_msg += "3. Use models fine-tuned from the same base checkpoint\n"
            error_msg += "4. Consider using task arithmetic merging instead of direct parameter merging"
            
            raise ValueError(error_msg)

    def mask_params(
        self,
        base_model,
        models_to_merge,
        exclude_param_names_regex,
        mask_merging
    ):
        """
        Applies a mask to the parameters of the models to be merged.
        This implementation reverts parameters with the smallest magnitude changes 
        (task vectors) back to their original base_model state, effectively 
        excluding them from the merge.
        """
        if mask_merging is None:
            return

        mask_rate = mask_merging.get("mask_rate", 0.0)
        if mask_rate == 0.0:
            return

        logger.info(f"Applying parameter masking with a rate of {mask_rate}.")

        base_params = dict(base_model.named_parameters())
        
        with torch.no_grad():
            # Handle both model objects and model dictionaries
            for model_data in models_to_merge:
                # Check if model_data is a dictionary with 'model' key or a direct model object
                if hasattr(model_data, '__getitem__') and 'model' in model_data:
                    model = model_data['model']
                else:
                    model = model_data
                    
                model_params = dict(model.named_parameters())
                
                param_names_to_merge = get_param_names_to_merge(
                    input_param_names=list(model_params.keys()),
                    exclude_param_names_regex=exclude_param_names_regex
                )

                for name in param_names_to_merge:
                    if name in base_params:
                        delta = model_params[name].data - base_params[name].data
                        
                        # Calculate threshold for this specific parameter tensor
                        num_elements = delta.numel()
                        if num_elements == 0:
                            continue
                        
                        k = int(num_elements * mask_rate)
                        if k == 0:
                            continue

                        threshold = torch.kthvalue(delta.flatten().abs(), k).values
                        
                        # Create mask and revert values below the threshold
                        mask = delta.abs() < threshold
                        model_params[name].data[mask] = base_params[name].data[mask]
    
        
    def _load_checkpoint(
        self,
        model_path
    ):
        res = {}
        
        # Parse model path to extract subfolder if present
        subfolder = None
        actual_model_path = model_path
        
        # Check if model_path contains subfolder information (format: model_name@subfolder)
        if '@' in model_path:
            actual_model_path, subfolder = model_path.split('@', 1)
            logger.info(f"Loading model {actual_model_path} from subfolder: {subfolder}")
        
        try:
            # For models with subfolders, don't use get_model_storage_path as it doesn't handle @ symbols
            if subfolder:
                res['model'] = ModelRegistry.load_model(
                    model_path=actual_model_path,
                    device_map="cpu",
                    cache_dir=CACHE_DIR,
                    subfolder=subfolder
                )
                res['tokenizer'] = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=actual_model_path,
                    cache_dir=CACHE_DIR,
                    subfolder=subfolder
                )
                res['config'] = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=actual_model_path,
                    cache_dir=CACHE_DIR,
                    subfolder=subfolder
                )
            else:
                # For models without subfolders, use the original logic
                temp_model_path = get_model_storage_path(model_path)
                res['model'] = ModelRegistry.load_model(
                    model_path=temp_model_path,
                    device_map="cpu"
                )
                res['tokenizer'] = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=temp_model_path
                )
                res['config'] = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=temp_model_path
                )
        except Exception as e:
            logger.error(e)
            res['model'] = ModelRegistry.load_model(
                model_path=actual_model_path,
                device_map="cpu",
                cache_dir=CACHE_DIR,
                subfolder=subfolder
            )
            res['tokenizer'] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=actual_model_path, 
                cache_dir=CACHE_DIR,
                subfolder=subfolder
            )
            res['config'] = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=actual_model_path, 
                cache_dir=CACHE_DIR,
                subfolder=subfolder
            )
        return res
            
    def _load_checkpoints(
        self,
        base_model_path,
        models_to_merge_paths
    ):
        based_model = {}
        merging_model_list = []
        based_model = self._load_checkpoint(base_model_path)
        for model_merge_path in models_to_merge_paths:
            merging_model_list.append(
                self._load_checkpoint(model_merge_path)
            )
        return based_model, merging_model_list
                   
    @abstractmethod
    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        pass
    
    @abstractmethod
    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name="default"
    ):
        pass
