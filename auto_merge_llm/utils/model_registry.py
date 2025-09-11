"""
Model registry for handling different model architectures in merging.
"""
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoConfig
)


class ModelRegistry:
    """Registry for mapping model types to their appropriate AutoModel classes."""
    
    MODEL_TYPE_MAPPING = {
        # Decoder-only models
        'gpt2': AutoModelForCausalLM,
        'gpt_neo': AutoModelForCausalLM,
        'gptj': AutoModelForCausalLM,
        'llama': AutoModelForCausalLM,
        'mistral': AutoModelForCausalLM,
        'qwen2': AutoModelForCausalLM,
        'phi': AutoModelForCausalLM,
        'starcoder2': AutoModelForCausalLM,
        
        # Encoder-only models
        'roberta': AutoModelForMaskedLM,
        'bert': AutoModelForMaskedLM,
        'xlm-roberta': AutoModelForMaskedLM,
        'distilbert': AutoModelForMaskedLM,
        'albert': AutoModelForMaskedLM,
        'camembert': AutoModelForMaskedLM,
        
        # Default fallback
        'default': AutoModel
    }
    
    TASK_SPECIFIC_MAPPING = {
        # If config has these architectures, use task-specific models
        'RobertaForSequenceClassification': AutoModelForSequenceClassification,
        'BertForSequenceClassification': AutoModelForSequenceClassification,
        'XLMRobertaForSequenceClassification': AutoModelForSequenceClassification,
        'RobertaForTokenClassification': AutoModelForTokenClassification,
        'BertForTokenClassification': AutoModelForTokenClassification,
        'RobertaForQuestionAnswering': AutoModelForQuestionAnswering,
        'BertForQuestionAnswering': AutoModelForQuestionAnswering,
    }
    
    @classmethod
    def get_auto_model_class(cls, model_path, subfolder=None):
        """
        Get the appropriate AutoModel class for a given model path.
        
        Args:
            model_path: Path or identifier for the model
            subfolder: Subfolder to load from (for hyperparameter tuning results)
            
        Returns:
            The appropriate AutoModel class
        """
        try:
            # First, try to load the config to determine the model type
            if subfolder:
                config = AutoConfig.from_pretrained(model_path, subfolder=subfolder)
            else:
                config = AutoConfig.from_pretrained(model_path)
            
            # Check if it's a task-specific model
            if hasattr(config, 'architectures') and config.architectures:
                architecture = config.architectures[0]
                if architecture in cls.TASK_SPECIFIC_MAPPING:
                    return cls.TASK_SPECIFIC_MAPPING[architecture]
            
            # For encoder models, check if we should use sequence classification
            model_type = getattr(config, 'model_type', None)
            if model_type in ['roberta', 'bert', 'xlm-roberta', 'distilbert']:
                # Check if it has id2label which indicates it's a classification model
                if hasattr(config, 'id2label') and config.id2label:
                    return AutoModelForSequenceClassification
            
            # Fall back to model type mapping
            if model_type in cls.MODEL_TYPE_MAPPING:
                return cls.MODEL_TYPE_MAPPING[model_type]
            
            # Default fallback
            return cls.MODEL_TYPE_MAPPING['default']
            
        except Exception as e:
            # If we can't determine, use AutoModel as safe default
            print(f"Warning: Could not determine model type for {model_path}, using AutoModel. Error: {e}")
            return AutoModel
    
    @classmethod
    def load_model(cls, model_path, device_map="cpu", subfolder=None, **kwargs):
        """
        Load a model with the appropriate architecture.
        
        Args:
            model_path: Path or identifier for the model
            device_map: Device mapping for the model
            subfolder: Subfolder to load from (for hyperparameter tuning results)
            **kwargs: Additional arguments for from_pretrained
            
        Returns:
            Loaded model
        """
        auto_model_class = cls.get_auto_model_class(model_path, subfolder)
        
        # Handle cache_dir separately as it's not a direct parameter
        cache_dir = kwargs.pop('cache_dir', None)
        
        # If subfolder is specified, load from that subfolder
        if subfolder:
            print(f"Loading model {model_path} from subfolder: {subfolder}")
            try:
                if cache_dir:
                    return auto_model_class.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        subfolder=subfolder,
                        device_map=device_map,
                        cache_dir=cache_dir,
                        **kwargs
                    )
                else:
                    return auto_model_class.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        subfolder=subfolder,
                        device_map=device_map,
                        **kwargs
                    )
            except Exception as e:
                print(f"Warning: Failed to load {model_path} from subfolder {subfolder}: {e}")
                print(f"Falling back to loading base model without subfolder...")
        
        # Fall back to loading without subfolder
        if cache_dir:
            return auto_model_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map=device_map,
                cache_dir=cache_dir,
                **kwargs
            )
        else:
            return auto_model_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                device_map=device_map,
                **kwargs
            )