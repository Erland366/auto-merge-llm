
import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

# --- Add parent directory to path to allow importing the library ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- End Path Setup ---

from methods.base_method import MergeMethod

# --- Test Setup ---

class SimpleModel(nn.Module):
    """A simple model for testing purposes."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer2(self.layer1(x))

class ConcreteMergeMethod(MergeMethod):
    """A concrete implementation of MergeMethod for testing."""
    def merge(self, **kwargs):
        pass
    def merge_tensor(self, **kwargs):
        pass

@pytest.fixture
def setup_models():
    """Pytest fixture to create a base model and a model to merge."""
    torch.manual_seed(0)
    base_model = SimpleModel()
    model_to_merge = SimpleModel()
    
    # Initialize base_model with all ones
    with torch.no_grad():
        for param in base_model.parameters():
            param.fill_(1.0)
            
    # Copy weights from base_model to model_to_merge so they start identical
    model_to_merge.load_state_dict(base_model.state_dict())
    
    return base_model, model_to_merge

# --- Tests ---

def test_mask_params_half_rate(setup_models):
    """
    Tests that a mask_rate of 0.5 correctly reverts the 50% of parameters
    with the smallest magnitude difference back to the base model's values.
    """
    base_model, model_to_merge = setup_models
    merger = ConcreteMergeMethod()

    # Modify the model_to_merge to create a task vector (difference)
    # We will modify the weights of layer1. It has 100 parameters.
    with torch.no_grad():
        # Create a clear, ordered difference
        for i in range(10):
            model_to_merge.layer1.weight.data[i] = 1.0 + (i * 0.1) # Deltas: 0.0, 0.1, ..., 0.9

    # Define the masking strategy
    mask_rate = 0.5  # Mask 50% of the parameters
    mask_merging = {"mask_rate": mask_rate}

    # Apply the masking
    merger.mask_params(
        base_model=base_model,
        models_to_merge=[model_to_merge],
        exclude_param_names_regex=[],
        mask_merging=mask_merging
    )

    # --- Verification ---
    base_params = dict(base_model.named_parameters())
    merged_params = dict(model_to_merge.named_parameters())

    # 1. Verify layer1.weight (the one we modified)
    l1_weight_base = base_params['layer1.weight']
    l1_weight_merged = merged_params['layer1.weight']
    
    # The first 5 rows (50 params) should have been reverted to 1.0
    # because their deltas (0.0, 0.1, 0.2, 0.3, 0.4) are the smallest.
    assert torch.all(l1_weight_merged[:5] == l1_weight_base[:5]).item()
    
    # The last 5 rows (50 params) should remain untouched
    # because their deltas (0.5, 0.6, 0.7, 0.8, 0.9) are the largest.
    assert torch.all(l1_weight_merged[5:] != l1_weight_base[5:]).item()

    # 2. Verify other parameters (biases, layer2) were not changed
    # (since their delta is 0, they are technically masked but the value is the same)
    assert torch.all(merged_params['layer1.bias'] == base_params['layer1.bias']).item()
    assert torch.all(merged_params['layer2.weight'] == base_params['layer2.weight']).item()
    assert torch.all(merged_params['layer2.bias'] == base_params['layer2.bias']).item()


def test_mask_params_zero_rate(setup_models):
    """Tests that a mask_rate of 0.0 results in no changes."""
    base_model, model_to_merge = setup_models
    merger = ConcreteMergeMethod()
    
    original_state_dict = model_to_merge.state_dict()

    mask_merging = {"mask_rate": 0.0}
    merger.mask_params(
        base_model=base_model,
        models_to_merge=[model_to_merge],
        exclude_param_names_regex=[],
        mask_merging=mask_merging
    )

    for name, param in model_to_merge.named_parameters():
        assert torch.equal(param.data, original_state_dict[name])

def test_mask_params_none(setup_models):
    """Tests that a mask_merging of None results in no changes."""
    base_model, model_to_merge = setup_models
    merger = ConcreteMergeMethod()
    
    original_state_dict = model_to_merge.state_dict()

    merger.mask_params(
        base_model=base_model,
        models_to_merge=[model_to_merge],
        exclude_param_names_regex=[],
        mask_merging=None
    )

    for name, param in model_to_merge.named_parameters():
        assert torch.equal(param.data, original_state_dict[name])

def test_mask_params_exclude_regex(setup_models):
    """Tests that parameters matching the exclude regex are not masked."""
    base_model, model_to_merge = setup_models
    merger = ConcreteMergeMethod()

    # Make a change to layer1 and layer2
    with torch.no_grad():
        model_to_merge.layer1.weight.data += 0.1
        model_to_merge.layer2.weight.data += 0.5 # This should not be masked

    mask_merging = {"mask_rate": 0.9} # High rate to ensure layer1 would be masked
    
    merger.mask_params(
        base_model=base_model,
        models_to_merge=[model_to_merge],
        exclude_param_names_regex=[r"layer2.*"], # Exclude layer2
        mask_merging=mask_merging
    )

    # layer1 weights should be reverted back to base model values
    assert torch.allclose(model_to_merge.layer1.weight.data, base_model.layer1.weight.data)
    
    # layer2 weights should NOT be reverted
    assert not torch.allclose(model_to_merge.layer2.weight.data, base_model.layer2.weight.data)
