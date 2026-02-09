"""
StatsMerging: Statistics-Guided Model Merging via Task-Specific Teacher Distillation.
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from auto_merge_llm.utils import TaskVector, logger
from .base_method import MergeMethod


class StatsMergeLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: List[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class StatsMerging(MergeMethod):
    def merge_tensor(self, base_tensor, tensors_to_merge, method_params, mask_merging=None, tensor_name="default"):
        raise NotImplementedError("StatsMerging uses whole-model merge(), not per-tensor merge_tensor()")

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[]
    ):
        mode = method_params.get("statsmerging_mode", "task_wise")
        svd_rank = method_params.get("stats_svd_rank", 3)
        hidden_dim = method_params.get("stats_hidden_dim", 64)
        num_layers = method_params.get("stats_num_layers", 2)
        lr = method_params.get("stats_lr", 1e-3)
        epochs = method_params.get("stats_epochs", 100)
        normalize = method_params.get("stats_normalize", "softmax")
        dataset_name = method_params.get("dataset_name")
        dataset_split = method_params.get("dataset_split", "validation")
        text_column = method_params.get("text_column", "utt")
        max_seq_length = method_params.get("max_seq_length", 128)
        batch_size = method_params.get("stats_batch_size", 16)
        num_examples = method_params.get("stats_num_examples")
        seed = method_params.get("stats_seed", 42)
        model_locales = method_params.get("model_locales")

        if dataset_name is None:
            raise ValueError("dataset_name is required for statsmerging")
        if model_locales is None:
            raise ValueError("model_locales must be provided for statsmerging")

        logger.info(f"StatsMerging mode: {mode}")

        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, exclude_param_names_regex
        )
        base_model_obj = base_model_dict["model"]

        task_vectors = [
            TaskVector(
                pretrained_model=base_model_obj,
                finetuned_model=model_to_merge["model"],
                exclude_param_names_regex=exclude_param_names_regex,
            )
            for model_to_merge in merging_model_list
        ]

        if len(model_locales) != len(merging_model_list):
            raise ValueError("model_locales must align with models_to_merge order")

        stats_features, layer_names = self._compute_stats_features(task_vectors, mode, svd_rank)
        input_dim = stats_features.shape[-1] if mode == "task_wise" else next(iter(stats_features.values())).shape[-1]
        learner = StatsMergeLearner(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        dataloaders = self._create_task_dataloaders(
            model_locales=model_locales,
            tokenizer=base_model_dict["tokenizer"],
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            text_column=text_column,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            num_examples=num_examples,
            seed=seed,
        )
        if len(dataloaders) != len(merging_model_list):
            raise ValueError("dataloaders must align with models_to_merge order")

        self._train_stats_merge_learner(
            learner=learner,
            base_model_obj=base_model_obj,
            task_vectors=task_vectors,
            stats_features=stats_features,
            layer_names=layer_names,
            teacher_models=[m["model"] for m in merging_model_list],
            dataloaders=dataloaders,
            lr=lr,
            epochs=epochs,
            mode=mode,
            normalize=normalize,
        )

        device = next(learner.parameters()).device
        if mode == "task_wise":
            stats_features = stats_features.to(device)
        else:
            stats_features = {k: v.to(device) for k, v in stats_features.items()}

        coefficients, layer_names = self._compute_coefficients(
            learner=learner,
            stats_features=stats_features,
            layer_names=layer_names,
            mode=mode,
            normalize=normalize,
        )

        with torch.no_grad():
            if mode == "task_wise":
                merged_params = self._merge_task_wise(
                    base_model_obj, task_vectors, coefficients
                )
            else:
                merged_params = self._merge_layer_wise(
                    base_model_obj, task_vectors, coefficients, layer_names
                )

        return self.finalize_merge(
            base_model_obj, base_model_dict, merging_model_list, merged_params
        )

    def _create_task_dataloaders(
        self,
        model_locales: List[str],
        tokenizer,
        dataset_name: str,
        dataset_split: str,
        text_column: str,
        max_seq_length: int,
        batch_size: int,
        num_examples: Optional[int],
        seed: int,
    ) -> List[DataLoader]:
        dataloaders = []
        for locale in model_locales:
            dataset = load_dataset(
                dataset_name,
                locale,
                split=dataset_split,
                trust_remote_code=True,
            )

            if num_examples is not None:
                cap = min(num_examples, len(dataset))
                dataset = dataset.shuffle(seed=seed).select(range(cap))

            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )

            tokenized = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
            tokenized.set_format("torch")

            dataloaders.append(
                DataLoader(
                    tokenized,
                    batch_size=batch_size,
                    shuffle=True,
                )
            )
        return dataloaders

    def _compute_stats_features(
        self,
        task_vectors: List[TaskVector],
        mode: str,
        svd_rank: int,
    ) -> Tuple[Any, Optional[List[str]]]:
        if mode == "task_wise":
            model_features = []
            for tv in task_vectors:
                layer_stats = []
                for _, tensor in tv.task_vector_param_dict.items():
                    layer_stats.append(self._tensor_stats(tensor, svd_rank))
                if not layer_stats:
                    model_features.append(torch.zeros(3 + svd_rank, dtype=torch.float32))
                else:
                    stacked = torch.stack(layer_stats, dim=0)
                    model_features.append(stacked.mean(dim=0))
            return torch.stack(model_features, dim=0), None

        layer_names = list(task_vectors[0].task_vector_param_dict.keys())
        features_by_layer: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        for tv in task_vectors:
            for name in layer_names:
                features_by_layer[name].append(self._tensor_stats(tv.task_vector_param_dict[name], svd_rank))
        for name in layer_names:
            features_by_layer[name] = torch.stack(features_by_layer[name], dim=0)
        return features_by_layer, layer_names

    def _tensor_stats(self, tensor: torch.Tensor, svd_rank: int) -> torch.Tensor:
        if tensor.numel() == 0:
            return torch.zeros(3 + svd_rank, dtype=torch.float32)
        t = tensor.detach().float().cpu()
        mean = t.mean().item()
        var = t.var(unbiased=False).item()
        norm = t.norm().item()
        matrix = self._reshape_for_svd(t)
        try:
            sv = torch.linalg.svdvals(matrix)
        except RuntimeError:
            sv = torch.zeros(0, dtype=torch.float32)
        if sv.numel() >= svd_rank:
            top = sv[:svd_rank]
        else:
            pad = torch.zeros(svd_rank - sv.numel(), dtype=sv.dtype)
            top = torch.cat([sv, pad], dim=0)
        base = torch.tensor([mean, var, norm], dtype=torch.float32)
        return torch.cat([base, top.to(dtype=torch.float32)], dim=0)

    def _reshape_for_svd(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.view(1, 1)
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _compute_coefficients(
        self,
        learner: StatsMergeLearner,
        stats_features: Any,
        layer_names: Optional[List[str]],
        mode: str,
        normalize: str,
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        if mode == "task_wise":
            coeffs = learner(stats_features)
            if normalize == "softmax":
                coeffs = torch.softmax(coeffs, dim=0)
            return coeffs, None

        if layer_names is None:
            layer_names = list(stats_features.keys())
        coeffs = []
        for name in layer_names:
            coeffs.append(learner(stats_features[name]))
        coeffs = torch.stack(coeffs, dim=0)
        if normalize == "softmax":
            coeffs = torch.softmax(coeffs, dim=1)
        return coeffs, layer_names

    def _apply_merged_weights(
        self,
        model,
        base_state_dict: Dict[str, torch.Tensor],
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor,
        mode: str,
        layer_names: Optional[List[str]],
    ):
        if mode == "task_wise":
            for name, param in model.named_parameters():
                merged_delta = torch.zeros_like(param)
                for idx, tv in enumerate(task_vectors):
                    if name in tv.task_vector_param_dict:
                        merged_delta += coefficients[idx] * tv.task_vector_param_dict[name].to(param.device)
                param.data = base_state_dict[name].to(param.device) + merged_delta
            return

        if layer_names is None:
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
        coefficients: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        merged_params = {}
        base_state_dict = dict(base_model.named_parameters())
        for name, base_param in base_state_dict.items():
            merged_delta = torch.zeros_like(base_param)
            for idx, tv in enumerate(task_vectors):
                if name in tv.task_vector_param_dict:
                    merged_delta += coefficients[idx].item() * tv.task_vector_param_dict[name].to(base_param.device)
            merged_params[name] = base_param + merged_delta
        logger.info(f"StatsMerging task-wise coefficients: {coefficients.tolist()}")
        return merged_params

    def _merge_layer_wise(
        self,
        base_model,
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor,
        layer_names: Optional[List[str]],
    ) -> Dict[str, torch.Tensor]:
        merged_params = {}
        base_state_dict = dict(base_model.named_parameters())
        if layer_names is None:
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
        logger.info(f"StatsMerging layer-wise coefficients shape: {coefficients.shape}")
        return merged_params

    def _build_merged_state_dict(
        self,
        base_state_dict: Dict[str, torch.Tensor],
        task_vectors: List[TaskVector],
        coefficients: torch.Tensor,
        mode: str,
        layer_names: Optional[List[str]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build merged state dict preserving gradient flow through coefficients."""
        name_to_layer_idx = (
            {name: idx for idx, name in enumerate(layer_names)}
            if layer_names is not None
            else {}
        )
        merged_state = {}
        for name, base_param in base_state_dict.items():
            delta = torch.zeros_like(base_param)
            if mode == "task_wise":
                for idx, tv in enumerate(task_vectors):
                    if name in tv.task_vector_param_dict:
                        delta = delta + coefficients[idx] * tv.task_vector_param_dict[name].to(device)
            else:
                if name in name_to_layer_idx:
                    layer_idx = name_to_layer_idx[name]
                    for model_idx, tv in enumerate(task_vectors):
                        if name in tv.task_vector_param_dict:
                            delta = delta + coefficients[layer_idx, model_idx] * tv.task_vector_param_dict[name].to(device)
            merged_state[name] = base_param + delta
        return merged_state

    def _train_stats_merge_learner(
        self,
        learner: StatsMergeLearner,
        base_model_obj,
        task_vectors: List[TaskVector],
        stats_features: Any,
        layer_names: Optional[List[str]],
        teacher_models: List[nn.Module],
        dataloaders: List[DataLoader],
        lr: float,
        epochs: int,
        mode: str,
        normalize: str,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learner = learner.to(device)
        if mode == "task_wise":
            stats_features = stats_features.to(device)
        else:
            stats_features = {k: v.to(device) for k, v in stats_features.items()}

        optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        base_model_obj = base_model_obj.to(device)
        base_model_obj.eval()
        base_state_dict = {
            name: param.clone().detach()
            for name, param in base_model_obj.named_parameters()
        }
        # Move task vector params to device once for efficiency
        for tv in task_vectors:
            for name in tv.task_vector_param_dict:
                tv.task_vector_param_dict[name] = tv.task_vector_param_dict[name].to(device)

        prev_loss = float("inf")
        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0
            for teacher, dataloader in zip(teacher_models, dataloaders):
                teacher = teacher.to(device)
                teacher.eval()
                for batch in dataloader:
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        teacher_outputs = teacher(**inputs)
                        pseudo_labels = teacher_outputs.logits.argmax(dim=-1)

                    coefficients, layer_names = self._compute_coefficients(
                        learner=learner,
                        stats_features=stats_features,
                        layer_names=layer_names,
                        mode=mode,
                        normalize=normalize,
                    )

                    # Use functional_call to maintain gradient flow:
                    # loss -> outputs -> merged_state -> coefficients -> learner
                    merged_state = self._build_merged_state_dict(
                        base_state_dict=base_state_dict,
                        task_vectors=task_vectors,
                        coefficients=coefficients,
                        mode=mode,
                        layer_names=layer_names,
                        device=device,
                    )
                    outputs = torch.func.functional_call(base_model_obj, merged_state, args=(), kwargs=inputs)
                    loss = F.cross_entropy(outputs.logits, pseudo_labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_batches += 1
                teacher = teacher.to("cpu")

            scheduler.step()
            avg_loss = total_loss / max(1, total_batches)
            loss_delta = prev_loss - avg_loss
            logger.info(
                f"StatsMerging epoch {epoch + 1}/{epochs}, "
                f"avg loss: {avg_loss:.4f}, "
                f"delta: {loss_delta:+.4f}, "
                f"lr: {scheduler.get_last_lr()[0]:.1e}"
            )
            prev_loss = avg_loss
