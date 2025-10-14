from collections import defaultdict
from typing import List, Dict, Any, Iterable

import torch
from torch.utils.data import DataLoader

from .base_method import MergeMethod
from .fisher import FisherMerging


class _SimpleBatchDataset:
    def __init__(self, encodings: Dict[str, torch.Tensor]):
        self.encodings = encodings
        self.length = encodings["input_ids"].size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.encodings.items()}


class _SimpleTrainerWrapper:
    """
    Minimal wrapper to provide the Trainer-like API used by FisherMerging:
    - get_train_dataloader()
    - _prepare_inputs(batch)
    - _train_batch_size
    """

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._train_batch_size = dataloader.batch_size or 8

    def get_train_dataloader(self):
        return self._dataloader

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]):
        # Identity; FisherMerging forwards inputs to model(**inputs)
        return inputs


def _build_shared_dataloader(
    tokenizer,
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 128,
    num_examples: int = 1000,
) -> DataLoader:
    # Truncate/limit examples deterministically
    if num_examples and len(texts) > num_examples:
        texts = texts[:num_examples]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = _SimpleBatchDataset(enc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _sample_texts_from_locales(
    datasets_module,
    dataset_name: str,
    split: str,
    locales: List[str],
    text_column: str,
    per_locale_cap: int,
) -> List[str]:
    texts: List[str] = []
    for locale in locales:
        try:
            ds = datasets_module.load_dataset(dataset_name, locale, split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name} subset {locale}/{split}: {e}")

        if text_column not in ds.column_names:
            raise ValueError(
                f"Text column '{text_column}' not found in subset {locale}. Columns: {ds.column_names}"
            )

        take_n = min(per_locale_cap, len(ds)) if per_locale_cap else len(ds)
        # MASSIVE has field 'utt' for utterance
        texts.extend([str(x) for x in ds.select(range(take_n))[text_column]])
    return texts


class DatasetEnabledFisherMerging(MergeMethod):
    """
    Fisher merging that builds a shared dataloader from HF datasets (e.g., MASSIVE)
    and computes Fisher on a common distribution for all models.

    method_params supports:
      - dataset_config: {
            dataset_name: str (e.g., 'AmazonScience/massive'),
            dataset_split: str (e.g., 'train' or 'validation'),
            text_column: str (e.g., 'utt'),
        }
      - num_fisher_examples: int (total examples across all selected locales)
      - fisher_data_mode: str in {'target', 'sources', 'both'}
      - target_locale: str (e.g., 'af-ZA')
      - source_locales: List[str] (e.g., 5 locales chosen for merging)
      - batch_size: int (default 16)
      - max_seq_length: int (default 128)
      - normalize_fisher_weight: bool
      - minimal_fisher_weight: float
      - fisher_scaling_coefficients: List[float] or None
    """

    def merge(
        self,
        base_model,
        models_to_merge,
        method_params,
        mask_merging=None,
        exclude_param_names_regex=[],
    ):
        # Prepare models
        base_model_dict, merging_model_list = self.prepare_merge(
            base_model, models_to_merge, exclude_param_names_regex
        )

        # Build dataset-driven shared dataloader
        dataset_config: Dict[str, Any] = method_params.get("dataset_config", {})
        dataset_name: str = dataset_config.get("dataset_name", "AmazonScience/massive")
        dataset_split: str = dataset_config.get("dataset_split", "train")
        text_column: str = dataset_config.get("text_column", "utt")

        fisher_data_mode: str = method_params.get("fisher_data_mode", "target")
        target_locale: str = method_params.get("target_locale")
        source_locales: List[str] = method_params.get("source_locales", [])
        batch_size: int = method_params.get("batch_size", 16)
        max_seq_length: int = method_params.get("max_seq_length", 128)
        num_fisher_examples: int = int(method_params.get("num_fisher_examples", 1000))

        if fisher_data_mode not in {"target", "sources", "both"}:
            raise ValueError("fisher_data_mode must be one of {'target','sources','both'}")
        if fisher_data_mode in {"target", "both"} and not target_locale:
            raise ValueError("target_locale is required when fisher_data_mode is 'target' or 'both'")

        try:
            import datasets as hf_datasets
        except Exception as e:
            raise RuntimeError(
                "The 'datasets' library is required for dataset-enabled Fisher merging. "
                "Please install it (pip install datasets)."
            )

        # Decide locales and cap per locale
        locales: List[str] = []
        if fisher_data_mode == "target":
            locales = [target_locale]
        elif fisher_data_mode == "sources":
            locales = list(source_locales)
        else:  # both
            locales = [target_locale] + list(source_locales)

        per_locale_cap = max(1, num_fisher_examples // max(1, len(locales)))
        texts = _sample_texts_from_locales(
            hf_datasets,
            dataset_name=dataset_name,
            split=dataset_split,
            locales=locales,
            text_column=text_column,
            per_locale_cap=per_locale_cap,
        )

        # Shared tokenizer from base model
        tokenizer = base_model_dict["tokenizer"]
        dataloader = _build_shared_dataloader(
            tokenizer,
            texts=texts,
            batch_size=batch_size,
            max_length=max_seq_length,
            num_examples=num_fisher_examples,
        )
        trainer = _SimpleTrainerWrapper(dataloader)

        # Prepare method_params for FisherMerging
        fisher_params = {
            "trainers": [trainer for _ in merging_model_list],
            "nums_fisher_examples": [num_fisher_examples for _ in merging_model_list],
            "normalize_fisher_weight": bool(method_params.get("normalize_fisher_weight", True)),
            "minimal_fisher_weight": float(method_params.get("minimal_fisher_weight", 1e-6)),
            "fisher_scaling_coefficients": method_params.get("fisher_scaling_coefficients", None),
        }

        fisher_merger = FisherMerging()
        merged_result = fisher_merger.merge(
            base_model=base_model,
            models_to_merge=models_to_merge,
            method_params=fisher_params,
            mask_merging=mask_merging,
            exclude_param_names_regex=exclude_param_names_regex,
        )

        return merged_result

    def merge_tensor(
        self,
        base_tensor,
        tensors_to_merge,
        method_params,
        mask_merging=None,
        tensor_name: str = "default",
    ):
        """
        Delegate tensor-wise merging to FisherMerging's implementation. For dataset-enabled
        Fisher we don't compute per-tensor Fisher here, so fall back to the same behavior
        as FisherMerging (simple average) to satisfy the abstract interface.
        """
        fisher_merger = FisherMerging()
        return fisher_merger.merge_tensor(
            base_tensor=base_tensor,
            tensors_to_merge=tensors_to_merge,
            method_params=method_params,
            mask_merging=mask_merging,
            tensor_name=tensor_name,
        )
