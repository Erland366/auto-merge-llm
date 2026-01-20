from .linear import LinearMerging
from .breadcrumbs import BreadcrumbsMerging
from .fisher import FisherMerging, SimplifiedFisherMerging
from .fisher_dataset import DatasetEnabledFisherMerging
from .regmean import RegmeanMerging
from .slerp import SlerpMerging
from .stock import StockMerging
from .task_arithmetic import TaskArithmetic
from .ties import TiesMerging
from .widen import WidenMerging
from .passthrough import PassthroughMerging
from .adamerging import AdaMergingMethod
from .neuromerging import NeuroMerging
from .directional_consensus import DirectionalConsensusMerging

merging_methods_dict = {
    "linear": LinearMerging,
    "breadcrumbs": BreadcrumbsMerging,
    "fisher": FisherMerging,
    "fisher_simple": SimplifiedFisherMerging,
    "fisher_dataset": DatasetEnabledFisherMerging,
    "regmean": RegmeanMerging,
    "slerp": SlerpMerging,
    "stock": StockMerging,
    "task_arithmetic": TaskArithmetic,
    "ties": TiesMerging,
    "widen": WidenMerging,
    "passthrough": PassthroughMerging,
    "adamerging": AdaMergingMethod,
    "neuromerging": NeuroMerging,
    "directional_consensus": DirectionalConsensusMerging,
}


__all__ = ["merging_methods_dict"]
