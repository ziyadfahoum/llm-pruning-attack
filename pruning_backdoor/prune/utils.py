import warnings
from dataclasses import dataclass


@dataclass
class PruningConfig:
    pruning_method: str = "wanda"
    sparsity: float = None
    mask_structure: str = None
    calibration_dataset: str = "allenai/c4"  # "json" if local file
    calibration_name: str = "en"
    calibration_split: str = "train"
    calibration_data_files: str = None  # e.g., "dataset/train/clean.jsonl" if local file
    calibration_num_samples: int = 512
    metrics_savedir: str = None
    quantization_scheme: str = None  # e.g., "FP8", "FP8_DYNAMIC"

    def __post_init__(self):
        if self.metrics_savedir is None:
            if self.pruning_method == "both":
                self.metrics_savedir = "metrics_wanda"
            else:
                self.metrics_savedir = f"metrics_{self.pruning_method}"

        # one of sparsity and mask_structure should be provided
        if self.pruning_method != "magnitude" and self.sparsity is None and self.mask_structure is None:
            warnings.warn(f"For {self.pruning_method}, either sparsity or mask_structure must be provided.")
        if self.sparsity is not None and self.mask_structure is not None:
            warnings.warn(
                f"Both sparsity ({self.sparsity}) and mask_structure ({self.mask_structure}) are provided. Only mask_structure will be used"
            )
