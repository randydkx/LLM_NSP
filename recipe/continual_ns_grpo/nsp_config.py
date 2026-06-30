from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple, Literal

BLOCK_SIZE: int = 1536

@dataclass
class NullSpaceProjConfig:
    """Encapsulates all Null-Space Projection (NSP) hyperparameters."""

    enable_nullspace_projection: bool = field(
        default=False,
        metadata={"help": "Master switch for enabling null-space projection (NSP)."},
    )

    # Optimizer-related hyperparameters for NSP
    svd_lr: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for NSP-consolidated parameters."},
    )
    bn_lr: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for norm/bias parameters (usually smaller)."},
    )
    svd_thres: float = field(
        default=1e-3,
        metadata={"help": "Eigenvalue threshold for NSP basis selection (sigma < sigma_max * svd_thres)."},
    )
    num_eigen: int = field(
        default=100,
        metadata={"help": "Number of smallest eigenvalues used to form the NSP basis."},
    )

    # Covariance / block settings
    inner_steps_for_update_cov: int = field(
        default=16,
        metadata={"help": "Gap of steps for updating covariance statistics (within an update cycle)."},
    )
    max_feature_width_allow: int = field(
        default=BLOCK_SIZE,
        metadata={"help": "Maximum block size for block-diagonal covariance computation."},
    )
    block_null_space_projection: bool = field(
        default=True,
        metadata={"help": "Whether to use block-wise null-space projection."},
    )

    # Layer range filters
    NSP_mlp_layer_range: Optional[Tuple[int, int]] = field(
        default=(7,12),
        metadata={"help": "Closed interval [start, end] for applying NSP to MLP submodules."},
    )
    NSP_attention_layer_range: Optional[Tuple[int, int]] = field(
        default=(20,23),
        metadata={"help": "Closed interval [start, end] for applying NSP to attention submodules."},
    )

    # Projection update cadence
    update_projections_every: int = field(
        default=4,
        metadata={"help": "Steps between recomputing NSP projection matrices (periodic mode)."},
    )
    projection_update_mode: Literal["periodic", "task_end", "dynamic"] = field(
        default="periodic",
        metadata={"help": "When to update projection matrices: periodic, task_end, or dynamic replay-anchor refresh."},
    )
    cov_update_every: int = field(
        default=1,
        metadata={"help": "Steps between covariance/activation statistic updates within a task."},
    )
    reset_stats_on_task_start: bool = field(
        default=True,
        metadata={"help": "Whether to reset NSP statistics cache at the start of each task."},
    )
    should_update_NSP_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to force NSP recomputation at the first optimization step."},
    )
    nsp_log_details: bool = field(
        default=False,
        metadata={"help": "If True, emit detailed NSP construction logs (ids, sizes, block stats)."},
    )
    nsp_log_level: Literal["off", "basic", "detail", "trace"] = field(
        default="basic",
        metadata={"help": "NSP log verbosity: off/basic/detail/trace. detail implies nsp_log_details."},
    )
    cov_update_batch_size: int = field(
        default=0,
        metadata={"help": "Batch size when feeding masked tokens into update_cov (0 = no chunking)."},
    )
    save_covariance_every: int = field(
        default=100,
        metadata={"help": "Period (steps) to save covariance statistics."},
    )
    resume_fea_in_and_num_for_cov_update: bool = field(
        default=False,
        metadata={"help": "Whether to resume covariance statistics from disk."},
    )
    path_for_fea_in: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load covariance statistics from disk."},
    )
    do_update_statistics: bool = field(
        default=True,
        metadata={"help": "Whether to periodically update NSP projection matrices."},
    )

    # Token-level adaptivity (importance weighting)
    token_importance: bool = field(
        default=True,
        metadata={"help": "Enable token-level importance weighting for covariance updates."},
    )
    token_importance_top_ratio: float = field(
        default=0.3,
        metadata={"help": "Ratio of highest-norm tokens kept for covariance update."},
    )

    # High/low frequency segmenting (feature-level)
    feature_activity_metric: Literal["variance", "mean_abs"] = field(
        default="variance",
        metadata={"help": "Metric for feature activity scoring."},
    )
    high_freq_block_ratio: float = field(
        default=0.25,
        metadata={"help": "Top ratio of blocks to receive NSP projection (others skipped)."},
    )
    high_freq_min_blocks: int = field(
        default=1,
        metadata={"help": "Minimum number of blocks to keep for NSP projection."},
    )

    # Layer-wise adaptivity (Fisher-based)
    layer_adaptivity: bool = field(
        default=False,
        metadata={"help": "Enable layer-wise adaptivity using Fisher information."},
    )
    layer_selection_metric: Literal["fisher", "variance", "mean"] = field(
        default="mean",
        metadata={"help": "Layer selection metric: fisher (backward), variance (activation variance), mean (activation mean)."},
    )
    layer_top_k: int = field(
        default=0,
        metadata={"help": "Select top-k layers by Fisher score (0 disables top-k selection)."},
    )
    layer_top_ratio: float = field(
        default=0.75,
        metadata={"help": "Select top-ratio layers by Fisher score (0 disables ratio selection)."},
    )

    # Soft projection blend
    soft_projection_alpha: float = field(
        default=0.2,
        metadata={"help": "Blend weight: g <- alpha*g + (1-alpha)*g_calibrated."},
    )

    # Anchor data (previous task) settings
    anchor_task_id: int = field(
        default=0,
        metadata={"help": "Task id to use as anchor data for NSP updates (e.g., math task)."},
    )
    anchor_dataset_list: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Optional list of jsonl paths for anchor data (generated from previous tasks). Overrides anchor_task_id if provided.",
        },
    )
    anchor_max_examples: int = field(
        default=2048,
        metadata={"help": "Max anchor examples used per NSP update."},
    )
    anchor_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for anchor forward passes."},
    )
    anchor_shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle anchor data each update."},
    )
    anchor_update_interval: int = field(
        default=100,
        metadata={"help": "Steps between anchor-based NSP recomputation."},
    )
