"""Video encoders for stable-pretraining.

The submodules live under ``stable_pretraining.backbone.video`` rather than
the flat ``stable_pretraining.backbone`` namespace because the video model
zoo is large enough to deserve its own scope. Import what you need::

    from stable_pretraining.backbone.video import magvit2_base, MAGVIT2Encoder

The naming convention for factory functions mirrors the ViT family
(``<family>_<size>``) so models scale predictably across the package.
"""

from .causal_conv3d import CausalConv3d
from .info import count_parameters, print_video_zoo, summarize
from .cosmos import (
    CosmosCausalTemporalAttention,
    CosmosEncoder,
    CosmosOutput,
    CosmosSpatialAttention,
    cosmos_tiny,
    cosmos_small,
    cosmos_base,
    cosmos_large,
    cosmos_huge,
    cosmos_giant,
    cosmos_gigantic,
)
from .norms import GroupNormPerFrame
from .magvit2 import (
    MAGVIT2Encoder,
    MAGVIT2Output,
    magvit2_tiny,
    magvit2_small,
    magvit2_base,
    magvit2_large,
    magvit2_huge,
    magvit2_giant,
    magvit2_gigantic,
)
from .predrnn import (
    GHU,
    PredRNNv2,
    PredRNNv2Output,
    STLSTMCell,
    predrnn_v2_tiny,
    predrnn_v2_small,
    predrnn_v2_base,
    predrnn_v2_large,
    predrnn_v2_huge,
)
from .videomamba import (
    BiMambaBlock,
    CausalMambaBlock,
    MambaSSMBlock,
    VideoMamba,
    VideoMambaOutput,
    videomamba_tiny,
    videomamba_small,
    videomamba_base,
    videomamba_large,
    videomamba_huge,
    videomamba_giant,
    videomamba_gigantic,
)

__all__ = [
    "CausalConv3d",
    "count_parameters",
    "print_video_zoo",
    "summarize",
    "CosmosCausalTemporalAttention",
    "CosmosEncoder",
    "CosmosOutput",
    "CosmosSpatialAttention",
    "GroupNormPerFrame",
    "cosmos_tiny",
    "cosmos_small",
    "cosmos_base",
    "cosmos_large",
    "cosmos_huge",
    "cosmos_giant",
    "cosmos_gigantic",
    "MAGVIT2Encoder",
    "MAGVIT2Output",
    "magvit2_tiny",
    "magvit2_small",
    "magvit2_base",
    "magvit2_large",
    "magvit2_huge",
    "magvit2_giant",
    "magvit2_gigantic",
    "GHU",
    "PredRNNv2",
    "PredRNNv2Output",
    "STLSTMCell",
    "predrnn_v2_tiny",
    "predrnn_v2_small",
    "predrnn_v2_base",
    "predrnn_v2_large",
    "predrnn_v2_huge",
    "BiMambaBlock",
    "CausalMambaBlock",
    "MambaSSMBlock",
    "VideoMamba",
    "VideoMambaOutput",
    "videomamba_tiny",
    "videomamba_small",
    "videomamba_base",
    "videomamba_large",
    "videomamba_huge",
    "videomamba_giant",
    "videomamba_gigantic",
]
