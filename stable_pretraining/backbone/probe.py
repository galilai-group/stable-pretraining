import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy
from .utils import register_lr_scale_hook
from typing import List, Union, Optional, Callable, Dict, Any
import torch.nn as nn
from loguru import logger


class MultiHeadAttentiveProbe(torch.nn.Module):
    """A multi-head attentive probe for sequence representations.

    This module applies multiple attention heads to a sequence of embeddings,
    pools the sequence into a fixed-size representation per head, concatenates
    the results, and projects to a set of output classes.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        num_heads (int, optional): Number of attention heads. Default is 4.

    Attributes:
        ln (torch.nn.LayerNorm): Layer normalization applied to the input.
        attn_vectors (torch.nn.Parameter): Learnable attention vectors for each head, shape (num_heads, embedding_dim).
        fc (torch.nn.Linear): Final linear layer mapping concatenated head outputs to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D), where
            N = batch size,
            T = sequence length,
            D = embedding_dim.

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = MultiHeadAttentiveProbe(
        ...     embedding_dim=128, num_classes=10, num_heads=4
        ... )
        >>> x = torch.randn(32, 20, 128)  # batch of 32, sequence length 20
        >>> logits = probe(x)  # shape: (32, 10)
    """

    def __init__(self, embedding_dim: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.ln = torch.nn.LayerNorm(embedding_dim)
        self.attn_vectors = torch.nn.Parameter(torch.randn(num_heads, embedding_dim))
        self.fc = torch.nn.Linear(embedding_dim * num_heads, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (N, T, D)
        x = self.ln(x)
        # Compute attention for each head: (N, num_heads, T)
        attn_scores = torch.einsum("ntd,hd->nht", x, self.attn_vectors)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (N, num_heads, T)
        # Weighted sum for each head: (N, num_heads, D)
        pooled = torch.einsum("ntd,nht->nhd", x, attn_weights)
        pooled = pooled.reshape(x.size(0), -1)  # (N, num_heads * D)
        out = self.fc(pooled)  # (N, num_classes)
        return out


class LinearProbe(torch.nn.Module):
    """Linear using either CLS token or mean pooling with configurable normalization layer.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        pooling (str): Pooling strategy, either 'cls' or 'mean'.
        norm_layer (callable or None): Normalization layer class (e.g., torch.nn.LayerNorm, torch.nn.BatchNorm1d),
            or None for no normalization. Should accept a single argument: normalized_shape or num_features.

    Attributes:
        norm (nn.Module or None): Instantiated normalization layer, or None.
        fc (nn.Linear): Linear layer mapping pooled representation to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D) or (N, D).
            If 3D, pooling and normalization are applied.
            If 2D, input is used directly (no pooling or normalization).

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = LinearProbe(
        ...     embedding_dim=128,
        ...     num_classes=10,
        ...     pooling="mean",
        ...     norm_layer=torch.nn.LayerNorm,
        ... )
        >>> x = torch.randn(32, 20, 128)
        >>> logits = probe(x)  # shape: (32, 10)
        >>> x2 = torch.randn(32, 128)
        >>> logits2 = probe(x2)  # shape: (32, 10)
    """

    def __init__(self, embedding_dim, num_classes, pooling="cls", norm_layer=None):
        super().__init__()
        assert pooling in (
            "cls",
            "mean",
            None,
        ), "pooling must be 'cls' or 'mean' or None"
        self.pooling = pooling
        self.norm = norm_layer(embedding_dim) if norm_layer is not None else None
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (N, T, D) or (N, D)
        if x.ndim == 2:
            # (N, D): no pooling or normalization
            pooled = x
        elif self.pooling == "cls":
            pooled = x[:, 0, :]  # (N, D)
        elif self.pooling == "mean":  # 'mean'
            pooled = x.mean(dim=1)  # (N, D)
        else:
            pooled = x.flatten(1)
        out = self.fc(self.norm(pooled))  # (N, num_classes)
        return out


class AutoLinearClassifier(torch.nn.Module):
    """Linear using either CLS token or mean pooling with configurable normalization layer.

    Args:
        embedding_dim (int): Dimensionality of the input embeddings.
        num_classes (int): Number of output classes.
        pooling (str): Pooling strategy, either 'cls' or 'mean'.
        norm_layer (callable or None): Normalization layer class (e.g., torch.nn.LayerNorm, torch.nn.BatchNorm1d),
            or None for no normalization. Should accept a single argument: normalized_shape or num_features.

    Attributes:
        norm (nn.Module or None): Instantiated normalization layer, or None.
        fc (nn.Linear): Linear layer mapping pooled representation to class logits.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (N, T, D) or (N, D).
            If 3D, pooling and normalization are applied.
            If 2D, input is used directly (no pooling or normalization).

    Returns:
        torch.Tensor: Output logits of shape (N, num_classes).

    Example:
        >>> probe = LinearProbe(
        ...     embedding_dim=128,
        ...     num_classes=10,
        ...     pooling="mean",
        ...     norm_layer=torch.nn.LayerNorm,
        ... )
        >>> x = torch.randn(32, 20, 128)
        >>> logits = probe(x)  # shape: (32, 10)
        >>> x2 = torch.randn(32, 128)
        >>> logits2 = probe(x2)  # shape: (32, 10)
    """

    def __init__(
        self,
        name,
        embedding_dim,
        num_classes,
        pooling=None,
        weight_decay=[0],
        lr_scaling=[1],
        normalization=["none", "norm", "bn"],
        dropout=[0, 0.5],
        label_smoothing=[0, 1],
    ):
        super().__init__()
        assert pooling in (
            "cls",
            "mean",
            None,
        ), "pooling must be 'cls' or 'mean' or None"
        self.pooling = pooling
        self.fc = torch.nn.ModuleDict()
        self.losses = torch.nn.ModuleDict()
        metrics = {}
        for lr in lr_scaling:
            for wd in weight_decay:
                for norm in normalization:
                    for drop in dropout:
                        for ls in label_smoothing:
                            if norm == "bn":
                                layer_norm = torch.nn.BatchNorm1d(embedding_dim)
                            elif norm == "norm":
                                layer_norm = torch.nn.LayerNorm(embedding_dim)
                            else:
                                assert norm == "none"
                                layer_norm = torch.nn.Identity()
                            id = f"{name}_{norm}_{drop}_{ls}_{lr}_{wd}".replace(".", "")
                            self.fc[id] = torch.nn.Sequential(
                                layer_norm,
                                torch.nn.Dropout(drop),
                                torch.nn.Linear(embedding_dim, num_classes),
                            )
                            register_lr_scale_hook(self.fc[id], lr, wd)
                            self.losses[id] = torch.nn.CrossEntropyLoss(
                                label_smoothing=ls / num_classes
                            )
                            metrics[id] = MulticlassAccuracy(num_classes)
        self.metrics = torchmetrics.MetricCollection(
            metrics, prefix="eval/", postfix="_top1"
        )

    def forward(self, x, y=None, pl_module=None):
        # x: (N, T, D) or (N, D)
        if x.ndim == 2:
            # (N, D): no pooling or normalization
            pooled = x
        elif self.pooling == "cls":
            assert x.ndim == 3
            pooled = x[:, 0, :]  # (N, D)
        elif self.pooling == "mean":  # 'mean'
            if x.ndim == 3:
                pooled = x.mean(dim=1)  # (N, D)
            else:
                assert x.ndim == 4
                pooled = x.mean(dim=(2, 3))  # (N, D)
        else:
            pooled = x.flatten(1)
        loss = {}
        for name in self.fc.keys():
            yhat = self.fc[name](pooled)
            loss[f"train/{name}"] = self.losses[name](yhat, y)
            if not self.training:
                self.metrics[name].update(yhat, y)
        if self.training and pl_module:
            pl_module.log_dict(loss, on_step=True, on_epoch=False, rank_zero_only=True)
        elif pl_module:
            pl_module.log_dict(
                self.metrics,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return sum(loss.values())


class AutoTuneMLP(nn.Module):
    """Automatically creates multiple MLP variants with different hyperparameter combinations.

    This module creates a grid of MLPs with different configurations (dropout, normalization,
    learning rates, architectures, etc.) to enable parallel hyperparameter tuning.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        hidden_features: Architecture specification. Can be:
            - List[int]: Single architecture, e.g., [256, 128]
            - List[List[int]]: Multiple architectures, e.g., [[256, 128], [512, 256, 128]]
            - []: Empty list for linear model (no hidden layers)
        name: Base name for this AutoTuneMLP instance
        loss_fn: Loss function to compute loss
        additional_weight_decay: List of weight decay values to try
        lr_scaling: List of learning rate scaling factors to try
        normalization: List of normalization types ['none', 'norm', 'bn']
        dropout: List of dropout rates to try
        activation: List of activation functions ['relu', 'leaky_relu', 'tanh']

    Examples:
        >>> # Single architecture
        >>> model = AutoTuneMLP(128, 10, [256, 128], "clf", nn.CrossEntropyLoss())

        >>> # Multiple architectures
        >>> model = AutoTuneMLP(
        ...     128, 10, [[256], [256, 128], [512, 256]], "clf", nn.CrossEntropyLoss()
        ... )

        >>> # Linear model (no hidden layers)
        >>> model = AutoTuneMLP(128, 10, [], "linear_clf", nn.CrossEntropyLoss())
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Union[List[int], List[List[int]]],
        name: str,
        loss_fn: Callable,
        additional_weight_decay: Union[float, List[float]] = [0],
        lr_scaling: Union[float, List[float]] = [1],
        normalization: Union[str, List[str]] = ["none"],
        dropout: Union[float, List[float]] = [0],
        activation: Union[str, List[str]] = ["relu"],
    ):
        super().__init__()

        logger.info(f"Initializing AutoTuneMLP: {name}")
        logger.debug(f"Input features: {in_features}, Output features: {out_features}")

        self.mlp = nn.ModuleDict()
        self.in_features = in_features
        self.out_features = out_features
        self.loss_fn = loss_fn
        self.name = name

        # Normalize hidden_features to list of lists
        self.hidden_features = self._normalize_architectures(hidden_features)
        logger.debug(f"Architectures to try: {self.hidden_features}")

        # Store hyperparameter configurations
        self.lr_scaling = lr_scaling
        self.additional_weight_decay = additional_weight_decay
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation

        # Generate all MLP variants
        self._build_mlp_variants()

        logger.info(f"Created {len(self.mlp)} MLP variants for {name}")

    @staticmethod
    def _normalize_architectures(
        hidden_features: Union[List[int], List[List[int]]],
    ) -> List[List[int]]:
        """Normalize hidden_features to list of lists format.

        Args:
            hidden_features: Single architecture or list of architectures

        Returns:
            List of architecture configurations

        Examples:
            >>> _normalize_architectures([256, 128])
            [[256, 128]]
            >>> _normalize_architectures([[256], [256, 128]])
            [[256], [256, 128]]
            >>> _normalize_architectures([])
            [[]]
        """
        # Empty list means linear model
        if len(hidden_features) == 0:
            logger.info("Linear model configuration (no hidden layers)")
            return [[]]

        # Check if it's a list of lists or single list
        if isinstance(hidden_features[0], list):
            logger.info(f"Multiple architectures: {len(hidden_features)} variants")
            return hidden_features
        else:
            logger.info(f"Single architecture: {hidden_features}")
            return [hidden_features]

    def _build_mlp_variants(self) -> None:
        """Build all MLP variants based on hyperparameter grid."""
        variant_count = 0

        for arch_idx, arch in enumerate(self.hidden_features):
            arch_name = self._get_arch_name(arch, arch_idx)

            for lr in self._to_list(self.lr_scaling):
                for wd in self._to_list(self.additional_weight_decay):
                    for norm in self._get_norm_layers():
                        for act in self._get_activation_layers():
                            for drop in self._to_list(self.dropout):
                                # Create unique ID for this variant
                                norm_name = self._get_layer_name(norm)
                                act_name = self._get_layer_name(act)
                                variant_id = (
                                    f"{self.name}_{arch_name}_{norm_name}_{act_name}_"
                                    f"drop{drop}_lr{lr}_wd{wd}"
                                ).replace(".", "_")

                                logger.debug(f"Creating variant: {variant_id}")

                                # Build MLP
                                self.mlp[variant_id] = self._create_mlp(
                                    arch, drop, norm, act
                                )

                                # Register learning rate and weight decay hooks
                                self._register_lr_scale_hook(
                                    self.mlp[variant_id], lr, wd
                                )
                                variant_count += 1

        logger.info(f"Successfully built {variant_count} MLP variants")

    @staticmethod
    def _get_arch_name(architecture: List[int], index: int) -> str:
        """Get a readable name for an architecture.

        Args:
            architecture: List of hidden dimensions
            index: Architecture index

        Returns:
            String representation of architecture
        """
        if len(architecture) == 0:
            return "linear"
        return f"arch{index}_" + "x".join(map(str, architecture))

    @staticmethod
    def _to_list(value: Union[Any, List[Any]]) -> List[Any]:
        """Convert single value to list if needed."""
        return value if isinstance(value, (list, tuple)) else [value]

    def _get_norm_layers(self) -> List[Optional[type]]:
        """Get list of normalization layer types."""
        norm_map = {
            "bn": nn.BatchNorm1d,
            "norm": nn.LayerNorm,
            "none": None,
            None: None,
        }

        layers = []
        for case in self._to_list(self.normalization):
            if case not in norm_map:
                logger.warning(f"Unknown normalization: {case}, skipping")
                continue
            layers.append(norm_map[case])

        return layers if layers else [None]

    def _get_activation_layers(self) -> List[type]:
        """Get list of activation layer types."""
        act_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            None: nn.Identity,
        }

        layers = []
        for case in self._to_list(self.activation):
            if case not in act_map:
                logger.warning(f"Unknown activation: {case}, skipping")
                continue
            layers.append(act_map[case])

        return layers if layers else [nn.Identity]

    @staticmethod
    def _get_layer_name(layer: Optional[type]) -> str:
        """Get readable name for a layer type."""
        if layer is None:
            return "none"
        return layer.__name__.lower()

    def _create_mlp(
        self,
        hidden_features: List[int],
        dropout: float,
        norm_layer: Optional[type],
        activation: type,
    ) -> nn.Sequential:
        """Create a single MLP with specified configuration.

        Args:
            hidden_features: List of hidden dimensions (empty for linear model)
            dropout: Dropout rate
            norm_layer: Normalization layer class (or None)
            activation: Activation layer class

        Returns:
            Sequential module containing the MLP
        """
        layers = []

        # Handle linear model (no hidden layers)
        if len(hidden_features) == 0:
            logger.trace("Creating linear model (no hidden layers)")
            layers.append(nn.Linear(self.in_features, self.out_features))
            return nn.Sequential(*layers)

        # Build hidden layers
        in_dim = self.in_features
        for i, hidden_dim in enumerate(hidden_features):
            layers.append(nn.Linear(in_dim, hidden_dim))

            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation())
            layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, self.out_features))

        logger.trace(f"Created MLP with {len(hidden_features)} hidden layers")

        return nn.Sequential(*layers)

    def _register_lr_scale_hook(
        self, module: nn.Module, lr_scale: float, weight_decay: float
    ) -> None:
        """Register learning rate scaling and weight decay for a module.

        Note: This is a placeholder - implement based on your training framework.
        """
        # Store as module attributes for optimizer to access
        module.lr_scale = lr_scale
        module.weight_decay = weight_decay

        logger.trace(f"Registered lr_scale={lr_scale}, weight_decay={weight_decay}")

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all MLP variants.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            y: Optional target tensor for loss computation

        Returns:
            Dictionary with predictions and losses for each variant
            Format: {'pred/{variant_id}': tensor, 'loss/{variant_id}': tensor}
        """
        output = {}

        logger.debug(f"Forward pass with input shape: {x.shape}")

        for variant_id, mlp in self.mlp.items():
            # Get prediction
            pred = mlp(x)
            output[f"pred/{variant_id}"] = pred

            # Compute loss if targets provided
            if y is not None:
                loss = self.loss_fn(pred, y)
                output[f"loss/{variant_id}"] = loss
                logger.trace(f"{variant_id} loss: {loss.item():.4f}")

        logger.debug(f"Computed outputs for {len(self.mlp)} variants")

        return output

    def keys(self) -> List[str]:
        """Get list of all MLP variant names.

        Returns:
            List of variant IDs (strings)

        Example:
            >>> model = AutoTuneMLP(
            ...     128, 10, [[256], [512]], "clf", nn.CrossEntropyLoss()
            ... )
            >>> model.keys()
            ['clf_arch0_256_none_relu_drop0_lr1_wd0', 'clf_arch1_512_none_relu_drop0_lr1_wd0']
        """
        return list(self.mlp.keys())

    def get_variant(self, key: str) -> nn.Module:
        """Get a specific MLP variant by key.

        Args:
            key: Variant ID

        Returns:
            The MLP module

        Raises:
            KeyError: If key doesn't exist
        """
        if key not in self.mlp:
            available = self.keys()
            logger.error(f"Variant '{key}' not found. Available: {available[:5]}...")
            raise KeyError(f"Variant '{key}' not found")

        return self.mlp[key]

    def get_best_variant(
        self, metric_dict: Dict[str, float], lower_is_better: bool = True
    ) -> str:
        """Get the best performing variant based on metrics.

        Args:
            metric_dict: Dictionary mapping variant_id to metric values
            lower_is_better: If True, lower metric is better (e.g., loss).
                           If False, higher is better (e.g., accuracy)

        Returns:
            ID of the best performing variant
        """
        if lower_is_better:
            best_variant = min(metric_dict, key=metric_dict.get)
        else:
            best_variant = max(metric_dict, key=metric_dict.get)

        best_score = metric_dict[best_variant]

        logger.info(f"Best variant: {best_variant} with score: {best_score:.4f}")

        return best_variant

    def num_variants(self) -> int:
        """Get the number of MLP variants."""
        return len(self.mlp)

    def __len__(self) -> int:
        """Get the number of MLP variants."""
        return len(self.mlp)

    def __repr__(self) -> str:
        return (
            f"AutoTuneMLP(name={self.name}, variants={len(self.mlp)}, "
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"architectures={len(self.hidden_features)})"
        )
