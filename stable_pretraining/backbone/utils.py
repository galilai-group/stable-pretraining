import copy
import math
from typing import Union, Iterable, List, Optional, Any, Dict

import torch
import torchvision
from loguru import logger as logging
from torch import nn
from dataclasses import dataclass

# Try to import optional dependencies
try:
    from timm.layers.classifier import ClassifierHead

    _TIMM_AVAILABLE = True
except ImportError:
    ClassifierHead = None
    _TIMM_AVAILABLE = False

try:
    from transformers import TimmWrapperModel, ViTConfig, ViTModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    TimmWrapperModel = None
    ViTConfig = None
    ViTModel = None
    _TRANSFORMERS_AVAILABLE = False


def register_lr_scale_hook(module, lr_scale, weight_decay=0.0):
    """Registers a hook that scales gradients and applies weight decay during backward pass.

    Args:
        module: PyTorch module/layer
        lr_scale: Scaling factor for the learning rate (scales gradients)
        weight_decay: L2 penalty coefficient (default: 0.0)

    Returns:
        module: The same module (for chaining)
    """

    def make_hook(param):
        def gradient_scaling_hook(grad):
            # Add weight decay (L2 regularization)
            if weight_decay != 0.0:
                grad = grad + weight_decay * param.data
            # Scale gradient (equivalent to scaling learning rate)
            return grad * lr_scale

        return gradient_scaling_hook

    for param in module.parameters():
        param.register_hook(make_hook(param))

    return module


def vit_hf(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Vision Transformer using HuggingFace transformers.

    This provides a clean, well-maintained ViT implementation with native support for:
    - Masking via bool_masked_pos parameter
    - Learnable mask token
    - Easy access to CLS and patch tokens

    Args:
        size: Model size - "tiny", "small", "base", or "large"
        patch_size: Patch size (default: 16)
        image_size: Input image size (default: 224)
        pretrained: Load pretrained weights from HuggingFace Hub
        use_mask_token: Whether to include learnable mask token (needed for iBOT)
        **kwargs: Additional ViTConfig parameters

    Returns:
        HuggingFace ViTModel

    Example:
        >>> backbone = vit_hf("tiny", use_mask_token=True)
        >>> x = torch.randn(2, 3, 224, 224)
        >>>
        >>> # Without masking
        >>> output = backbone(x)
        >>> cls_token = output.last_hidden_state[:, 0, :]
        >>> patch_tokens = output.last_hidden_state[:, 1:, :]
        >>>
        >>> # With masking (for iBOT student)
        >>> masks = torch.zeros(2, 196, dtype=torch.bool)
        >>> masks[:, :59] = True  # Mask 30%
        >>> output = backbone(x, bool_masked_pos=masks)
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is required for vit_hf. "
            "Install with: pip install transformers"
        )

    # ViT size configurations (matching timm/DINOv3)
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
        },
        "huge": {
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
        },
    }

    if size not in size_configs:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(size_configs.keys())}"
        )

    config_params = size_configs[size]
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    if pretrained:
        # Try to load pretrained model from HF Hub
        model_name = f"google/vit-{size}-patch{patch_size}-{image_size}"
        logging.info(f"Loading pretrained ViT from {model_name}")
        model = ViTModel.from_pretrained(
            model_name, add_pooling_layer=False, use_mask_token=use_mask_token
        )
    else:
        config = ViTConfig(**config_params)
        model = ViTModel(config, add_pooling_layer=False, use_mask_token=use_mask_token)
        logging.info(f"Created ViT-{size} from scratch with config: {config_params}")

    # IMPORTANT: Set model to always interpolate position encodings for dynamic input sizes
    # This allows processing images of different sizes (e.g., 224x224 global + 96x96 local views)
    # Must be set as instance attribute, not in config
    model.config.interpolate_pos_encoding = True

    return model


class EvalOnly(nn.Module):
    """Wrapper that forces a module to remain in evaluation mode."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone.train(False)
        self.requires_grad_(False)
        self.training = False  # Also set wrapper to eval mode
        assert not self.backbone.training

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to backbone if not found on self
            backbone = self._modules.get("backbone")
            if backbone is not None:
                return getattr(backbone, name)
            raise

    def train(self, mode):
        self.training = False
        return self

    def forward(self, *args, **kwargs):
        if self.backbone.training:
            raise RuntimeError("EvalOnly module is in training mode")
        return self.backbone.forward(*args, **kwargs)


class FeaturesConcat(nn.Module):
    """Aggregates and concatenates features from a dictionary input, then classifies.

    Args:
        names (List[str]): Keys to extract from the input dictionary.
            if not given then we aggregate everything from dict/list
    """

    def __init__(self, agg: callable, names: Union[str, Iterable[str]] = None):
        super().__init__()
        if type(names) is str:
            names = [names]
        self.names = names
        self.agg = agg

    def forward(self, inputs: Union[dict, Iterable]):
        if type(inputs) is dict:
            assert self.names is not None
            tensors = [inputs[n] for n in self.names]
        else:
            tensors = inputs
        reps = []
        for t in tensors:
            reps.append(self.agg(t))
        concat = torch.cat(reps, dim=1)
        return concat

    @staticmethod
    def get_output_shape(
        agg: callable, shapes: Union[list[str], Dict[str, Iterable[int]]]
    ):
        """Given a list of shapes (tuples), returns the expected concatenated shape.

        Assumes all shapes have the same batch size (shapes[0][0]).

        Args:
            shapes (List[Tuple[int]]): List of shapes after aggregation.
            agg (callable): How to aggregate, can be None.

        Returns:
            Tuple[int]: The concatenated shape.
        """
        if not shapes:
            raise ValueError("Shape list is empty.")
        if type(shapes) is dict:
            shapes = list(shapes.values())
        x = [torch.empty(shape, device="meta") for shape in shapes]
        obj = FeaturesConcat(agg)
        out = obj(x)
        return out.shape


@dataclass
class EmbeddingOutput:
    """HuggingFace-style output container for model embeddings.

    Attributes:
        last_hidden_state: The final output from the backbone model.
        hidden_states: Dictionary mapping layer names to their intermediate outputs.
    """

    last_hidden_state: Any
    hidden_states: dict[str, torch.Tensor]

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def keys(self):
        return ["last_hidden_state", "hidden_states"]


class HiddenStateExtractor(nn.Module):
    """Wrapper that captures intermediate embeddings from specified layers.

    Returns outputs in HuggingFace Transformers style with `last_hidden_state`
    for the final backbone output and `hidden_states` for intermediate layers.

    Args:
        backbone: The neural network module to wrap.
        module_names: List of module names to capture (e.g., ['layer1', 'encoder.block1']).
            Supports nested modules using dot notation.

    Returns:
        EmbeddingOutput with:
            - last_hidden_state: Final backbone output
            - hidden_states: Dict mapping module names to their outputs

    Example:
        >>> model = ReturnEmbedding(
        ...     torchvision.models.swin_v2_s(),
        ...     ["features.0", "features.2", "features.4"],
        ... )
        >>> output = model(images)
        >>> output.last_hidden_state  # final output
        >>> output.hidden_states["features.2"]  # intermediate layer

    Raises:
        ValueError: If any module name is not found in the backbone.
    """

    def __init__(self, backbone: nn.Module, module_names: list[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.module_names = list(module_names)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._cache: dict[str, torch.Tensor] = {}

        logging.info(f"ReturnEmbedding: hooking {len(module_names)} modules")

        for name in self.module_names:
            module = self._get_nested_module(name)
            if module is None:
                raise ValueError(f"Module '{name}' not found in backbone.")
            hook = module.register_forward_hook(self._create_hook(name))
            self._hooks.append(hook)

    def forward(self, *args, **kwargs) -> EmbeddingOutput:
        """Run forward pass and return embeddings in HuggingFace style."""
        self._cache.clear()
        output = self.backbone(*args, **kwargs)
        return EmbeddingOutput(
            last_hidden_state=output,
            hidden_states=dict(self._cache),
        )

    def _create_hook(self, name: str):
        """Create a forward hook that caches the output under the given name."""

        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self._cache[name] = output

        return hook

    def _get_nested_module(self, name: str) -> nn.Module | None:
        """Retrieve a nested module by dot-separated path."""
        module = self.backbone
        for attr in name.split("."):
            module = getattr(module, attr, None)
            if module is None:
                return None
        return module

    def remove_hooks(self) -> None:
        """Remove all registered hooks to free resources."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class TeacherStudentWrapper(nn.Module):
    """Backbone wrapper that implements teacher-student distillation via EMA.

    This is a wrapper for backbones that creates a teacher model as an exponential moving average (EMA) of the student model.
    It should be passed as the backbone to stable_pretraining.Module and accessed via
    forward_student() and forward_teacher() methods in your custom forward function.

    The teacher model is updated by taking a running average of the student's
    parameters and buffers. When `ema_coefficient == 0.0`, the teacher and student
    are literally the same object, saving memory but forward passes through the teacher
    will not produce any gradients.

    Usage example:
        backbone = ResNet18()
        wrapped_backbone = TeacherStudentWrapper(backbone)
        module = ssl.Module(
            backbone=wrapped_backbone,
            projector=projector,
            forward=forward_with_teacher_student,
            ...
        )

    Args:
        student (torch.nn.Module): The student model whose parameters will be tracked.
        warm_init (bool, optional): If True, performs an initialization step to match the student's parameters
            immediately. Default is True.
        base_ema_coefficient (float, optional): EMA decay factor at the start of training.
            This value will be updated following a cosine schedule.
            Should be in [0, 1]. A value of 0.0 means the teacher is fully
            updated to the student's parameters on every step, while a value of 1.0 means
            the teacher remains unchanged.
            Default is 0.996.
        final_ema_coefficient (float, optional): EMA decay factor at the end of training.
            Default is 1.
    """

    def __init__(
        self,
        student: nn.Module,
        warm_init: bool = True,
        base_ema_coefficient: float = 0.996,
        final_ema_coefficient: float = 1,
    ):
        if not (0.0 <= base_ema_coefficient <= 1.0) or not (
            0.0 <= final_ema_coefficient <= 1.0
        ):
            error_msg = (
                f"ema_coefficient must be in [0, 1]. Found: "
                f"base_ema_coefficient={base_ema_coefficient}, "
                f"final_ema_coefficient={final_ema_coefficient}."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        super().__init__()
        self.student = student
        # Register EMA coefficients as buffers so they persist through checkpointing
        self.register_buffer("base_ema_coefficient", torch.tensor(base_ema_coefficient))
        self.register_buffer(
            "final_ema_coefficient", torch.tensor(final_ema_coefficient)
        )

        if self.base_ema_coefficient == 0.0 and self.final_ema_coefficient == 0.0:
            # No need to create a teacher network if the EMA coefficient is 0.0.
            self.teacher = student
            # Even when teacher == student, register the buffer for consistency
            self.register_buffer("ema_coefficient", self.base_ema_coefficient.clone())
        else:
            # Create a teacher network with the same architecture as the student.
            if isinstance(student, HiddenStateExtractor):
                self.teacher = HiddenStateExtractor(
                    copy.deepcopy(student.backbone), student.module_names
                )
            else:
                self.teacher = copy.deepcopy(student)
            self.teacher.requires_grad_(False)  # Teacher should not require gradients.

            if warm_init:  # Initialization step to match the student's parameters.
                # Temporarily set ema_coefficient to 0 for warm init
                self.register_buffer("ema_coefficient", torch.zeros(()))
                self.update_teacher()
                # Now set to base value after warm init
                self.ema_coefficient.copy_(self.base_ema_coefficient)
            else:
                self.register_buffer(
                    "ema_coefficient", self.base_ema_coefficient.clone()
                )

    @torch.no_grad
    def update_teacher(self):
        """Perform one EMA update step on the teacher’s parameters.

        The update rule is:
            teacher_param = ema_coefficient * teacher_param
            + (1 - ema_coefficient) * student_param

        This is done in a `no_grad` context to ensure the teacher’s parameters do
        not accumulate gradients, but the student remains fully trainable.

        Everything is updated, including buffers (e.g. batch norm running averages).
        """
        if not self.training:
            return  # We don't update in eval
        elif self.ema_coefficient.item() == 0.0:
            return  # Nothing to update when the teacher is the student.
        elif self.ema_coefficient.item() == 1.0:
            return  # No need to update when the teacher is fixed.

        for teacher_group, student_group in [
            (self.teacher.parameters(), self.student.parameters()),
            (self.teacher.buffers(), self.student.buffers()),
        ]:
            for t, s in zip(teacher_group, student_group):
                ty = t.dtype
                t.mul_(self.ema_coefficient.to(dtype=ty))
                t.add_((1.0 - self.ema_coefficient).to(dtype=ty) * s)

    @torch.no_grad
    def update_ema_coefficient(self, epoch: int, total_epochs: int):
        """Update the EMA coefficient following a cosine schedule.

        The EMA coefficient is updated following a cosine schedule:
            ema_coefficient = final_ema_coefficient -
            0.5 * (final_ema_coefficient - base_ema_coefficient)
            * (1 + cos(epoch / total_epochs * pi))

        Args:
            epoch (int): Current epoch in the training loop.
            total_epochs (int): Total number of epochs in the training loop.
        """
        new_value = self.final_ema_coefficient - 0.5 * (
            self.final_ema_coefficient - self.base_ema_coefficient
        ) * (1 + math.cos(epoch / total_epochs * math.pi))
        # Update the buffer in-place to maintain persistence
        self.ema_coefficient.copy_(new_value)

    def forward_student(self, *args, **kwargs):
        """Forward pass through the student network. Gradients will flow normally."""
        return self.student(*args, **kwargs)

    def forward_teacher(self, *args, **kwargs):
        """Forward pass through the teacher network.

        By default, the teacher network does not require grad.
        If ema_coefficient == 0, then teacher==student,
        so we wrap in torch.no_grad() to ensure no gradients flow.
        """
        with torch.no_grad():
            return self.teacher(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass through either the student or teacher network.

        You can choose which model to run in the default forward.
        Commonly the teacher is evaluated, so we default to that.
        """
        return self.forward_teacher(*args, **kwargs)


def from_torchvision(model_name, low_resolution=False, **kwargs):
    """Load a backbone model.

    If num_classes is provided, the last layer is replaced by a linear layer of
    output size num_classes. Otherwise, the last layer is replaced by an identity layer.

    Args:
        model_name (str): Name of the backbone model. Supported models are:
            - Any model from torchvision.models
            - "Resnet9"
            - "ConvMixer"
        low_resolution (bool, optional): Whether to adapt the resolution of the model (for CIFAR typically).
            By default False.
        **kwargs: Additional keyword arguments for the model. Special handling:
            - in_channels (int): Number of input channels. If provided for ResNet models, the first
              conv layer will be modified to accept this many channels. Default is 3.

    Returns:
        torch.nn.Module: The neural network model.
    """
    # Extract in_channels before passing to torchvision (which doesn't accept it)
    in_channels = kwargs.pop("in_channels", 3)

    try:
        model = torchvision.models.__dict__[model_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}.")

    # Modify conv1 for custom number of input channels and/or low resolution
    if "resnet" in model_name and (in_channels != 3 or low_resolution):
        if low_resolution:
            # Low resolution: smaller kernel, stride=1, no maxpool (for CIFAR)
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
            model.maxpool = nn.Identity()
        else:
            # Full resolution: keep original kernel/stride, just change in_channels
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
    elif low_resolution and "resnet" not in model_name:
        logging.warning(f"Cannot adapt resolution for model: {model_name}.")

    # Handle num_classes parameter as documented
    num_classes = kwargs.get("num_classes", None)
    if num_classes is not None:
        # Replace the last layer with a linear layer of the specified size
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, (nn.ModuleList, nn.Sequential)):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
    else:
        # Replace the last layer with an identity layer for feature extraction
        if hasattr(model, "fc"):
            model.fc = nn.Identity()
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, (nn.ModuleList, nn.Sequential)):
                model.classifier[-1] = nn.Identity()
            else:
                model.classifier = nn.Identity()

    return model


def from_huggingface(model_name, pretrained, attn_implementation="sdpa", **kwargs):
    """Loads a Hugging Face Transformers base model, optionally with pretrained weights, and returns the backbone model.

    This function wraps the Hugging Face `transformers` library to load a model specified by `model_name`.
    It supports loading either pretrained weights or initializing from configuration only. The returned object
    is the model's backbone (`model.base_model`), which is useful for extracting the core architecture
    without task-specific heads.

    Args:
        model_name (str): The Hugging Face model repository identifier or local path. Examples include
            "bert-base-uncased", "facebook/opt-1.3b", or a local directory containing model files.
        pretrained (bool): If True, loads pretrained weights via `AutoModel.from_pretrained`. If False,
            initializes the model from configuration only via `AutoConfig.from_pretrained` and
            `AutoModel.from_config`.
        attn_implementation (str, optional): The attention backend to use. Supported values include
            "sdpa" (default), "eager", "flash_attention_2", etc., as supported by the installed
            version of `transformers` and your hardware. This is forwarded to the underlying model
            constructor.
        **kwargs: Additional keyword arguments forwarded to `AutoModel.from_pretrained` or
            `AutoConfig.from_pretrained`. Common options include:
            - `revision` (str): Model version or branch to use.
            - `cache_dir` (str): Directory to cache downloaded models.
            - `trust_remote_code` (bool): Allow loading custom code from model repo.
            - `torch_dtype` (str or torch.dtype): Data type for model weights.
            - `device_map` (str or dict): Device placement for model parameters.
            - And others supported by Hugging Face Transformers.

    Returns:
        transformers.PreTrainedModel: The base (backbone) model instance, typically accessible via
        `model.base_model`. For some architectures, this may be the model itself.

    Raises:
        ImportError: If the `transformers` library is not installed.
        OSError: If the model or configuration cannot be found or downloaded.
        ValueError: If invalid arguments are provided.
        Exception: Propagates any other exceptions raised by Hugging Face Transformers.

    Notes:
        - The returned `base_model` may differ depending on the architecture. For some models,
          `base_model` is the same as the full model.
        - The availability of certain attention implementations (e.g., "flash_attention_2") depends
          on your hardware, installed libraries, and the version of `transformers`.
        - Ensure that your environment meets the requirements for the selected attention backend.

    Examples:
        >>> # Load a pretrained BERT model with default attention
        >>> model = from_huggingface("bert-base-uncased", pretrained=True)
        >>> # Initialize a model from config only, specifying a revision and device
        >>> model = from_huggingface(
        ...     "facebook/opt-1.3b",
        ...     pretrained=False,
        ...     revision="main",
        ...     device_map="auto",
        ... )
        >>> # Load a pretrained model using flash attention (if supported)
        >>> model = from_huggingface(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     pretrained=True,
        ...     attn_implementation="flash_attention_2",
        ... )
    """
    from transformers import AutoModel, AutoConfig

    if pretrained:
        model = AutoModel.from_pretrained(
            model_name, attn_implementation=attn_implementation, **kwargs
        )
    else:
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        model = AutoModel.from_config(
            config,
            attn_implementation=attn_implementation,
        )
    return model.base_model


def from_timm(model_name, low_resolution=False, **kwargs):
    import timm

    model = timm.create_model(model_name, **kwargs)
    if low_resolution:  # reduce resolution, for instance for CIFAR
        if "resnet" in model_name:
            in_channels = kwargs.get("in_channels", 3)
            model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            )
            model.maxpool = nn.Identity()
        else:
            logging.warning(f"Cannot adapt resolution for model: {model_name}.")
    return model


def _map_shapes(obj: Any) -> Any:
    """Recursively maps a nested structure, replacing torch.Tensor objects with their .shape.

    We preserve the original structure for lists, tuples, dicts, sets, namedtuples, and dataclasses.
    Non-tensor objects are left unchanged.
    """
    import dataclasses

    if isinstance(obj, torch.Tensor):
        return obj.shape
    elif isinstance(obj, dict):
        return {k: _map_shapes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_map_shapes(v) for v in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_map_shapes(v) for v in obj))
    elif isinstance(obj, tuple):
        return tuple(_map_shapes(v) for v in obj)
    elif isinstance(obj, set):
        return {_map_shapes(v) for v in obj}
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _map_shapes(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
            },
        )
    else:
        return obj


def get_output_shape(model: torch.nn.Module, *inputs, **kwargs) -> Any:
    """Infers the output shapes of a PyTorch nn.Module by forwarding fake inputs on the 'meta' device using FakeTensorMode.

    Handles arbitrary nested output structures (lists, dicts, tuples, sets, namedtuples, dataclasses), preserving their
    structure but replacing torch.Tensor objects with their .shape.
    This function temporarily replaces the model's parameters and buffers with fake tensors on the 'meta' device,
    converts all tensor inputs and keyword arguments to 'meta', and runs the forward pass under FakeTensorMode.
    After execution, the original parameters and buffers are restored. No real computation or memory allocation occurs.

    Args:
        model (torch.nn.Module): The PyTorch module to evaluate. Must be on a real device (e.g., CPU).
        *inputs: Positional arguments to pass to the model's forward method. All torch.Tensor inputs are converted to 'meta'.
        **kwargs: Keyword arguments to pass to the model's forward method. All torch.Tensor values are converted to 'meta'.

    Returns:
        Any: The output structure from the model's forward pass, with all torch.Tensor objects replaced by their .shape.
             Non-tensor objects are left unchanged.

    Notes:
        - Supports nested output structures: dict, list, tuple, set, namedtuple, and dataclasses.
        - No real memory is allocated; all tensors are on the 'meta' device.
        - Not thread-safe: concurrent calls may interfere with parameter/buffer swapping.
        - Requires PyTorch 1.11+ for FakeTensorMode.
        - If the model contains custom buffers or state, ensure they are handled appropriately.
        - Raises exceptions if model forward fails or if parameters/buffers cannot be swapped.
        - Non-tensor outputs are returned unchanged.

    Example:
        shapes = get_output_shape_multi_input(model, input1, input2, key1=kwarg1)
        # shapes will have the same structure as the model's output, but with torch.Size in place of tensors.
    """
    from torch.func import functional_call
    import dataclasses

    # Try to use FakeTensorConverter if available (PyTorch 2.x+)
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensorConverter

        fake_mode = FakeTensorMode()
        converter = FakeTensorConverter()

        def to_fake(t):
            return converter.from_real_tensor(fake_mode, t)

    except ImportError:
        # Fallback: just use .to('meta') inside FakeTensorMode
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()

        def to_fake(t):
            return t.to("meta")

    # Prepare fake params and buffers
    params_and_buffers = dict(model.named_parameters())
    params_and_buffers.update(model.named_buffers())
    fake_params_and_buffers = {k: to_fake(v) for k, v in params_and_buffers.items()}

    # Recursively convert all tensor inputs/kwargs to fake/meta
    def convert_inputs(obj):
        if isinstance(obj, torch.Tensor):
            return to_fake(obj)
        elif isinstance(obj, dict):
            return {k: convert_inputs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_inputs(v) for v in obj]
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*(convert_inputs(v) for v in obj))
        elif isinstance(obj, tuple):
            return tuple(convert_inputs(v) for v in obj)
        elif isinstance(obj, set):
            return {convert_inputs(v) for v in obj}
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.replace(
                obj,
                **{
                    f.name: convert_inputs(getattr(obj, f.name))
                    for f in dataclasses.fields(obj)
                },
            )
        else:
            return obj

    fake_inputs = [convert_inputs(inp) for inp in inputs]
    fake_kwargs = {k: convert_inputs(v) for k, v in kwargs.items()}
    with fake_mode:
        output = functional_call(
            model, fake_params_and_buffers, tuple(fake_inputs), fake_kwargs
        )
    return _map_shapes(output)


def set_embedding_dim(
    module,
    dim,
    bias=True,
    expected_input_shape: Optional[Union[tuple, list]] = None,
    expected_output_shape: Optional[Union[tuple, list]] = None,
):
    if isinstance(module, TimmWrapperModel):
        module = module.timm_model

    def embedder(in_features):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, out_features=dim, bias=bias)
        )

    # For models like ResNet.
    if hasattr(module, "fc"):
        in_features = module.fc.in_features
        module.fc = embedder(in_features)
    # For modules like VGG or AlexNet.
    elif hasattr(module, "classifier"):
        if isinstance(module.classifier, nn.ModuleList) or isinstance(
            module.classifier, nn.Sequential
        ):
            in_features = module.classifier[-1].in_features
            module.classifier[-1] = embedder(in_features)
        else:
            in_features = module.classifier.in_features
            module.classifier = embedder(in_features)
    # For modules like ViT.
    elif hasattr(module, "heads"):
        in_features = module.heads.head.in_features
        module.heads.head = embedder(in_features)
    # For modules like Swin Transformer.
    elif hasattr(module, "head") and (
        ClassifierHead is None or not isinstance(module.head, ClassifierHead)
    ):
        in_features = module.head.in_features
        module.head = embedder(in_features)
    else:
        logging.warning(
            f"Unknown module structure for : '{module}'.\n\n"
            "We will use the default's output and attach a "
            "linear module on top."
        )
        if expected_input_shape is None:
            logging.error("Can't do that without `expected_input_shape`")
            raise ValueError("Can't do that without `expected_input_shape`")
        test_input = torch.empty(expected_input_shape, device="meta")
        out_shape = module.to("meta")(test_input)
        in_features = out_shape.flatten(1).size(1)
        embedder = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, out_features=dim, bias=bias)
        )
        return nn.Sequential(module, embedder)

    if expected_input_shape is None:
        logging.warning(
            "No `expected_input_shape` provided, can't verify"
            "the behavior of `set_emebdding_dim`"
        )
    else:
        assert expected_output_shape is not None
        x = torch.empty(expected_input_shape, device="meta")
        # Save original device before moving to meta
        original_device = next(module.parameters()).device
        out = module.to("meta")(x)
        if isinstance(out, tuple):
            assert out[0].shape == expected_output_shape
        elif hasattr(out, "logits"):
            assert out["logits"].shape == expected_output_shape
        else:
            assert out.shape == expected_output_shape
        # Move module back to original device
        # Use to_empty() for meta tensors which have no data
        module = module.to_empty(device=original_device)
    return module


def get_children_modules(
    model: nn.Module, parent_name: str, L: int = 1, partial_match: bool = False
) -> List[str]:
    """Extracts unique module names matching a given parent_name and L submodules.

    Args:
        model: The root nn.Module.
        parent_name: The string or path component to match (e.g., 'blocks').
        L: Number of levels after the parent_name to include in the result.
        partial_match: whether to check with == or in

    Returns:
        Sorted list of unique qualified module names at depth L after the parent_name.
    """
    result: List[str] = []
    for name, _ in model.named_modules():
        parts = name.split(".")
        matches = [
            i
            for i, p in enumerate(parts)
            if (parent_name in p if partial_match else parent_name == p)
        ]
        if not matches:
            continue
        for idx in matches:
            target_idx = idx + L
            if target_idx < len(parts):
                truncated = ".".join(parts[: target_idx + 1])
                if truncated in result:
                    continue
                # Ensure this is a valid submodule
                try:
                    model.get_submodule(truncated)
                    result.append(truncated)
                except AttributeError:
                    continue
            elif L == 0:
                truncated = ".".join(parts[: idx + 1])
                try:
                    model.get_submodule(truncated)
                    result.append(truncated)
                except AttributeError:
                    continue
    return result


class EfficientMaskedTimmViT(nn.Module):
    """Optimized Vision Transformer wrapper that efficiently handles NaN patches.

    This module is designed to work with timm ViT models and provides:
    - Per-sample NaN masking (different NaN patterns per image in batch)
    - Fast path for same masking pattern across batch
    - Support for class tokens (cls_token), distillation tokens (dist_token), and register tokens
    - Compatibility with various timm ViT architectures (vit_*, deit_*, beit_*, etc.)
    - Minimal overhead when no masking is present

    Key Optimizations:
    - Early exit when no NaN patches detected
    - Simpler indexing for same masking patterns
    - Cached batch indices for repeated operations
    - Zero-copy operations where possible

    Args:
        vit: A timm Vision Transformer model instance

    Raises:
        ValueError: If samples have different numbers of NaN patches
        ValueError: If all patches are NaN
        RuntimeError: If the model structure is incompatible

    Example:
        >>> import timm
        >>> vit = timm.create_model(
        ...     "vit_base_patch16_224", pretrained=False, reg_tokens=4
        ... )
        >>> masked_vit = EfficientMaskedTimmViT(vit)
        >>>
        >>> # Create input with some NaN patches
        >>> x = torch.randn(4, 3, 224, 224)
        >>> output = masked_vit(x)

    Performance:
        - Same pattern masking: ~0-5% overhead vs different patterns
        - No masking: <2% overhead vs original model
        - 50% masking: ~1.5x speedup
        - 90% masking: ~2.5-3x speedup

    Note:
        All samples in a batch must have the same NUMBER of NaN patches,
        but the LOCATION of NaN patches can differ per sample.

        Register tokens (DINOv2 style) do NOT receive positional embeddings.
    """

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

        # Cache for batch indices to avoid repeated allocation
        self._batch_indices_cache = {}

        # Validate model has required components
        if not hasattr(vit, "patch_embed"):
            raise RuntimeError(
                "Model must have 'patch_embed' attribute. "
                "This wrapper only supports patch-based ViT models."
            )

        if not hasattr(vit, "blocks"):
            raise RuntimeError(
                "Model must have 'blocks' attribute containing transformer blocks."
            )

        def nan_gradient_hook(grad):
            """Replace NaN gradients with zeros."""
            if torch.isnan(grad).any():
                return torch.nan_to_num(grad)
            return grad

        # Register hook for all parameters
        for name, param in self.vit.patch_embed.named_parameters():
            if param.requires_grad:
                param.register_hook(nan_gradient_hook)
                logging.debug(f"Registered NaN hook for: {name}")

    def _get_num_extra_tokens(self) -> int:
        """Determine the number of extra tokens (cls, dist, register) the model uses.

        Returns:
            int: Number of extra tokens (cls + dist + register)

        Note:
            This counts ALL extra tokens that occupy sequence positions.
            Register tokens don't receive positional embeddings but do occupy positions.
        """
        num_extra = 0

        # CLS token
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            num_extra += 1

        # Distillation token (DeiT)
        if hasattr(self.vit, "dist_token") and self.vit.dist_token is not None:
            num_extra += 1

        # Register tokens (DINOv2 style)
        if hasattr(self.vit, "reg_token") and self.vit.reg_token is not None:
            num_extra += self.vit.reg_token.shape[1]
        elif hasattr(self.vit, "num_reg_tokens"):
            num_extra += self.vit.num_reg_tokens

        return num_extra

    def _get_num_pos_tokens(self) -> int:
        """Get the number of tokens that RECEIVE positional embeddings.

        Returns:
            int: Number of tokens with positional embeddings

        Note:
            With timm's dynamic_img_size=True, register tokens ARE included in pos_embed.
            This method returns CLS + DIST (not register) for non-dynamic models,
            but we need to check pos_embed.shape to know the actual structure.
        """
        num_pos = 0

        # CLS token gets positional embedding
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            num_pos += 1

        # Distillation token gets positional embedding
        if hasattr(self.vit, "dist_token") and self.vit.dist_token is not None:
            num_pos += 1

        # Note: Register tokens may or may not be in pos_embed depending on timm config
        # This is checked dynamically in _interpolate_pos_embed

        return num_pos

    def _add_extra_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Add cls_token, dist_token, and/or register tokens to the sequence.

        Args:
            x: Input tensor of shape (B, N, D) containing patch embeddings

        Returns:
            torch.Tensor: Tensor with extra tokens prepended

        Note:
            Token order: [cls_token, dist_token (if present), register_tokens (if present), patches]
            This matches the timm convention for ViTs with register tokens.
        """
        B = x.shape[0]

        # Add cls_token if present
        if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
            cls_tokens = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add dist_token if present (for DeiT models)
        if hasattr(self.vit, "dist_token") and self.vit.dist_token is not None:
            dist_tokens = self.vit.dist_token.expand(B, -1, -1)
            if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
                x = torch.cat([x[:, :1, :], dist_tokens, x[:, 1:, :]], dim=1)
            else:
                x = torch.cat([dist_tokens, x], dim=1)

        # Add register tokens if present (DINOv2 style)
        if hasattr(self.vit, "reg_token") and self.vit.reg_token is not None:
            reg_tokens = self.vit.reg_token.expand(B, -1, -1)
            # Register tokens come after cls/dist but before patches
            num_prefix = 0
            if hasattr(self.vit, "cls_token") and self.vit.cls_token is not None:
                num_prefix += 1
            if hasattr(self.vit, "dist_token") and self.vit.dist_token is not None:
                num_prefix += 1

            if num_prefix > 0:
                x = torch.cat(
                    [x[:, :num_prefix, :], reg_tokens, x[:, num_prefix:, :]], dim=1
                )
            else:
                x = torch.cat([reg_tokens, x], dim=1)

        return x

    def _get_batch_indices(
        self, B: int, num_keep: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create cached batch indices for gathering operations.

        Args:
            B: Batch size
            num_keep: Number of patches to keep
            device: Device for the tensor

        Returns:
            torch.Tensor: Batch indices of shape (B, num_keep) for advanced indexing

        Note:
            Results are cached to avoid repeated allocations for common batch sizes.
        """
        key = (B, num_keep, device)
        if key not in self._batch_indices_cache:
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_keep)
            self._batch_indices_cache[key] = batch_idx
        return self._batch_indices_cache[key]

    def _subsample_pos_embed_same_pattern(
        self, keep_idx: torch.Tensor, B: int, N: int
    ) -> torch.Tensor:
        """Subsample positional embeddings when all samples have the same mask pattern."""
        pos_embed = self.vit.pos_embed
        num_pos_tokens = self._get_num_pos_tokens()

        # Check if model has register tokens
        num_register_tokens = 0
        if hasattr(self.vit, "reg_token") and self.vit.reg_token is not None:
            num_register_tokens = self.vit.reg_token.shape[1]
        elif hasattr(self.vit, "num_reg_tokens"):
            num_register_tokens = self.vit.num_reg_tokens

        # Interpolate if needed for dynamic image sizes
        pos_embed = self._interpolate_pos_embed(pos_embed, N)

        # Determine positional embedding structure
        # With dynamic_img_size=True, pos_embed may include register tokens
        # Check both: with and without register tokens
        if pos_embed.shape[1] == N + num_pos_tokens + num_register_tokens:
            # pos_embed includes register tokens: [CLS, REG, PATCHES]
            extra_tokens_pos = pos_embed[:, : num_pos_tokens + num_register_tokens, :]
            patch_pos_embed = pos_embed[:, num_pos_tokens + num_register_tokens :, :]

            # Subsample patch positions
            patch_pos_embed = patch_pos_embed[:, keep_idx, :]

            pos_embed = torch.cat(
                [
                    extra_tokens_pos.expand(B, -1, -1),
                    patch_pos_embed.expand(B, -1, -1),
                ],
                dim=1,
            )
        elif pos_embed.shape[1] == N + num_pos_tokens:
            # pos_embed doesn't include register tokens: [CLS, PATCHES]
            extra_tokens_pos = pos_embed[:, :num_pos_tokens, :]
            patch_pos_embed = pos_embed[:, num_pos_tokens:, :]

            # Subsample patch positions
            patch_pos_embed = patch_pos_embed[:, keep_idx, :]

            if num_pos_tokens > 0:
                pos_embed = torch.cat(
                    [
                        extra_tokens_pos.expand(B, -1, -1),
                        patch_pos_embed.expand(B, -1, -1),
                    ],
                    dim=1,
                )
            else:
                pos_embed = patch_pos_embed.expand(B, -1, -1)
        elif pos_embed.shape[1] == N:
            # No extra tokens at all
            patch_pos_embed = pos_embed[:, keep_idx, :]
            pos_embed = patch_pos_embed.expand(B, -1, -1)
        else:
            raise RuntimeError(
                f"Unexpected pos_embed shape after interpolation: {pos_embed.shape}. "
                f"Expected shape[1] to be {N + num_pos_tokens + num_register_tokens}, "
                f"{N + num_pos_tokens}, or {N}"
            )

        return pos_embed

    def _subsample_pos_embed_different_patterns(
        self, keep_indices: torch.Tensor, B: int, N: int, num_keep: int
    ) -> torch.Tensor:
        """Subsample positional embeddings when samples have different mask patterns."""
        pos_embed = self.vit.pos_embed
        num_pos_tokens = self._get_num_pos_tokens()

        # Check if model has register tokens
        num_register_tokens = 0
        if hasattr(self.vit, "reg_token") and self.vit.reg_token is not None:
            num_register_tokens = self.vit.reg_token.shape[1]
        elif hasattr(self.vit, "num_reg_tokens"):
            num_register_tokens = self.vit.num_reg_tokens

        # Interpolate if needed for dynamic image sizes
        pos_embed = self._interpolate_pos_embed(pos_embed, N)

        # Determine positional embedding structure
        # With dynamic_img_size=True, pos_embed may include register tokens
        if pos_embed.shape[1] == N + num_pos_tokens + num_register_tokens:
            # pos_embed includes register tokens: [CLS, REG, PATCHES]
            extra_tokens_pos = pos_embed[:, : num_pos_tokens + num_register_tokens, :]
            patch_pos_embed = pos_embed[:, num_pos_tokens + num_register_tokens :, :]
        elif pos_embed.shape[1] == N + num_pos_tokens:
            # pos_embed doesn't include register tokens: [CLS, PATCHES]
            extra_tokens_pos = pos_embed[:, :num_pos_tokens, :]
            patch_pos_embed = pos_embed[:, num_pos_tokens:, :]
        elif pos_embed.shape[1] == N:
            # No extra tokens
            extra_tokens_pos = None
            patch_pos_embed = pos_embed
        else:
            raise RuntimeError(
                f"Unexpected pos_embed shape after interpolation: {pos_embed.shape}. "
                f"Expected shape[1] to be {N + num_pos_tokens + num_register_tokens}, "
                f"{N + num_pos_tokens}, or {N}"
            )

        # Subsample patch positional embeddings per sample
        patch_pos_embed = patch_pos_embed.expand(B, -1, -1)
        batch_idx = self._get_batch_indices(B, num_keep, keep_indices.device)
        patch_pos_embed = patch_pos_embed[batch_idx, keep_indices, :]

        if extra_tokens_pos is not None:
            extra_tokens_pos = extra_tokens_pos.expand(B, -1, -1)
            pos_embed = torch.cat([extra_tokens_pos, patch_pos_embed], dim=1)
        else:
            pos_embed = patch_pos_embed

        return pos_embed

    def _apply_head(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the classification head to the transformer output.

        Args:
            x: Output from transformer blocks, shape (B, N, D)

        Returns:
            torch.Tensor: Classification logits or features

        Note:
            Handles multiple head types used by different timm models.
        """
        # Try different head application methods used by timm models
        if hasattr(self.vit, "forward_head"):
            # Newer timm models with forward_head method
            return self.vit.forward_head(x)
        elif hasattr(self.vit, "head"):
            # Standard ViT: use cls token (first token)
            if hasattr(self.vit, "fc_norm") and self.vit.fc_norm is not None:
                # Some models apply additional norm before head
                x = self.vit.fc_norm(x[:, 0])
                return self.vit.head(x)
            else:
                return self.vit.head(x[:, 0])
        elif hasattr(self.vit, "head_dist"):
            # DeiT with distillation - has two heads
            x_cls = self.vit.head(x[:, 0])
            x_dist = self.vit.head_dist(x[:, 1])
            if self.training:
                # Return both during training
                return x_cls, x_dist
            else:
                # Average predictions during inference
                return (x_cls + x_dist) / 2
        else:
            # No head - return raw features
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the masked ViT.

        This method implements an optimized forward pass with the following features:
        - Early exit for inputs without NaN patches (fast path)
        - Optimized indexing for same masking patterns across batch
        - Per-sample masking support with advanced indexing
        - Automatic NaN replacement for partial NaN patches
        - Support for register tokens (DINOv2 style)

        Args:
            x: Input tensor, either:
            - Raw images: shape (B, C, H, W)
            - Pre-patchified: shape (B, N, D) where N is number of patches

        Returns:
            torch.Tensor: Model output (logits if head exists, features otherwise)

        Raises:
            ValueError: If samples have different numbers of NaN patches
            ValueError: If all patches are NaN

        Performance Notes:
            - No NaN patches: Uses fast path with <2% overhead
            - Same pattern: Optimized indexing, ~0-5% overhead vs different patterns
            - Different patterns: Uses advanced indexing, ~10-35% slower at high masking
        """
        # Detect if this is a FakeTensor (used for shape inference/tracing)
        is_fake_tensor = (
            x.__class__.__name__ == "FakeTensor"
            or hasattr(x, "fake_mode")
            or "Fake" in type(x).__name__
        )
        if is_fake_tensor or not torch.isnan(x).any():
            return self.vit(x)

        # Patchify if needed
        if x.ndim == 4:  # (B, C, H, W) - raw image
            # Apply patch embedding
            x = self.vit.patch_embed(x)
            # Ensure 3D output (B, N, C)
            if x.ndim == 4:
                # Dynamic: (B, H, W, C) -> (B, H*W, C)
                x = x.flatten(1, 2)
            elif x.ndim != 3:
                raise ValueError(
                    f"Expected patch_embed output to be 3D or 4D, got {x.ndim}D with shape {x.shape}"
                )
        elif x.ndim == 3:  # (B, N, D) - already patchified
            pass
        else:
            raise ValueError(
                f"Input must be 4D (B, C, H, W) image or 3D (B, N, D) patches. "
                f"Got shape: {x.shape}"
            )

        B, N, D = x.shape
        device = x.device
        nan_mask = torch.isnan(x).any(dim=2)  # (B, N)

        # Verify same number of NaN patches across batch
        num_nans = nan_mask.sum(dim=1)
        if not (num_nans == num_nans[0]).all():
            raise ValueError(
                f"All samples must have the same number of NaN patches. "
                f"Got counts: {num_nans.tolist()}"
            )

        num_keep = N - num_nans[0].item()
        if num_keep == 0:
            raise ValueError("All patches are NaN - cannot process input")

        # Check if all samples have the same masking pattern
        same_pattern = (nan_mask == nan_mask[0]).all().item()

        if same_pattern:
            # OPTIMIZED PATH: Same pattern for all samples
            keep_idx = (~nan_mask[0]).nonzero(as_tuple=True)[0]  # (num_keep,)
            x = x[:, keep_idx, :]  # Simple indexing - faster

            # Subsample positional embeddings (optimized)
            pos_embed = self._subsample_pos_embed_same_pattern(keep_idx, B, N)

        else:
            # GENERAL PATH: Different patterns per sample
            keep_indices = self._get_keep_indices_vectorized(nan_mask, num_keep)

            # Gather non-NaN patches (advanced indexing)
            batch_idx = self._get_batch_indices(B, num_keep, device)
            x = x[batch_idx, keep_indices, :]

            # Subsample positional embeddings
            pos_embed = self._subsample_pos_embed_different_patterns(
                keep_indices, B, N, num_keep
            )

        # Replace any remaining NaNs with zeros (partial NaNs in patches)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # Add cls_token, dist_token, and/or register tokens
        x = self._add_extra_tokens(x)

        # Add positional embeddings
        # The subsample methods ensure pos_embed matches x in length
        # (includes register tokens when using dynamic_img_size=True)
        x = x + pos_embed

        # Apply positional dropout if it exists
        if hasattr(self.vit, "pos_drop") and self.vit.pos_drop is not None:
            x = self.vit.pos_drop(x)

        # Apply patch dropout if it exists (some models have this)
        if hasattr(self.vit, "patch_drop") and self.vit.patch_drop is not None:
            x = self.vit.patch_drop(x)

        # Forward through transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        # Apply final norm
        if hasattr(self.vit, "norm") and self.vit.norm is not None:
            x = self.vit.norm(x)

        # Apply head and return
        return self._apply_head(x)

    def clear_cache(self):
        """Clear the cached batch indices.

        Useful if you want to free memory after processing different batch sizes.
        The cache will be rebuilt as needed during forward passes.
        """
        self._batch_indices_cache.clear()

    def _get_keep_indices_vectorized(
        self, nan_mask: torch.Tensor, num_keep: int
    ) -> torch.Tensor:
        """Get keep indices for all samples without Python loops (faster).

        This vectorized approach is ~2-3x faster than iterating over the batch.

        Args:
            nan_mask: Boolean mask indicating NaN patches, shape (B, N)
            num_keep: Number of patches to keep per sample

        Returns:
            torch.Tensor: Keep indices per sample, shape (B, num_keep)

        Note:
            Uses topk instead of nonzero to avoid Python loops. The indices
            are sorted in ascending order.
        """
        B, N = nan_mask.shape
        device = nan_mask.device

        # Create index tensor for all samples
        indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)

        # Mask out NaN positions by setting them to a large value
        indices_masked = indices.float()
        indices_masked[nan_mask] = float(N + 1)  # Larger than any valid index

        # Use topk to get smallest indices (non-NaN positions)
        keep_indices, _ = torch.topk(
            indices_masked,
            k=num_keep,
            dim=1,
            largest=False,  # Get smallest values
            sorted=True,  # Keep sorted for cache friendliness
        )

        return keep_indices.long()

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor, N: int) -> torch.Tensor:
        """Interpolate positional embeddings to match the number of patches.

        This is needed when dynamic_image_size=True and the input size differs
        from the default/training size.

        Args:
            pos_embed: Original positional embeddings, shape (1, N_orig, D)
            N: Target number of patches

        Returns:
            torch.Tensor: Interpolated positional embeddings

        Note:
            When using timm with dynamic_img_size=True and reg_tokens, the pos_embed
            INCLUDES register tokens: [CLS_pos, REG_pos, PATCH_pos]
        """
        num_pos_tokens = self._get_num_pos_tokens()

        # Check if model has register tokens
        num_register_tokens = 0
        if hasattr(self.vit, "reg_token") and self.vit.reg_token is not None:
            num_register_tokens = self.vit.reg_token.shape[1]
        elif hasattr(self.vit, "num_reg_tokens"):
            num_register_tokens = self.vit.num_reg_tokens

        N_orig = pos_embed.shape[1]

        # If already correct size, return as-is
        # Check both possibilities: with and without register tokens
        if (
            N_orig == N + num_pos_tokens + num_register_tokens
            or N_orig == N + num_pos_tokens
            or N_orig == N
        ):
            return pos_embed

        # Determine structure: timm may include register tokens in pos_embed when dynamic_img_size=True
        # Structure can be: [CLS_pos, REG_pos, PATCH_pos] or [CLS_pos, PATCH_pos]

        # Calculate expected position with register tokens
        expected_with_reg = num_pos_tokens + num_register_tokens

        if N_orig > expected_with_reg and num_register_tokens > 0:
            # pos_embed includes register tokens: [CLS, REG, PATCHES]
            extra_tokens_pos = pos_embed[:, :expected_with_reg, :]
            patch_pos_embed = pos_embed[:, expected_with_reg:, :]
        elif num_pos_tokens > 0 and N_orig > num_pos_tokens:
            # pos_embed doesn't include register tokens: [CLS, PATCHES]
            extra_tokens_pos = pos_embed[:, :num_pos_tokens, :]
            patch_pos_embed = pos_embed[:, num_pos_tokens:, :]
        else:
            # No extra tokens
            extra_tokens_pos = None
            patch_pos_embed = pos_embed

        # Calculate grid sizes
        N_orig_patches = patch_pos_embed.shape[1]
        gs_orig = int(N_orig_patches**0.5)
        gs_new = int(N**0.5)

        if gs_orig * gs_orig != N_orig_patches:
            raise RuntimeError(
                f"Original positional embeddings ({N_orig_patches}) don't form a square grid. "
                f"Non-square grids require custom interpolation."
            )

        if gs_new * gs_new != N:
            raise RuntimeError(
                f"Target number of patches ({N}) doesn't form a square grid. "
                f"Non-square grids require custom interpolation."
            )

        # Reshape to 2D grid: (1, N_orig, D) -> (1, D, H_orig, W_orig)
        D = patch_pos_embed.shape[2]
        patch_pos_embed = patch_pos_embed.reshape(1, gs_orig, gs_orig, D).permute(
            0, 3, 1, 2
        )

        # Interpolate using bicubic (same as timm)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(gs_new, gs_new),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )

        # Reshape back: (1, D, H_new, W_new) -> (1, N, D)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, N, D)

        # Recombine with extra token positions
        if extra_tokens_pos is not None:
            pos_embed = torch.cat([extra_tokens_pos, patch_pos_embed], dim=1)
        else:
            pos_embed = patch_pos_embed

        return pos_embed
