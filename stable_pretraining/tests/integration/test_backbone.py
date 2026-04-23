"""Integration tests for backbone models requiring actual model loading."""

import pytest
from transformers import AutoModel, AutoModelForImageClassification


@pytest.mark.integration
@pytest.mark.parametrize(
    "name",
    [
        "alexnet",
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "mobilenet_v2",
        # Add more models as needed, keeping a smaller subset for integration tests
    ],
)
def test_torchvision_embedding_dim(name):
    """Test setting embedding dimension on torchvision models."""
    import torchvision

    import stable_pretraining as spt

    if "vit" in name:
        shape = (10, 3, 224, 224)
    else:
        shape = (10, 3, 512, 512)

    module = torchvision.models.__dict__[name]()
    spt.backbone.set_embedding_dim(
        module,
        dim=20,
        expected_input_shape=shape,
        expected_output_shape=(shape[0], 20),
    )


@pytest.mark.integration
@pytest.mark.download
@pytest.mark.parametrize(
    "name,method,shape",
    [
        ("microsoft/resnet-18", AutoModelForImageClassification, 224),
        pytest.param(
            "timm/swin_tiny_patch4_window7_224.ms_in1k",
            AutoModel,
            224,
            marks=pytest.mark.v1,
        ),
    ],
)
def test_hf_embedding_dim(name, method, shape):
    """Test setting embedding dimension on HuggingFace models."""
    import torch

    import stable_pretraining as spt

    module = method.from_pretrained(name)

    module = spt.backbone.set_embedding_dim(
        module,
        dim=20,
        expected_input_shape=(10, 3, shape, shape),
        expected_output_shape=(10, 20),
    )
    module(torch.zeros(size=(10, 3, shape, shape)))
