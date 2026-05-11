import pytest
import torch
import dataclasses
import collections
import sys
from stable_pretraining.callbacks.utils import detach_tensors
from typing import NamedTuple
from dataclasses import dataclass

pytestmark = pytest.mark.unit


# --- Helper classes for testing ---
class MyNamedTuple(collections.namedtuple("MyNamedTuple", ["a", "b"])):
    """Only for tests."""

    pass


@dataclasses.dataclass(frozen=True)
class FrozenDC:
    """Only for tests."""

    x: any
    y: any


@dataclasses.dataclass
class InitFalseDC:
    """Only for tests."""

    x: any
    y: any = dataclasses.field(init=False, default=42)


class WithDict:
    """Only for tests."""

    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar


class WithSlots:
    """Only for tests."""

    __slots__ = ("foo", "bar")

    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar


@pytest.mark.parametrize(
    "tensor_factory",
    [
        lambda: torch.tensor([1.0, 2.0], requires_grad=True),
        lambda: torch.ones(2, requires_grad=True),
        lambda: torch.randn(2, requires_grad=True),
    ],
)
def test_tensor_detach(tensor_factory):
    t = tensor_factory()
    out = detach_tensors(t)
    assert isinstance(out, torch.Tensor)
    assert out is not t
    assert out.device == t.device
    assert out.dtype == t.dtype
    assert out.shape == t.shape
    assert out.stride() == t.stride()
    assert out.storage().data_ptr() == t.storage().data_ptr()  # No clone
    assert out.grad_fn is None


def test_parameter_detach():
    p = torch.nn.Parameter(torch.randn(3, requires_grad=True))
    out = detach_tensors(p)
    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, torch.nn.Parameter)
    assert out.shape == p.shape
    assert out.grad_fn is None


def test_mapping_subclass():
    class MyMap(dict):
        pass

    t = torch.tensor([1.0], requires_grad=True)
    m = MyMap(a=t)
    out = detach_tensors(m)
    assert isinstance(out, MyMap)
    assert out["a"].grad_fn is None


def test_list_tuple_namedtuple():
    t = torch.tensor([1.0], requires_grad=True)
    item1 = [t, 42]
    tup = (t, 42)
    nt = MyNamedTuple(a=t, b=42)
    for container in (item1, tup, nt):
        out = detach_tensors(container)
        assert out[0].grad_fn is None
        assert out[1] == 42
        if isinstance(container, MyNamedTuple):
            assert isinstance(out, MyNamedTuple)


def test_sets():
    t = torch.tensor([1.0], requires_grad=True)
    s = {t, 42}
    fs = frozenset([t, 42])
    for container in (s, fs):
        out = detach_tensors(container)
        assert any(isinstance(x, torch.Tensor) and x.grad_fn is None for x in out)
        assert 42 in out


def test_dataclasses():
    t = torch.tensor([1.0], requires_grad=True)
    dc = FrozenDC(x=t, y=[t])
    out = detach_tensors(dc)
    assert isinstance(out, FrozenDC)
    assert out.x.grad_fn is None
    assert out.y[0].grad_fn is None


def test_dataclass_init_false():
    t = torch.tensor([1.0], requires_grad=True)
    dc = InitFalseDC(x=t)
    out = detach_tensors(dc)
    assert isinstance(out, InitFalseDC)
    assert out.x.grad_fn is None
    assert out.y == 42


@pytest.mark.skipif("attr" not in sys.modules, reason="attrs not installed")
def test_attrs_class():
    import attr

    @attr.s(frozen=True)
    class AttrCls:
        x = attr.ib()
        y = attr.ib()

    t = torch.tensor([1.0], requires_grad=True)
    a = AttrCls(x=t, y=[t])
    out = detach_tensors(a)
    assert isinstance(out, AttrCls)
    assert out.x.grad_fn is None
    assert out.y[0].grad_fn is None


def test_with_dict_and_slots():
    t = torch.tensor([1.0], requires_grad=True)
    obj = WithDict(foo=t, bar=[t])
    out = detach_tensors(obj)
    assert isinstance(out, WithDict)
    assert out.foo.grad_fn is None
    assert out.bar[0].grad_fn is None
    obj2 = WithSlots(foo=t, bar=[t])
    out2 = detach_tensors(obj2)
    assert isinstance(out2, WithSlots)
    assert out2.foo.grad_fn is None
    assert out2.bar[0].grad_fn is None


def test_shared_references():
    t = torch.tensor([1.0], requires_grad=True)
    item1 = [t, t]
    d = {"a": item1, "b": item1}
    out = detach_tensors(d)
    assert out["a"] is out["b"]
    assert out["a"][0] is out["a"][1]
    assert out["a"][0].grad_fn is None


def test_cycles():
    t = torch.tensor([1.0], requires_grad=True)
    item1 = [t]
    d = {"t": t, "l": item1}
    item1.append(d)  # cycle
    out = detach_tensors(item1)
    assert out[1] is not d
    assert out[1]["l"] is out
    assert out[0].grad_fn is None


def test_no_input_mutation():
    t = torch.tensor([1.0], requires_grad=True)
    assert t.requires_grad
    item1 = [t]
    d = {"t": t, "l": item1}
    item1.append(d)
    orig_ids = (id(t), id(item1), id(d))
    detach_tensors(d)
    # Original objects unchanged
    assert id(t) == orig_ids[0]
    assert id(item1) == orig_ids[1]
    assert id(d) == orig_ids[2]
    assert t.requires_grad
    assert item1[0] is t
    assert item1[1] is d


@pytest.mark.slow
def test_large_structure_performance():
    import time

    t = torch.tensor([1.0], requires_grad=True)
    big = [t for _ in range(10000)]
    nested = [big for _ in range(100)]
    start = time.time()
    out = detach_tensors(nested)
    elapsed = time.time() - start
    assert elapsed < 5.0  # Should run quickly
    # All tensors detached, shared structure preserved
    for cand in out:
        assert cand is out[0]
        for tt in cand:
            assert tt.grad_fn is None
    # No excessive memory use: check that only one new list and one new tensor created per unique input
    assert out[0][0] is out[0][1]


# ==================== Fixtures ====================
@pytest.fixture
def simple_tensor():
    """Single tensor with gradients enabled."""
    return torch.randn(3, 4, requires_grad=True)


@pytest.fixture
def tensor_no_grad():
    """Single tensor without gradients."""
    return torch.randn(3, 4, requires_grad=False)


@pytest.fixture
def tensor_with_grad_fn():
    """Tensor with grad_fn from operations."""
    x = torch.randn(3, 3, requires_grad=True)
    return x * 2 + 1


@pytest.fixture
def list_of_tensors():
    """List containing multiple tensors."""
    return [
        torch.randn(2, 3, requires_grad=True),
        torch.randn(4, 5, requires_grad=True),
        torch.randn(1, 1, requires_grad=False),
    ]


@pytest.fixture
def nested_dict_structure():
    """Complex nested dictionary structure."""
    return {
        "model_outputs": {
            "predictions": [
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
            ],
            "loss": torch.tensor(0.5, requires_grad=True),
            "metadata": {
                "batch_size": 10,
                "features": (
                    torch.randn(10, 20, requires_grad=True),
                    torch.randn(10, 30, requires_grad=True),
                ),
            },
        },
        "stats": [1, 2, 3],
    }


class CustomNamedTuple(NamedTuple):
    """Only used for tests."""

    a: torch.Tensor
    b: int


@dataclass
class CustomDataClass:
    """Only used for tests."""

    tensor: torch.Tensor
    value: int


# ==================== Basic Tensor Tests ====================
class TestBasicTensors:
    """Tests for basic single tensor operations."""

    def test_single_tensor_with_grad(self, simple_tensor):
        """Test detaching a single tensor that requires grad."""
        result = detach_tensors(simple_tensor)

        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad
        assert torch.equal(result, simple_tensor.detach())
        assert result is not simple_tensor

    def test_single_tensor_without_grad(self, tensor_no_grad):
        """Test detaching a tensor that doesn't require grad."""
        result = detach_tensors(tensor_no_grad)

        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad
        assert torch.equal(result, tensor_no_grad)

    def test_already_detached_tensor(self):
        """Test that already detached tensors work correctly."""
        tensor = torch.randn(2, 3).detach()
        result = detach_tensors(tensor)

        assert isinstance(result, torch.Tensor)
        assert not result.requires_grad

    def test_tensor_with_grad_fn(self, tensor_with_grad_fn):
        """Test tensor that has grad_fn from operations."""
        result = detach_tensors(tensor_with_grad_fn)

        assert result.grad_fn is None
        assert not result.requires_grad

    @pytest.mark.parametrize("shape", [(2, 3), (1, 1), (10, 5, 3), (1,)])  # scalar
    def test_various_tensor_shapes(self, shape):
        """Test tensors with various shapes."""
        tensor = torch.randn(*shape, requires_grad=True)
        result = detach_tensors(tensor)

        assert result.shape == shape
        assert not result.requires_grad


# ==================== List Tests ====================
class TestLists:
    """Tests for list structures."""

    def test_list_of_tensors(self, list_of_tensors):
        """Test list containing multiple tensors."""
        result = detach_tensors(list_of_tensors)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(not t.requires_grad for t in result)

    def test_empty_list(self):
        """Test empty list."""
        result = detach_tensors([])

        assert result == []
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "mixed_list",
        [
            [torch.randn(2, 2, requires_grad=True), 42, "string", None, 3.14],
            [1, 2, 3, torch.randn(1, 1, requires_grad=True)],
            [None, None, torch.randn(2, 2, requires_grad=True), None],
        ],
    )
    def test_list_with_mixed_types(self, mixed_list):
        """Test list with tensors and non-tensors."""
        result = detach_tensors(mixed_list)

        assert isinstance(result, list)
        assert len(result) == len(mixed_list)

        for orig, res in zip(mixed_list, result):
            if isinstance(orig, torch.Tensor):
                assert not res.requires_grad
            else:
                assert res == orig

    def test_nested_lists(self):
        """Test deeply nested lists."""
        nested = [
            [torch.randn(2, 2, requires_grad=True)],
            [[torch.randn(3, 3, requires_grad=True)]],
            [[[torch.randn(1, 1, requires_grad=True)]]],
        ]
        result = detach_tensors(nested)

        assert not result[0][0].requires_grad
        assert not result[1][0][0].requires_grad
        assert not result[2][0][0][0].requires_grad

    @pytest.mark.parametrize("depth", [1, 5, 10])
    def test_very_deep_nesting(self, depth):
        """Test very deeply nested structure."""
        data = torch.randn(2, 2, requires_grad=True)
        for _ in range(depth):
            data = [data]

        result = detach_tensors(data)

        current = result
        for _ in range(depth):
            current = current[0]

        assert not current.requires_grad


# ==================== Tuple Tests ====================
class TestTuples:
    """Tests for tuple structures."""

    def test_tuple_of_tensors(self):
        """Test tuple containing tensors."""
        tensors = (
            torch.randn(2, 3, requires_grad=True),
            torch.randn(4, 5, requires_grad=True),
        )
        result = detach_tensors(tensors)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(not t.requires_grad for t in result)

    def test_empty_tuple(self):
        """Test empty tuple."""
        result = detach_tensors(())

        assert result == ()
        assert isinstance(result, tuple)

    @pytest.mark.parametrize(
        "mixed_tuple",
        [
            (torch.randn(2, 2, requires_grad=True), "test", 100),
            (1, 2, torch.randn(1, 1, requires_grad=True)),
            (None, torch.randn(2, 2, requires_grad=True), 3.14),
        ],
    )
    def test_tuple_with_mixed_types(self, mixed_tuple):
        """Test tuple with tensors and non-tensors."""
        result = detach_tensors(mixed_tuple)

        assert isinstance(result, tuple)
        assert len(result) == len(mixed_tuple)


# ==================== Dictionary Tests ====================
class TestDictionaries:
    """Tests for dictionary structures."""

    def test_dict_of_tensors(self):
        """Test dictionary with tensor values."""
        tensor_dict = {
            "a": torch.randn(2, 3, requires_grad=True),
            "b": torch.randn(4, 5, requires_grad=True),
            "c": torch.randn(1, 1, requires_grad=False),
        }
        result = detach_tensors(tensor_dict)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b", "c"}
        assert all(not v.requires_grad for v in result.values())

    def test_empty_dict(self):
        """Test empty dictionary."""
        result = detach_tensors({})

        assert result == {}
        assert isinstance(result, dict)

    def test_dict_with_mixed_values(self):
        """Test dictionary with mixed value types."""
        mixed_dict = {
            "tensor": torch.randn(2, 2, requires_grad=True),
            "int": 42,
            "str": "hello",
            "none": None,
            "list": [1, 2, 3],
        }
        result = detach_tensors(mixed_dict)

        assert isinstance(result, dict)
        assert not result["tensor"].requires_grad
        assert result["int"] == 42
        assert result["str"] == "hello"
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]

    def test_nested_dicts(self):
        """Test nested dictionaries."""
        nested = {
            "level1": {"level2": {"tensor": torch.randn(2, 2, requires_grad=True)}}
        }
        result = detach_tensors(nested)

        assert not result["level1"]["level2"]["tensor"].requires_grad


# ==================== Nested Structure Tests ====================
class TestNestedStructures:
    """Tests for complex nested data structures."""

    def test_list_of_dicts(self):
        """Test list containing dictionaries with tensors."""
        data = [
            {"a": torch.randn(2, 2, requires_grad=True)},
            {"b": torch.randn(3, 3, requires_grad=True)},
        ]
        result = detach_tensors(data)

        assert not result[0]["a"].requires_grad
        assert not result[1]["b"].requires_grad

    def test_dict_of_lists(self):
        """Test dictionary containing lists with tensors."""
        data = {
            "tensors": [
                torch.randn(2, 2, requires_grad=True),
                torch.randn(3, 3, requires_grad=True),
            ]
        }
        result = detach_tensors(data)

        assert not result["tensors"][0].requires_grad
        assert not result["tensors"][1].requires_grad

    def test_complex_nested_structure(self, nested_dict_structure):
        """Test very complex nested structure."""
        result = detach_tensors(nested_dict_structure)

        assert not result["model_outputs"]["predictions"][0].requires_grad
        assert not result["model_outputs"]["predictions"][1].requires_grad
        assert not result["model_outputs"]["loss"].requires_grad
        assert not result["model_outputs"]["metadata"]["features"][0].requires_grad
        assert not result["model_outputs"]["metadata"]["features"][1].requires_grad
        assert result["stats"] == [1, 2, 3]

    def test_dict_of_tuples_of_lists(self):
        """Test dict containing tuples containing lists."""
        data = {
            "key": (
                [torch.randn(2, 2, requires_grad=True)],
                [torch.randn(3, 3, requires_grad=True)],
            )
        }
        result = detach_tensors(data)

        assert not result["key"][0][0].requires_grad
        assert not result["key"][1][0].requires_grad


# ==================== Custom Object Tests ====================
class TestCustomObjects:
    """Tests for custom Python objects."""

    def test_named_tuple(self):
        """Test named tuple containing tensor."""
        nt = CustomNamedTuple(a=torch.randn(2, 2, requires_grad=True), b=42)
        result = detach_tensors(nt)

        if isinstance(result, CustomNamedTuple):
            assert not result.a.requires_grad
            assert result.b == 42
        else:
            assert not result[0].requires_grad
            assert result[1] == 42

    def test_dataclass(self):
        """Test dataclass containing tensor."""
        obj = CustomDataClass(tensor=torch.randn(2, 2, requires_grad=True), value=100)
        result = detach_tensors(obj)

        if hasattr(result, "tensor"):
            assert not result.tensor.requires_grad
            assert result.value == 100

    def test_set_of_primitives(self):
        """Test set with primitive types."""
        data = {1, 2, 3, "test"}
        result = detach_tensors(data)

        assert result == data


# ==================== Primitive Type Tests ====================
class TestPrimitiveTypes:
    """Tests for primitive data types."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (42, 42),
            (3.14, 3.14),
            ("hello", "hello"),
            (True, True),
            (False, False),
        ],
    )
    def test_primitive_types(self, value, expected):
        """Test various primitive types."""
        result = detach_tensors(value)
        assert result == expected


# ==================== Gradient Flow Tests ====================
class TestGradientFlow:
    """Tests for gradient flow blocking."""

    def test_gradient_flow_blocked(self):
        """Test that gradient flow is actually blocked."""
        x = torch.randn(3, 3, requires_grad=True)
        y = x * 2

        result = detach_tensors(y)

        loss = result.sum()
        with pytest.raises(RuntimeError):
            loss.backward()

    def test_original_tensor_unchanged(self):
        """Test that original tensors are not modified."""
        original = torch.randn(2, 3, requires_grad=True)
        original_clone = original.clone()

        detach_tensors({"tensor": original})

        assert original.requires_grad
        assert torch.equal(original, original_clone)

    def test_detach_preserves_values(self):
        """Test that detaching preserves tensor values."""
        tensors = {
            "a": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
            "b": torch.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True),
        }
        result = detach_tensors(tensors)

        assert torch.equal(result["a"], tensors["a"].detach())
        assert torch.equal(result["b"], tensors["b"].detach())


# ==================== Edge Cases ====================
class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_shared_tensor_references(self):
        """Test that the same tensor appears multiple times."""
        shared_tensor = torch.randn(2, 2, requires_grad=True)
        data = {
            "a": shared_tensor,
            "b": shared_tensor,
            "c": [shared_tensor, shared_tensor],
        }

        result = detach_tensors(data)

        assert not result["a"].requires_grad
        assert not result["b"].requires_grad
        assert not result["c"][0].requires_grad
        assert not result["c"][1].requires_grad

    @pytest.mark.parametrize("num_tensors", [10, 100, 1000])
    def test_large_batch_of_tensors(self, num_tensors):
        """Test performance with many tensors."""
        large_list = [
            torch.randn(10, 10, requires_grad=True) for _ in range(num_tensors)
        ]
        result = detach_tensors(large_list)

        assert len(result) == num_tensors
        assert all(not t.requires_grad for t in result)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float64,
            torch.float16,
        ],
    )
    def test_different_tensor_dtypes_float(self, dtype):
        """Test tensors with different floating point data types."""
        tensor = torch.randn(2, 2, dtype=dtype, requires_grad=True)
        result = detach_tensors(tensor)

        assert result.dtype == dtype
        assert not result.requires_grad

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.int32,
            torch.int64,
            torch.bool,
        ],
    )
    def test_different_tensor_dtypes_non_float(self, dtype):
        """Test tensors with non-floating point data types."""
        if dtype == torch.bool:
            tensor = torch.tensor([True, False], dtype=dtype)
        else:
            tensor = torch.randint(0, 10, (2, 2), dtype=dtype)

        result = detach_tensors(tensor)
        assert result.dtype == dtype

    def test_tensor_devices_cpu(self):
        """Test tensors on CPU."""
        cpu_tensor = torch.randn(2, 2, requires_grad=True)
        result = detach_tensors(cpu_tensor)

        assert result.device.type == "cpu"
        assert not result.requires_grad

    @pytest.mark.gpu
    def test_tensor_devices_cuda(self):
        """Test tensors on CUDA."""
        cuda_tensor = torch.randn(2, 2, requires_grad=True, device="cuda")
        result = detach_tensors(cuda_tensor)

        assert result.device.type == "cuda"
        assert not result.requires_grad

    @pytest.mark.gpu
    def test_mixed_devices(self):
        """Test tensors on different devices in same structure."""
        data = {
            "cpu": torch.randn(2, 2, requires_grad=True),
            "cuda": torch.randn(2, 2, requires_grad=True, device="cuda"),
        }
        result = detach_tensors(data)

        assert result["cpu"].device.type == "cpu"
        assert result["cuda"].device.type == "cuda"
        assert not result["cpu"].requires_grad
        assert not result["cuda"].requires_grad


# ==================== Integration Tests ====================
class TestIntegration:
    """Integration tests with PyTorch modules."""

    def test_torch_nn_module_parameters(self):
        """Test with PyTorch nn.Module parameters."""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        data = {"weight": model.weight, "bias": model.bias}
        result = detach_tensors(data)

        assert not result["weight"].requires_grad
        assert not result["bias"].requires_grad

    def test_model_output_dict(self):
        """Test typical model output dictionary."""
        output = {
            "logits": torch.randn(32, 10, requires_grad=True),
            "hidden_states": [
                torch.randn(32, 128, requires_grad=True),
                torch.randn(32, 128, requires_grad=True),
            ],
            "attention_weights": torch.randn(32, 8, 10, 10, requires_grad=True),
        }
        result = detach_tensors(output)

        assert not result["logits"].requires_grad
        assert not result["hidden_states"][0].requires_grad
        assert not result["hidden_states"][1].requires_grad
        assert not result["attention_weights"].requires_grad


# ==================== Performance Tests ====================
@pytest.mark.slow
class TestPerformance:
    """Performance-related tests (marked as slow)."""

    def test_deeply_nested_performance(self):
        """Test performance with very deep nesting (20 levels)."""
        data = torch.randn(2, 2, requires_grad=True)
        for _ in range(20):
            data = {"nested": data}

        result = detach_tensors(data)

        # Navigate to deepest level
        current = result
        for _ in range(20):
            current = current["nested"]

        assert not current.requires_grad

    def test_wide_structure_performance(self):
        """Test performance with wide structures (1000 keys)."""
        data = {f"key_{i}": torch.randn(2, 2, requires_grad=True) for i in range(1000)}
        result = detach_tensors(data)

        assert len(result) == 1000
        assert all(not v.requires_grad for v in result.values())
