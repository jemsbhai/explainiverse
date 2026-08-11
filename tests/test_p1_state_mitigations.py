"""Falsifiable P1 gates for model, explainer, tensor, layout, and adapter state."""

from __future__ import annotations

import inspect
import random
import threading
from types import MethodType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from explainiverse.adapters import PyTorchAdapter
from explainiverse.core.explainer import BaseExplainer
from explainiverse.explainers._validation import normalize_classifier_outputs
from explainiverse.explainers.gradient import (
    ConceptActivationVector,
    DeepLIFTShapExplainer,
    EigenCAMExplainer,
    GradCAMExplainer,
    HiResCAMExplainer,
    IntegratedGradientsExplainer,
    TCAVExplainer,
)
from explainiverse.explainers.gradient import cam_variants as cam_variants_module
from explainiverse.explainers.gradient import gradcam as gradcam_module
from explainiverse.explainers.gradient._model_state import ModelStateIsolationError
from explainiverse.explainers.gradient.saliency import SaliencyExplainer


class _OwnedStateProtocol:
    """Own every mutable exclusion exercised by ``_AdversarialStateModel``."""

    def snapshot(self, module):
        return {
            "counter": module.counter,
            "parameter": module.weight,
            "parameter_value": module.weight.detach().clone(),
            "buffer": module.cache,
            "buffer_value": module.cache.detach().clone(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }

    def restore(self, module, snapshot):
        module.counter = snapshot["counter"]
        module.weight = snapshot["parameter"]
        module.cache = snapshot["buffer"]
        with torch.no_grad():
            module.weight.copy_(snapshot["parameter_value"])
            module.cache.copy_(snapshot["buffer_value"])
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])


class _AdversarialStateModel(nn.Module):
    def __init__(self, generator, *, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
        self.register_buffer("cache", torch.tensor([3.0], dtype=torch.float64))
        self.generator = generator
        self.counter = 0
        self.fail = fail

    def forward(self, inputs):
        output = inputs @ self.weight.t()
        _ = torch.rand(1, generator=self.generator)
        random.random()
        np.random.random()
        self.counter += 1
        self.weight = nn.Parameter(self.weight.detach().clone().add_(10.0))
        self.cache = self.cache.detach().clone().add_(20.0)
        if self.fail:
            raise RuntimeError("adversarial forward failure")
        return output


def _adversarial_fingerprint(module):
    return {
        "python attribute 'counter'": module.counter,
        "parameter binding 'weight'": id(module.weight),
        "parameter value 'weight'": module.weight.detach().clone(),
        "buffer binding 'cache'": id(module.cache),
        "buffer value 'cache'": module.cache.detach().clone(),
        "Python RNG": random.getstate(),
        "NumPy RNG": np.random.get_state(),
        "custom torch.Generator": module.generator.get_state().clone(),
    }


@pytest.mark.parametrize("fail", [False, True])
def test_opt_in_state_protocol_restores_every_declared_exclusion(fail):
    generator = torch.Generator(device="cpu").manual_seed(9123)
    model = _AdversarialStateModel(generator, fail=fail)
    adapter = PyTorchAdapter(
        model,
        task="regression",
        model_generators=[generator],
        model_state_protocol=_OwnedStateProtocol(),
        model_state_fingerprint=_adversarial_fingerprint,
    )
    before = _adversarial_fingerprint(model)
    explainer = SaliencyExplainer(adapter, ["x"], absolute_value=False)

    if fail:
        with pytest.raises(RuntimeError, match="adversarial forward failure"):
            explainer.explain(np.array([4.0], dtype=np.float64), target_class=0)
    else:
        explanation = explainer.explain(np.array([4.0], dtype=np.float64), target_class=0)
        np.testing.assert_array_equal(explanation.explanation_data["attributions_raw"], [2.0])

    after = _adversarial_fingerprint(model)
    assert set(before) == set(after)
    for name in before:
        left, right = before[name], after[name]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), name
        elif isinstance(left, tuple) and len(left) == 5 and isinstance(left[1], np.ndarray):
            assert left[0] == right[0] and np.array_equal(left[1], right[1]), name
            assert left[2:] == right[2:], name
        else:
            assert left == right, name


class _UndeclaredMutation(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def forward(self, inputs):
        self.counter += 1
        return inputs.sum(dim=1, keepdim=True)


def test_fingerprint_only_mode_fails_before_returning_unsupported_mutation():
    model = _UndeclaredMutation()
    adapter = PyTorchAdapter(
        model,
        task="regression",
        model_state_fingerprint=lambda module: {"python attribute 'counter'": module.counter},
    )

    with pytest.raises(ModelStateIsolationError, match="python attribute 'counter'"):
        SaliencyExplainer(adapter, ["x"]).explain(np.array([1.0]), target_class=0)


class _HiddenRegisteredStateChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
        self.register_buffer("cache", torch.tensor([3.0], dtype=torch.float64))

    def forward(self, inputs):
        with torch.no_grad():
            self.cache.add_(10.0)
            self.weight.grad.add_(20.0)
        return inputs @ self.weight.t()


class _HidingStateTraversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _HiddenRegisteredStateChild()

    def modules(self):
        yield self

    def parameters(self, recurse=True):
        del recurse
        return iter(())

    def buffers(self, recurse=True):
        del recurse
        return iter(())

    def forward(self, inputs):
        return self.child(inputs)


def test_overridden_public_traversal_cannot_hide_registered_state_restoration():
    model = _HidingStateTraversal()
    adapter = PyTorchAdapter(model, task="regression")
    model.train(True)
    original_gradient = torch.tensor([[5.0]], dtype=torch.float64)
    model.child.weight.grad = original_gradient
    buffer_before = model.child.cache.detach().clone()

    explanation = SaliencyExplainer(adapter, ["x"], absolute_value=False).explain(
        np.array([4.0], dtype=np.float64), target_class=0
    )

    np.testing.assert_array_equal(explanation.explanation_data["attributions_raw"], [2.0])
    assert model.training is True and model.child.training is True
    torch.testing.assert_close(model.child.cache, buffer_before)
    assert model.child.weight.grad is original_gradient
    torch.testing.assert_close(model.child.weight.grad, torch.tensor([[5.0]], dtype=torch.float64))


class _RngCallbackProtocol:
    def __init__(self, generator, fail_stage=None):
        self.generator = generator
        self.fail_stage = fail_stage

    def _consume(self):
        torch.rand(3)
        torch.rand(3, generator=self.generator)

    def snapshot(self, module):
        del module
        self._consume()
        if self.fail_stage == "snapshot":
            raise RuntimeError("snapshot callback failure")
        return None

    def restore(self, module, snapshot):
        del module, snapshot
        self._consume()
        if self.fail_stage == "restore":
            raise RuntimeError("restore callback failure")


class _RngFingerprint:
    def __init__(self, generator, fail_stage=None):
        self.generator = generator
        self.fail_stage = fail_stage
        self.calls = 0

    def __call__(self, module):
        del module
        self.calls += 1
        torch.rand(3)
        torch.rand(3, generator=self.generator)
        if self.fail_stage == "fingerprint_before" and self.calls == 1:
            raise RuntimeError("fingerprint-before callback failure")
        if self.fail_stage == "fingerprint_after" and self.calls == 2:
            raise RuntimeError("fingerprint-after callback failure")
        return {"stable callback value": 1}


class _RngCallbackModel(nn.Module):
    def __init__(self, generator, fail=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
        self.generator = generator
        self.fail = fail

    def forward(self, inputs):
        torch.rand(2)
        torch.rand(2, generator=self.generator)
        if self.fail:
            raise RuntimeError("model callback-boundary failure")
        return inputs @ self.weight.t()


@pytest.mark.parametrize(
    "fail_stage",
    [None, "snapshot", "restore", "fingerprint_before", "fingerprint_after", "model"],
)
def test_rng_consuming_state_callbacks_never_leak_default_or_injected_generator(fail_stage):
    generator = torch.Generator(device="cpu").manual_seed(8123)
    protocol = _RngCallbackProtocol(generator, fail_stage=fail_stage)
    fingerprint = _RngFingerprint(generator, fail_stage=fail_stage)
    adapter = PyTorchAdapter(
        _RngCallbackModel(generator, fail=fail_stage == "model"),
        task="regression",
        model_generators=[generator],
        model_state_protocol=protocol,
        model_state_fingerprint=fingerprint,
    )
    torch.manual_seed(4123)
    default_before = torch.random.get_rng_state().clone()
    custom_before = generator.get_state().clone()

    if fail_stage is None:
        SaliencyExplainer(adapter, ["x"]).explain(np.array([4.0], dtype=np.float64), target_class=0)
    else:
        expected = {
            "snapshot": "snapshot callback failure",
            "restore": "declared model_state_protocol",
            "fingerprint_before": "fingerprint-before callback failure",
            "fingerprint_after": "fingerprint-after callback failure",
            "model": "model callback-boundary failure",
        }[fail_stage]
        with pytest.raises((RuntimeError, ModelStateIsolationError), match=expected):
            SaliencyExplainer(adapter, ["x"]).explain(
                np.array([4.0], dtype=np.float64), target_class=0
            )

    assert torch.equal(torch.random.get_rng_state(), default_before)
    assert torch.equal(generator.get_state(), custom_before)


class _BarrierGradientAdapter:
    task = "regression"

    def __init__(self, *, fail_first=False):
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.calls = 0
        self.fail_first = fail_first

    def predict_with_gradients(self, values, target_class=None):
        del target_class
        self.calls += 1
        if self.calls == 1:
            self.first_entered.set()
            if not self.release_first.wait(timeout=3.0):
                raise RuntimeError("test did not release first IG call")
            if self.fail_first:
                raise RuntimeError("first IG call failed")
        return np.zeros((len(values), 1)), np.ones_like(values)


@pytest.mark.parametrize("fail_first", [False, True])
def test_shared_ig_first_call_commit_is_serial_and_atomic(fail_first):
    adapter = _BarrierGradientAdapter(fail_first=fail_first)
    explainer = IntegratedGradientsExplainer(adapter, n_steps=1)
    outcomes = []
    errors = []

    def run(values):
        try:
            outcomes.append(explainer.explain(np.asarray(values), target_class=0))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=run, args=([1.0, 2.0],), name="ig-first")
    second = threading.Thread(target=run, args=([1.0, 2.0, 3.0],), name="ig-second")
    first.start()
    assert adapter.first_entered.wait(timeout=2.0)
    second.start()
    assert adapter.calls == 1
    adapter.release_first.set()
    first.join(timeout=3.0)
    second.join(timeout=3.0)

    assert not first.is_alive() and not second.is_alive()
    if fail_first:
        assert explainer.input_shape == (3,)
        assert adapter.calls == 2
        assert len(outcomes) == 1
        assert any("first IG call failed" in str(error) for error in errors)
    else:
        assert explainer.input_shape == (2,)
        assert adapter.calls == 1
        assert len(outcomes) == 1
        assert any("input_shape exactly" in str(error) for error in errors)


class _OneShotBlockingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
        self.forward_entered = threading.Event()
        self.release_forward = threading.Event()
        self.block_next = True

    def forward(self, inputs):
        if self.block_next:
            self.block_next = False
            self.forward_entered.set()
            if not self.release_forward.wait(timeout=3.0):
                raise RuntimeError("test did not release model forward")
        return self.linear(inputs)


def test_predict_and_device_move_follow_one_serial_adapter_schedule():
    model = _OneShotBlockingModel()
    adapter = PyTorchAdapter(model, task="regression")
    predictions = []
    errors = []
    move_started = threading.Event()
    move_done = threading.Event()

    def predict():
        try:
            predictions.append(adapter.predict(np.array([[3.0]], dtype=np.float64)))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    def move():
        move_started.set()
        try:
            adapter.to("cpu")
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            move_done.set()

    prediction_thread = threading.Thread(target=predict, name="adapter-predict")
    move_thread = threading.Thread(target=move, name="adapter-move")
    prediction_thread.start()
    assert model.forward_entered.wait(timeout=2.0)
    move_thread.start()
    assert move_started.wait(timeout=2.0)
    assert not move_done.wait(timeout=0.1)
    model.release_forward.set()
    prediction_thread.join(timeout=3.0)
    move_thread.join(timeout=3.0)

    assert not prediction_thread.is_alive() and not move_thread.is_alive()
    assert errors == []
    np.testing.assert_array_equal(predictions, [[[6.0]]])
    assert adapter.device.type == "cpu"


@pytest.mark.parametrize(
    ("mode_operation", "initial_training", "expected_training"),
    [
        ("train_mode", False, True),
        ("eval_mode", True, False),
    ],
)
def test_explanation_and_mode_change_follow_one_serial_adapter_schedule(
    mode_operation, initial_training, expected_training
):
    model = _OneShotBlockingModel()
    adapter = PyTorchAdapter(model, task="regression")
    adapter.train_mode() if initial_training else adapter.eval_mode()
    errors = []
    explanation_done = threading.Event()
    mode_started = threading.Event()
    mode_done = threading.Event()

    def explain():
        try:
            SaliencyExplainer(adapter, ["x"]).explain(
                np.array([3.0], dtype=np.float64), target_class=0
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            explanation_done.set()

    def change_mode():
        mode_started.set()
        try:
            getattr(adapter, mode_operation)()
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            mode_done.set()

    explanation_thread = threading.Thread(target=explain, name="adapter-explanation")
    mode_thread = threading.Thread(target=change_mode, name="adapter-mode")
    explanation_thread.start()
    assert model.forward_entered.wait(timeout=2.0)
    mode_thread.start()
    assert mode_started.wait(timeout=2.0)
    assert not mode_done.wait(timeout=0.1)
    model.release_forward.set()
    explanation_thread.join(timeout=3.0)
    mode_thread.join(timeout=3.0)

    assert not explanation_thread.is_alive() and not mode_thread.is_alive()
    assert explanation_done.is_set() and mode_done.is_set()
    assert errors == []
    assert model.training is expected_training
    assert model.linear.training is expected_training


def _explainer_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _explainer_subclasses(subclass)


def test_every_loaded_builtin_public_explainer_operation_is_instance_synchronized():
    missing = []
    for cls in _explainer_subclasses(BaseExplainer):
        if not cls.__module__.startswith("explainiverse.explainers."):
            continue
        for name, attribute in cls.__dict__.items():
            if name.startswith("_") or not inspect.isfunction(attribute):
                continue
            if not getattr(attribute, "__explainiverse_instance_synchronized__", False):
                missing.append(f"{cls.__module__}.{cls.__name__}.{name}")
    assert missing == []


class _BackgroundReadProbe(DeepLIFTShapExplainer):
    """Exercise the real background mutator without requiring Captum."""

    def __init__(self):
        BaseExplainer.__init__(self, None)
        self.feature_names = ["x"]
        self.n_background_samples = 10
        self._background_data = np.array([[1.0]])
        self.read_entered = threading.Event()
        self.release_read = threading.Event()

    def explain(self):
        snapshot = self._background_data.copy()
        self.read_entered.set()
        if not self.release_read.wait(timeout=3.0):
            raise RuntimeError("test did not release background read")
        return snapshot


def test_deepshap_background_mutation_cannot_interleave_with_read():
    explainer = _BackgroundReadProbe()
    reads = []
    errors = []
    writer_started = threading.Event()
    writer_done = threading.Event()

    def read():
        try:
            reads.append(explainer.explain())
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    def write():
        writer_started.set()
        try:
            explainer.set_background(np.array([[2.0]]))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            writer_done.set()

    reader = threading.Thread(target=read, name="background-reader")
    writer = threading.Thread(target=write, name="background-writer")
    reader.start()
    assert explainer.read_entered.wait(timeout=2.0)
    writer.start()
    assert writer_started.wait(timeout=2.0)
    assert not writer_done.wait(timeout=0.1)
    explainer.release_read.set()
    reader.join(timeout=3.0)
    writer.join(timeout=3.0)

    assert not reader.is_alive() and not writer.is_alive()
    assert errors == []
    np.testing.assert_array_equal(reads, [[[1.0]]])
    np.testing.assert_array_equal(explainer._background_data, [[2.0]])


class _BarrierKeysDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_entered = threading.Event()
        self.release_read = threading.Event()

    def keys(self):
        self.read_entered.set()
        if not self.release_read.wait(timeout=3.0):
            raise RuntimeError("test did not release concept read")
        return super().keys()


def test_tcav_concept_mutation_cannot_interleave_with_read():
    explainer = object.__new__(TCAVExplainer)
    BaseExplainer.__init__(explainer, None)
    concepts = _BarrierKeysDict({"striped": object()})
    explainer._concepts = concepts
    explainer._random_concepts = {}
    explainer._concept_activations = {"striped": np.array([[1.0]])}
    reads = []
    errors = []
    remover_started = threading.Event()
    remover_done = threading.Event()

    def read():
        try:
            reads.append(explainer.list_concepts())
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    def remove():
        remover_started.set()
        try:
            explainer.remove_concept("striped")
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            remover_done.set()

    reader = threading.Thread(target=read, name="concept-reader")
    remover = threading.Thread(target=remove, name="concept-remover")
    reader.start()
    assert concepts.read_entered.wait(timeout=2.0)
    remover.start()
    assert remover_started.wait(timeout=2.0)
    assert not remover_done.wait(timeout=0.1)
    concepts.release_read.set()
    reader.join(timeout=3.0)
    remover.join(timeout=3.0)

    assert not reader.is_alive() and not remover.is_alive()
    assert errors == []
    assert reads == [["striped"]]
    assert explainer._concepts == {}
    assert explainer._concept_activations == {}


class _BFloatLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, inputs):
        return self.identity(inputs).sum(dim=1, keepdim=True)


def test_tensor_and_dlpack_results_preserve_bfloat16_and_own_storage():
    adapter = PyTorchAdapter(_BFloatLayerModel(), task="regression", input_dtype="bfloat16")
    inputs = np.arange(6, dtype=np.float32).reshape(2, 3)

    prediction = adapter.predict(inputs, result_format="tensor")
    scores, input_gradients = adapter.predict_with_gradients(
        inputs, target_class=0, result_format="tensor"
    )
    activations = adapter.get_layer_output(inputs, "identity", result_format="tensor")
    layer_values, layer_gradients = adapter.get_layer_gradients(
        inputs, "identity", target_class=0, result_format="tensor"
    )
    for result in (
        prediction,
        scores,
        input_gradients,
        activations,
        layer_values,
        layer_gradients,
    ):
        assert result.dtype == torch.bfloat16
        assert result.grad_fn is None

    capsule = adapter.predict(inputs, result_format="dlpack")
    round_trip = torch.utils.dlpack.from_dlpack(capsule)
    assert round_trip.dtype == torch.bfloat16
    torch.testing.assert_close(round_trip, prediction)
    with pytest.raises(RuntimeError, match="invalid capsule|consumed only once"):
        torch.utils.dlpack.from_dlpack(capsule)


class _FailingAfterMove(nn.Linear):
    failing_moves = 0

    def to(self, *args, **kwargs):
        if self.failing_moves:
            self.failing_moves -= 1
            raise RuntimeError("custom move and rollback failure")
        return super().to(*args, **kwargs)


def test_unrecoverable_custom_to_poison_is_detected_before_next_prediction():
    model = _FailingAfterMove(2, 1)
    adapter = PyTorchAdapter(model, task="regression")
    model.failing_moves = 2

    with pytest.raises(RuntimeError, match="rollback failed"):
        adapter.to("cpu")
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.list_layers()


def test_meta_move_is_rejected_before_mutation_and_adapter_remains_usable():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0]]))
    adapter = PyTorchAdapter(model, task="regression")
    inputs = np.array([[4.0, 5.0]], dtype=np.float32)
    expected = adapter.predict(inputs)
    weight_before = model.weight.detach().clone()

    with pytest.raises(ValueError, match="do not support the meta device"):
        adapter.to("meta")

    assert adapter.device.type == "cpu"
    assert model.weight.device.type == "cpu"
    torch.testing.assert_close(model.weight, weight_before)
    np.testing.assert_array_equal(adapter.predict(inputs), expected)


def test_meta_model_or_explicit_meta_device_is_rejected_at_construction():
    with pytest.raises(ValueError, match="does not support meta-device models"):
        PyTorchAdapter(nn.Linear(2, 1), task="regression", device="meta")
    with pytest.raises(ValueError, match="does not support meta-device models"):
        PyTorchAdapter(nn.Linear(2, 1, device="meta"), task="regression")


class _CorruptingRollbackTo(nn.Linear):
    fail_next_move = False

    def to(self, *args, **kwargs):
        if self.fail_next_move:
            self.fail_next_move = False
            with torch.no_grad():
                self.weight.add_(100.0)
            raise RuntimeError("failed after corrupting parameter values")
        return super().to(*args, **kwargs)


def test_custom_to_successful_rollback_call_cannot_mask_semantic_corruption():
    model = _CorruptingRollbackTo(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    adapter = PyTorchAdapter(model, task="regression")
    model.fail_next_move = True

    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 102.0))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


def test_instance_shadowed_to_cannot_bypass_custom_move_poisoning():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    adapter = PyTorchAdapter(model, task="regression")
    fail_next_move = True

    def corrupting_to(self, *args, **kwargs):
        nonlocal fail_next_move
        if fail_next_move:
            fail_next_move = False
            with torch.no_grad():
                self.weight.add_(100.0)
            raise RuntimeError("instance-level move corrupted parameter values")
        return nn.Module.to(self, *args, **kwargs)

    model.to = MethodType(corrupting_to, model)
    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 102.0))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _CorruptingApply(nn.Linear):
    fail_next_apply = False

    def _apply(self, fn, recurse=True):
        if self.fail_next_apply:
            self.fail_next_apply = False
            with torch.no_grad():
                self.weight.add_(100.0)
            raise RuntimeError("custom _apply corrupted parameter values")
        # ``recurse`` was added to ``nn.Module._apply`` after the declared
        # Torch 2.0 floor. This fixture is a leaf, so omitting it preserves the
        # same rollback semantics on every supported Torch version.
        return super()._apply(fn)


@pytest.mark.parametrize("nested", [False, True])
def test_custom_apply_anywhere_in_module_tree_cannot_mask_semantic_corruption(nested):
    corrupting_layer = _CorruptingApply(2, 1, bias=False)
    with torch.no_grad():
        corrupting_layer.weight.fill_(2.0)
    model = nn.Sequential(corrupting_layer) if nested else corrupting_layer
    adapter = PyTorchAdapter(model, task="regression")
    corrupting_layer.fail_next_apply = True

    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(
        corrupting_layer.weight,
        torch.full_like(corrupting_layer.weight, 102.0),
    )
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _HidingModules(nn.Sequential):
    def modules(self):
        yield self


def test_overridden_modules_cannot_hide_registered_custom_apply():
    child = _CorruptingApply(2, 1, bias=False)
    with torch.no_grad():
        child.weight.fill_(2.0)
    adapter = PyTorchAdapter(_HidingModules(child), task="regression")
    child.fail_next_apply = True

    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(child.weight, torch.full_like(child.weight, 102.0))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _CorruptingChildren(nn.Sequential):
    fail_next_traversal = False

    def children(self):
        if self.fail_next_traversal:
            self.fail_next_traversal = False
            with torch.no_grad():
                self[0].weight.add_(100.0)
            raise RuntimeError("custom children traversal corrupted state")
        return super().children()


class _CorruptingNamedChildren(nn.Sequential):
    fail_next_traversal = False

    def named_children(self):
        if self.fail_next_traversal:
            self.fail_next_traversal = False
            with torch.no_grad():
                self[0].weight.add_(100.0)
            raise RuntimeError("custom named_children traversal corrupted state")
        return super().named_children()


@pytest.mark.parametrize("model_type", [_CorruptingChildren, _CorruptingNamedChildren])
def test_custom_child_traversal_failure_is_poisoned(model_type):
    child = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        child.weight.fill_(2.0)
    model = model_type(child)
    adapter = PyTorchAdapter(model, task="regression")
    model.fail_next_traversal = True

    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(child.weight, torch.full_like(child.weight, 102.0))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _RepeatedLayerModel(nn.Module):
    def __init__(self, *, dynamic=False):
        super().__init__()
        self.shared = nn.Identity()
        self.history = []
        self.dynamic = dynamic

    def forward(self, inputs):
        first = self.shared(inputs)
        self.history.append(first)
        second = self.shared(first * 2.0)
        self.history.append(second)
        values = [first, second]
        if not self.dynamic or bool(inputs[0, 0] > 0):
            third = self.shared(second * 3.0)
            self.history.append(third)
            values.append(third)
        score = values[0].sum(dim=1) + 5.0 * values[1].sum(dim=1)
        if len(values) == 3:
            score = score + 7.0 * values[2].sum(dim=1)
        return score[:, None]


def test_repeated_layer_first_middle_last_match_analytical_oracles_and_cleanup():
    model = _RepeatedLayerModel()
    adapter = PyTorchAdapter(model, task="regression")
    values = np.array([[2.0, 4.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="executed 3 times.*explicit zero-based occurrence"):
        adapter.get_layer_gradients(values, "shared", target_class=0)

    expected_activations = [values, values * 2.0, values * 6.0]
    expected_gradients = [
        np.full_like(values, 53.0),
        np.full_like(values, 26.0),
        np.full_like(values, 7.0),
    ]
    for occurrence in range(3):
        activations, gradients = adapter.get_layer_gradients(
            values, "shared", target_class=0, occurrence=occurrence
        )
        np.testing.assert_array_equal(activations, expected_activations[occurrence])
        np.testing.assert_array_equal(gradients, expected_gradients[occurrence])
        assert adapter.last_layer_call_count == 3
        assert adapter.last_layer_occurrence == occurrence

    with pytest.raises(ValueError, match="out of range"):
        adapter.get_layer_output(values, "shared", occurrence=3)
    assert len(model.shared._forward_hooks) == 0
    assert all(len(tensor._backward_hooks or {}) == 0 for tensor in model.history)


def test_repeated_layer_dynamic_call_count_fails_and_cleans_hooks():
    model = _RepeatedLayerModel(dynamic=True)
    adapter = PyTorchAdapter(model, task="regression")
    adapter.get_layer_output(np.array([[1.0]], dtype=np.float32), "shared", occurrence=0)

    with pytest.raises(RuntimeError, match="execution count changed from 3 to 2"):
        adapter.get_layer_gradients(
            np.array([[-1.0]], dtype=np.float32), "shared", target_class=0, occurrence=0
        )
    assert len(model.shared._forward_hooks) == 0
    assert all(len(tensor._backward_hooks or {}) == 0 for tensor in model.history)


class _RepeatedSpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Identity()

    def forward(self, inputs):
        first = self.shared(inputs)
        second = self.shared(first * 2.0)
        third = self.shared(second * 3.0)
        score = (
            first.mean(dim=(1, 2, 3))
            + 5.0 * second.mean(dim=(1, 2, 3))
            + 7.0 * third.mean(dim=(1, 2, 3))
        )
        return torch.stack((score, torch.zeros_like(score)), dim=1)


def test_gradcam_occurrence_selector_is_explicit_and_traced_in_metadata():
    adapter = PyTorchAdapter(
        _RepeatedSpatialModel(),
        task="classification",
        output_activation="none",
        classification_output_kind="scores",
    )
    image = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    for occurrence in range(3):
        explanation = GradCAMExplainer(
            adapter,
            "shared",
            input_layout="chw",
            target_occurrence=occurrence,
        ).explain(image, target_class=0)
        np.testing.assert_allclose(
            explanation.explanation_data["heatmap"], [[0.0, 1 / 3], [2 / 3, 1.0]]
        )
        assert explanation.explanation_data["target_occurrence"] == occurrence
        assert explanation.explanation_data["target_layer_call_count"] == 3
    assert len(adapter.model.shared._forward_hooks) == 0


class _DifferentLayerCountSpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.three = nn.Identity()
        self.two = nn.Identity()

    def forward(self, inputs):
        three_first = self.three(inputs)
        three_middle = self.three(three_first * 2.0)
        three_last = self.three(three_middle * 3.0)
        two_first = self.two(inputs * 4.0)
        two_last = self.two(two_first * 2.0)
        score = (
            three_first.mean(dim=(1, 2, 3))
            + three_middle.mean(dim=(1, 2, 3))
            + three_last.mean(dim=(1, 2, 3))
            + two_first.mean(dim=(1, 2, 3))
            + two_last.mean(dim=(1, 2, 3))
        )
        return torch.stack((score, torch.zeros_like(score)), dim=1)


@pytest.mark.parametrize(
    ("explainer_type", "normalization_module", "class_agnostic"),
    [
        (GradCAMExplainer, gradcam_module, False),
        (HiResCAMExplainer, cam_variants_module, False),
        (EigenCAMExplainer, cam_variants_module, True),
    ],
)
def test_cam_results_keep_atomic_per_call_trace_during_shared_adapter_race(
    monkeypatch, explainer_type, normalization_module, class_agnostic
):
    adapter = PyTorchAdapter(
        _DifferentLayerCountSpatialModel(),
        task="classification",
        output_activation="none",
        classification_output_kind="scores",
    )
    image = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    first = explainer_type(adapter, "three", input_layout="chw", target_occurrence=2)
    second = explainer_type(adapter, "two", input_layout="chw", target_occurrence=1)
    normalization_entered = threading.Event()
    release_normalization = threading.Event()
    original_normalization = normalization_module._cam_normalization_metadata

    def barrier_normalization(cam):
        if threading.current_thread().name == "cam-three":
            normalization_entered.set()
            if not release_normalization.wait(timeout=3.0):
                raise RuntimeError("test did not release CAM normalization")
        return original_normalization(cam)

    monkeypatch.setattr(
        normalization_module,
        "_cam_normalization_metadata",
        barrier_normalization,
    )
    results = {}
    errors = []

    def explain(name, explainer):
        try:
            target = None if class_agnostic else 0
            results[name] = explainer.explain(image, target_class=target)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first_thread = threading.Thread(target=explain, args=("three", first), name="cam-three")
    second_thread = threading.Thread(target=explain, args=("two", second), name="cam-two")
    first_thread.start()
    assert normalization_entered.wait(timeout=2.0)
    second_thread.start()
    second_thread.join(timeout=3.0)
    assert not second_thread.is_alive()
    assert adapter.last_layer_call_count == 2
    release_normalization.set()
    first_thread.join(timeout=3.0)

    assert not first_thread.is_alive()
    assert errors == []
    assert results["three"].explanation_data["target_occurrence"] == 2
    assert results["three"].explanation_data["target_layer_call_count"] == 3
    assert results["two"].explanation_data["target_occurrence"] == 1
    assert results["two"].explanation_data["target_layer_call_count"] == 2


def test_tcav_occurrence_selector_matches_first_middle_last_directional_oracles():
    adapter = PyTorchAdapter(
        _RepeatedSpatialModel(),
        task="classification",
        output_activation="none",
        classification_output_kind="scores",
    )
    values = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    expected_activations = [values, values * 2.0, values * 6.0]
    expected_derivatives = [26.5, 13.0, 3.5]

    with pytest.raises(RuntimeError, match="executed 3 times.*explicit zero-based occurrence"):
        TCAVExplainer(adapter, "shared")._get_activations(values)

    for occurrence in range(3):
        explainer = TCAVExplainer(adapter, "shared", layer_occurrence=occurrence)
        np.testing.assert_array_equal(
            explainer._get_activations(values),
            expected_activations[occurrence].reshape(1, -1),
        )
        cav = ConceptActivationVector(
            concept_name="unit",
            layer_name="shared",
            vector=np.ones(4, dtype=np.float64),
            classifier=object(),
            accuracy=1.0,
            metadata={"layer_occurrence": occurrence},
        )
        derivatives = explainer.compute_directional_derivative(values, cav, target_class=0)
        np.testing.assert_allclose(derivatives, [expected_derivatives[occurrence]])

    wrong_occurrence_cav = ConceptActivationVector(
        concept_name="wrong",
        layer_name="shared",
        vector=np.ones(4, dtype=np.float64),
        classifier=object(),
        accuracy=1.0,
        metadata={"layer_occurrence": 0},
    )
    with pytest.raises(ValueError, match="layer occurrence 0 does not match 1"):
        TCAVExplainer(adapter, "shared", layer_occurrence=1).compute_directional_derivative(
            values, wrong_occurrence_cav, target_class=0
        )
    assert len(adapter.model.shared._forward_hooks) == 0


class _FiveChannelSpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(5, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
        self.seen_shapes = []

    def forward(self, inputs):
        self.seen_shapes.append(tuple(inputs.shape))
        score = self.conv(inputs).mean(dim=(1, 2, 3))
        return score[:, None]


def test_explicit_layouts_cover_hwc_hw_nhw_and_custom_channels():
    hwc_model = _FiveChannelSpatialModel()
    hwc_adapter = PyTorchAdapter(hwc_model, task="regression")
    hwc = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    ig = IntegratedGradientsExplainer(hwc_adapter, n_steps=1, input_layout="hwc")
    explanation = ig.explain(hwc, target_class=0)
    np.testing.assert_allclose(explanation.explanation_data["attributions_raw"], hwc / 12.0)
    assert explanation.metadata["input_layout"] == "hwc"
    assert explanation.metadata["channel_axis"] == -1
    assert hwc_model.seen_shapes and set(hwc_model.seen_shapes) == {(1, 5, 3, 4)}

    cam = GradCAMExplainer(hwc_adapter, "conv", input_layout="hwc").explain(hwc, target_class=0)
    assert cam.explanation_data["input_layout"] == "hwc"
    assert cam.explanation_data["channel_axis"] == -1

    class GraySum(nn.Module):
        def forward(self, inputs):
            return inputs.flatten(1).sum(dim=1, keepdim=True)

    gray = IntegratedGradientsExplainer(
        PyTorchAdapter(GraySum(), task="regression"), n_steps=1, input_layout="hw"
    )
    batch = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    results = gray.explain_batch(batch, target_class=0)
    assert len(results) == 3
    assert all(result.explanation_data["input_layout"] == "hw" for result in results)
    np.testing.assert_allclose(results[2].explanation_data["attributions_raw"], batch[2])


def test_layout_rank_mismatch_fails_before_model_work():
    model = _FiveChannelSpatialModel()
    adapter = PyTorchAdapter(model, task="regression")
    values = np.zeros((5, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="input_layout='hw'.*rank-2"):
        IntegratedGradientsExplainer(adapter, n_steps=1, input_layout="hw").explain(
            values, target_class=0
        )
    with pytest.raises(ValueError, match="input_layout='hw'.*rank-2"):
        GradCAMExplainer(adapter, "conv", input_layout="hw").explain(values, target_class=0)
    assert model.seen_shapes == []


class _ConstantMatrix(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values.expand(inputs.shape[0], -1)


def test_declared_scores_probabilities_and_undeclared_matrix_take_distinct_paths():
    values = [[0.2, 0.8]]
    scores = PyTorchAdapter(
        _ConstantMatrix(values),
        task="classification",
        output_activation="none",
        classification_output_kind="scores",
    )
    probabilities = PyTorchAdapter(
        _ConstantMatrix(values),
        task="classification",
        output_activation="none",
        classification_output_kind="probabilities",
    )
    undeclared = PyTorchAdapter(
        _ConstantMatrix(values), task="classification", output_activation="none"
    )
    inputs = np.zeros((1, 1), dtype=np.float32)

    np.testing.assert_array_equal(scores.predict(inputs), probabilities.predict(inputs))
    assert scores.prediction_output_kind == "scores"
    assert probabilities.prediction_output_kind == "probabilities"
    with pytest.raises(ValueError, match="requires probabilities, not arbitrary class scores"):
        normalize_classifier_outputs(
            scores,
            inputs,
            context="probability consumer",
            require_probabilities=True,
        )
    np.testing.assert_allclose(
        normalize_classifier_outputs(
            probabilities,
            inputs,
            context="probability consumer",
            require_probabilities=True,
        ),
        values,
        rtol=0.0,
        atol=2e-8,
    )
    with pytest.raises(ValueError, match="cannot infer.*undeclared multiclass matrix"):
        normalize_classifier_outputs(
            undeclared,
            inputs,
            context="probability consumer",
            require_probabilities=True,
        )


@pytest.mark.parametrize("values", [[[1.2, -0.2]], [[0.2, 0.7]]])
def test_declared_probability_validation_rejects_range_and_simplex_violations(values):
    adapter = PyTorchAdapter(
        _ConstantMatrix(values),
        task="classification",
        output_activation="none",
        classification_output_kind="probabilities",
    )
    with pytest.raises(ValueError, match="probabilities must lie|probabilities must sum"):
        adapter.predict(np.zeros((1, 1), dtype=np.float32))


class _SneakyRegistry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_calls = 0

    def values(self):
        self.values_calls += 1
        return super().values()


def test_custom_module_registry_is_rejected_and_poisoned_without_iteration():
    model = nn.Sequential(nn.Linear(2, 1, bias=False))
    adapter = PyTorchAdapter(model, task="regression")
    state = object.__getattribute__(model, "__dict__")
    sneaky = _SneakyRegistry(state["_modules"])
    state["_modules"] = sneaky

    with pytest.raises(RuntimeError, match="registered-state graph is not trustworthy"):
        adapter.to("cpu")

    assert sneaky.values_calls == 0
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _BufferMutationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.tensor([1.0]))
        self.tensor_forward_calls = 0

    def forward(self, inputs):
        self.tensor_forward_calls += 1
        self.cache.add_(1.0)
        return inputs.sum(dim=1, keepdim=True)


def test_custom_buffer_registry_fails_before_explanation_model_work():
    model = _BufferMutationProbe()
    adapter = PyTorchAdapter(model, task="regression")
    state = object.__getattribute__(model, "__dict__")
    sneaky = _SneakyRegistry(state["_buffers"])
    state["_buffers"] = sneaky

    with pytest.raises(RuntimeError, match="exact built-in dict"):
        SaliencyExplainer(adapter, ["x0", "x1"]).explain(
            np.ones(2, dtype=np.float32), target_class=0
        )

    assert sneaky.values_calls == 0
    assert model.tensor_forward_calls == 0
    torch.testing.assert_close(model.cache, torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _FakeDictModule(nn.Module):
    def __init__(self):
        object.__setattr__(self, "hide_state", False)
        super().__init__()
        self.register_buffer("cache", torch.tensor([1.0]))
        self.tensor_forward_calls = 0

    def __getattribute__(self, name):
        if name == "__dict__" and object.__getattribute__(self, "hide_state"):
            return {"_modules": {}, "_parameters": {}, "_buffers": {}}
        return super().__getattribute__(name)

    def forward(self, inputs):
        self.tensor_forward_calls += 1
        object.__getattribute__(self, "__dict__")["_buffers"]["cache"].add_(1.0)
        return inputs.sum(dim=1, keepdim=True)


def test_custom_getattribute_cannot_hide_real_registered_state():
    model = _FakeDictModule()
    model.hide_state = True
    real_state = object.__getattribute__(model, "__dict__")

    with pytest.raises(RuntimeError, match="__getattribute__.*unsupported"):
        PyTorchAdapter(model, task="regression")

    assert model.tensor_forward_calls == 0
    torch.testing.assert_close(real_state["_buffers"]["cache"], torch.tensor([1.0]))


class _MoveCallableSpoof:
    __func__ = nn.Module.to

    def __init__(self, module):
        self.module = module
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            with torch.no_grad():
                self.module.weight.add_(10.0)
            raise RuntimeError("spoofed move corrupted weight")
        return self.module


def test_callable_func_spoof_cannot_evade_custom_move_poisoning():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    adapter = PyTorchAdapter(model, task="regression")
    model.to = _MoveCallableSpoof(model)

    with pytest.raises(RuntimeError, match="exact model restoration cannot be proven"):
        adapter.to("cpu")

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 12.0))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _CorruptingTrain(nn.Linear):
    armed = False

    def __init__(self):
        super().__init__(2, 1, bias=False)
        self.tensor_forward_calls = 0

    def train(self, mode=True):
        if self.armed:
            with torch.no_grad():
                self.weight.add_(10.0)
        return super().train(mode)

    def forward(self, inputs):
        self.tensor_forward_calls += 1
        return super().forward(inputs)


def test_adapter_construction_does_not_invoke_custom_train_dispatch():
    model = _CorruptingTrain()
    with torch.no_grad():
        model.weight.fill_(2.0)
    model.armed = True

    adapter = PyTorchAdapter(model, task="regression")

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))
    assert model.training is False
    assert adapter.predict(np.ones((1, 2), dtype=np.float32)).item() == 4.0


@pytest.mark.parametrize("nested", [False, True])
def test_custom_train_dispatch_is_rejected_before_explanation_and_poisoned(nested):
    layer = _CorruptingTrain()
    with torch.no_grad():
        layer.weight.fill_(2.0)
    model = nn.Sequential(layer) if nested else layer
    adapter = PyTorchAdapter(model, task="regression")
    layer.armed = True

    with pytest.raises(RuntimeError, match="train.*overrides canonical"):
        SaliencyExplainer(adapter, ["x0", "x1"]).explain(
            np.ones(2, dtype=np.float32), target_class=0
        )

    torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 2.0))
    assert layer.tensor_forward_calls == 0
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))


class _FailingRestoreProtocol:
    def snapshot(self, module):
        return module.weight.detach().clone()

    def restore(self, module, snapshot):
        with torch.no_grad():
            module.weight.add_(10.0)
        raise RuntimeError("restore failed after mutation")


def test_state_restoration_failure_poisons_every_later_adapter_operation():
    model = nn.Linear(2, 1, bias=False)
    adapter = PyTorchAdapter(
        model,
        task="regression",
        model_state_protocol=_FailingRestoreProtocol(),
    )

    with pytest.raises(ModelStateIsolationError, match="restoration failed"):
        SaliencyExplainer(adapter, ["x0", "x1"]).explain(
            np.ones(2, dtype=np.float32), target_class=0
        )

    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.predict(np.ones((1, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.to("cpu")
    with pytest.raises(RuntimeError, match="is poisoned.*Reconstruct"):
        adapter.list_layers()


def _owned_test_cav(name="striped"):
    classifier = {"coef": [1.0]}
    metadata = {"nested": {"value": 1}, "layer_occurrence": None}
    vector = np.array([3.0, 4.0])
    cav = ConceptActivationVector(name, "layer", vector, classifier, 0.75, metadata)
    return cav, vector, classifier, metadata


def test_cav_and_tcav_public_snapshots_have_no_mutable_aliases():
    cav, vector, classifier, metadata = _owned_test_cav()
    vector[:] = 99.0
    classifier["coef"][0] = 99.0
    metadata["nested"]["value"] = 99
    np.testing.assert_allclose(cav.vector, [0.6, 0.8])
    assert cav.classifier["coef"] == [1.0]
    assert cav.metadata["nested"]["value"] == 1
    with pytest.raises(AttributeError, match="immutable"):
        cav.accuracy = 0.0

    public_vector = cav.vector
    with pytest.raises(ValueError):
        public_vector[0] = 0.0
    public_vector.setflags(write=True)
    public_vector[:] = 0.0
    public_classifier = cav.classifier
    public_classifier["coef"][0] = -1.0
    public_metadata = cav.metadata
    public_metadata["nested"]["value"] = -1
    np.testing.assert_allclose(cav.vector, [0.6, 0.8])
    assert cav.classifier["coef"] == [1.0]
    assert cav.metadata["nested"]["value"] == 1

    rounding_sensitive = ConceptActivationVector(
        "rounding-sensitive",
        "layer",
        np.random.default_rng(1).normal(size=7),
        {"coef": [1.0]},
        0.75,
    )
    bit_exact_snapshot = rounding_sensitive.clone()
    np.testing.assert_array_equal(bit_exact_snapshot.vector, rounding_sensitive.vector)
    assert not np.shares_memory(bit_exact_snapshot.vector, rounding_sensitive.vector)

    explainer = object.__new__(TCAVExplainer)
    BaseExplainer.__init__(explainer, None)
    explainer._concepts = {"striped": cav}
    explainer._random_concepts = {"random": (cav,)}
    explainer._concept_activations = {}
    concept_snapshot = explainer.concepts
    random_snapshot = explainer.random_concepts
    with pytest.raises(TypeError):
        concept_snapshot["new"] = cav
    with pytest.raises(TypeError):
        random_snapshot["new"] = (cav,)
    assert isinstance(random_snapshot["random"], tuple)
    changed = concept_snapshot["striped"].vector
    changed.setflags(write=True)
    changed[:] = 0.0
    np.testing.assert_allclose(explainer.get_concept("striped").vector, [0.6, 0.8])


def test_tcav_snapshot_waits_for_atomic_private_store_update():
    first, *_ = _owned_test_cav("first")
    second, *_ = _owned_test_cav("second")
    explainer = object.__new__(TCAVExplainer)
    BaseExplainer.__init__(explainer, None)
    explainer._concepts = {"first": first}
    explainer._random_concepts = {}
    explainer._concept_activations = {}
    started = threading.Event()
    finished = threading.Event()
    snapshots = []

    def read_snapshot():
        started.set()
        snapshots.append(explainer.concepts)
        finished.set()

    with explainer._instance_lock:
        reader = threading.Thread(target=read_snapshot, name="tcav-snapshot-reader")
        reader.start()
        assert started.wait(timeout=2.0)
        assert not finished.wait(timeout=0.1)
        explainer._concepts["second"] = second
    reader.join(timeout=3.0)

    assert not reader.is_alive()
    assert list(snapshots[0]) == ["first", "second"]
