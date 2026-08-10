"""Verified DeepLIFT Rescale and DeepLiftShap explainers.

This module delegates the modified backward rules to Captum.  It intentionally
does not provide a gradient-at-midpoint or low-step Integrated Gradients
fallback: those computations are not DeepLIFT.

The currently verified input contract is a single, flat feature tensor handled
by :class:`~explainiverse.adapters.PyTorchAdapter`.  Models must expose their
nonlinearities as supported ``torch.nn`` modules so Captum can attach the hooks
required by the Rescale rule.  Unsupported or untraceable graphs fail clearly.

Reference:
    Shrikumar, Greenside, and Kundaje (2017), "Learning Important Features
    Through Propagating Activation Differences", ICML 2017.
    https://arxiv.org/abs/1704.02685
"""

from __future__ import annotations

import operator
from numbers import Integral
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np

from explainiverse.core.explainer import BaseExplainer
from explainiverse.core.explanation import Explanation
from explainiverse.explainers._validation import as_real_array, validate_name_sequence
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        import torch.nn as nn

        TORCH_AVAILABLE = True
    except ImportError:  # pragma: no cover - exercised in installations without torch
        torch = None
        nn = None
        TORCH_AVAILABLE = False

CAPTUM_VERSION: Optional[str]
try:
    import captum
    from captum.attr import DeepLift as CaptumDeepLift
    from captum.attr import DeepLiftShap as CaptumDeepLiftShap
    from captum.attr import IntegratedGradients as CaptumIntegratedGradients

    CAPTUM_AVAILABLE = True
    CAPTUM_VERSION = captum.__version__
except ImportError:  # pragma: no cover - exercised in installations without captum
    CaptumDeepLift = None
    CaptumDeepLiftShap = None
    CaptumIntegratedGradients = None
    CAPTUM_AVAILABLE = False
    CAPTUM_VERSION = None


def _require_backends() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("DeepLIFT requires PyTorch. Install the torch optional dependency.")
    if not CAPTUM_AVAILABLE:
        raise ImportError(
            "DeepLIFT and DeepSHAP require Captum's canonical modified-backward "
            "implementation. Install it with `pip install captum`; no gradient "
            "approximation fallback is used."
        )


_SUPPORTED_NONLINEAR_TYPES: Tuple[type, ...]
_SUPPORTED_LINEAR_TYPES: Tuple[type, ...]
if TORCH_AVAILABLE:
    _SUPPORTED_NONLINEAR_TYPES = (
        nn.ReLU,
        nn.ELU,
        nn.LeakyReLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Softplus,
        nn.Softmax,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
    )
    _SUPPORTED_LINEAR_TYPES = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.Identity,
        nn.Flatten,
        nn.Unflatten,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.ConstantPad1d,
        nn.ConstantPad2d,
        nn.ConstantPad3d,
    )
else:  # pragma: no cover
    _SUPPORTED_NONLINEAR_TYPES = ()
    _SUPPORTED_LINEAR_TYPES = ()


def _validate_supported_model(model: "nn.Module") -> None:
    """Reject graphs for which Captum would silently use ordinary gradients.

    Captum implements DeepLIFT by attaching hooks to supported nonlinear
    modules.  Functional activations and unsupported module types would evade
    those hooks, yielding gradient-times-input while still being labelled
    DeepLIFT.  FX tracing lets us reject that case before attribution.
    """

    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as exc:
        raise NotImplementedError(
            "DeepLIFT currently requires a torch.fx-traceable, feed-forward "
            "module graph. Dynamic control flow and untraceable models are not "
            "silently approximated."
        ) from exc

    allowed_modules = _SUPPORTED_LINEAR_TYPES + _SUPPORTED_NONLINEAR_TYPES
    nonlinear_call_counts: dict[str, int] = {}

    for node in traced.graph.nodes:
        if node.op == "call_module":
            module = traced.get_submodule(str(node.target))
            if not isinstance(module, allowed_modules):
                raise NotImplementedError(
                    "DeepLIFT does not have a verified Rescale rule for module "
                    f"{node.target!r} ({type(module).__name__}). Use only the "
                    "documented supported modules or choose another explainer."
                )
            if isinstance(module, _SUPPORTED_NONLINEAR_TYPES):
                module_name = str(node.target)
                nonlinear_call_counts[module_name] = nonlinear_call_counts.get(module_name, 0) + 1
        elif node.op == "call_method":
            if node.target not in {
                "view",
                "reshape",
                "flatten",
                "squeeze",
                "unsqueeze",
                "contiguous",
            }:
                raise NotImplementedError(
                    "DeepLIFT requires nonlinear operations to be explicit "
                    f"supported nn.Modules; method call {node.target!r} is not "
                    "in the verified graph subset."
                )
        elif node.op == "call_function":
            # Indexing is a linear routing operation. Arithmetic between graph
            # values can introduce products / interactions for which this
            # wrapper has no independently verified propagation contract.
            if node.target is not operator.getitem:
                name = getattr(node.target, "__name__", repr(node.target))
                raise NotImplementedError(
                    "DeepLIFT requires nonlinear operations to be explicit "
                    f"supported nn.Modules; function call {name!r} is not in "
                    "the verified graph subset."
                )

    reused = [name for name, count in nonlinear_call_counts.items() if count > 1]
    if reused:
        raise NotImplementedError(
            "Captum DeepLIFT cannot safely reuse one nonlinear module more than "
            f"once in a forward graph. Reused modules: {reused}. Instantiate a "
            "separate activation module at each use site."
        )


if TYPE_CHECKING:

    class _TorchModuleBase(nn.Module):
        """Static base for the optional runtime torch module."""

else:
    _TorchModuleBase = nn.Module if TORCH_AVAILABLE else object


class _AdapterScoreModel(_TorchModuleBase):
    """Torch module mirroring the adapter's differentiable score contract.

    Activations are module instances rather than functional calls so Captum can
    apply its Rescale rule through sigmoid and softmax output transformations.
    """

    def __init__(self, adapter):
        super().__init__()
        self.wrapped_model = adapter.model
        self.task = adapter.task
        self.output_activation = adapter.output_activation
        self.gradient_output = adapter.gradient_output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _as_matrix(output: "torch.Tensor") -> "torch.Tensor":
        return output.unsqueeze(-1) if output.dim() == 1 else output

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        raw = self._as_matrix(self.wrapped_model(inputs))

        # A one-output classifier has no independent raw class-0 score.  Match
        # PyTorchAdapter by explaining complementary probabilities.
        if self.task == "classification" and raw.shape[1] == 1:
            if self.output_activation in {"softmax", "sigmoid"}:
                positive = self.sigmoid(raw)
            else:
                positive = raw
            return torch.cat((1.0 - positive, positive), dim=1)

        if self.gradient_output == "prediction":
            if self.output_activation == "softmax":
                return self.softmax(raw)
            if self.output_activation == "sigmoid":
                return self.sigmoid(raw)

        return raw


class DeepLIFTExplainer(BaseExplainer):
    """DeepLIFT Rescale for verified PyTorchAdapter model graphs.

    ``multiply_by_inputs=True`` returns contributions and supports the
    summation-to-delta check.  ``False`` returns DeepLIFT multipliers.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        baseline: Optional[Union[np.ndarray, str, Callable]] = None,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        random_state: Optional[int] = None,
    ):
        _require_backends()

        from explainiverse.adapters.pytorch_adapter import PyTorchAdapter

        if not isinstance(model, PyTorchAdapter):
            raise TypeError(
                "DeepLIFT requires a PyTorchAdapter so the explained score "
                "space and class mapping are explicit."
            )
        validated_features = validate_name_sequence(feature_names, name="feature_names")
        validated_classes = validate_name_sequence(
            class_names,
            name="class_names",
            allow_none=True,
        )
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, Integral):
                raise TypeError("random_state must be an integer or None")
            if int(random_state) < 0:
                raise ValueError("random_state must be non-negative")

        super().__init__(model)
        _validate_supported_model(model.model)

        assert validated_features is not None
        self.feature_names: List[str] = validated_features
        self.class_names: Optional[List[str]] = validated_classes
        self.baseline: Optional[Union[np.ndarray, str, Callable]] = baseline
        self.multiply_by_inputs: bool = bool(multiply_by_inputs)
        self.eps: float = float(eps)
        self.random_state: Optional[int] = int(random_state) if random_state is not None else None
        # ``_AdapterScoreModel`` registers the caller's model as a child module.
        # Calling ``.to()`` or ``.eval()`` on the wrapper would therefore mutate
        # that shared model during explainer construction.  The adapter already
        # owns the model's device placement; attribution enters a temporary eval
        # context instead and restores every caller-visible state afterwards.
        self._score_model: _AdapterScoreModel = _AdapterScoreModel(model)

    def _as_feature_vector(self, value, name: str) -> np.ndarray:
        raw = as_real_array(value, name=name)
        if raw.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional flat feature vector; " f"got shape {raw.shape}"
            )
        if raw.size != len(self.feature_names):
            raise ValueError(
                f"{name} contains {raw.size} values, but "
                f"{len(self.feature_names)} feature_names were provided. "
                "DeepLIFT currently supports flat feature vectors only."
            )
        try:
            array = as_real_array(
                raw,
                name=name,
                dtype=np.float32,
            ).copy()
        except ValueError as error:
            raise TypeError(f"{name} must contain real numeric values") from error
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _new_rng(self) -> np.random.Generator:
        """Create a per-operation generator without touching NumPy global state."""
        return np.random.default_rng(self.random_state)

    def _get_baseline(self, instance: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            baseline = np.zeros_like(instance)
        elif isinstance(self.baseline, str):
            if self.baseline == "random":
                baseline = (
                    self._new_rng()
                    .uniform(
                        low=float(instance.min()),
                        high=float(instance.max()),
                        size=instance.shape,
                    )
                    .astype(np.float32)
                )
            elif self.baseline == "mean":
                raise ValueError(
                    "Baseline 'mean' requires calling set_baseline(data, 'mean') first."
                )
            else:
                raise ValueError(f"Unknown baseline type: {self.baseline!r}")
        elif callable(self.baseline):
            baseline = self.baseline(instance.copy())
        else:
            baseline = self.baseline
        return self._as_feature_vector(baseline, "baseline")

    def set_baseline(self, data: np.ndarray, method: str = "mean") -> "DeepLIFTExplainer":
        data = as_real_array(
            data,
            name="data",
            dtype=np.float32,
            require_finite=True,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError("data must have shape (n_samples, len(feature_names))")

        if method == "mean":
            self.baseline = np.mean(data, axis=0).astype(np.float32)
        elif method == "median":
            self.baseline = np.median(data, axis=0).astype(np.float32)
        elif method == "zeros":
            self.baseline = None
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return self

    def _prepare(
        self,
        instance: np.ndarray,
        baselines: np.ndarray,
        target_class: Optional[int],
    ) -> Tuple["torch.Tensor", "torch.Tensor", int, str]:
        instance_tensor = torch.as_tensor(
            instance.reshape(1, -1), dtype=torch.float32, device=self.model.device
        )
        instance_tensor.requires_grad_(True)
        baseline_tensor = torch.as_tensor(
            baselines.reshape(-1, len(self.feature_names)),
            dtype=torch.float32,
            device=self.model.device,
        )

        with preserve_adapter_model_eval(self.model), torch.no_grad():
            raw = self.model.model(instance_tensor)
            adapter_scores, output_space = self.model._gradient_score_output(raw)
            wrapper_scores = self._score_model(instance_tensor)

        if adapter_scores.shape != wrapper_scores.shape or not torch.allclose(
            adapter_scores, wrapper_scores, atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError(
                "DeepLIFT's score wrapper no longer matches PyTorchAdapter's "
                "gradient score contract; attribution was aborted."
            )

        n_outputs = int(wrapper_scores.shape[1])
        if target_class is None:
            if self.model.task == "classification":
                target = int(wrapper_scores[0].argmax().item())
            elif n_outputs == 1:
                target = 0
            else:
                raise ValueError("target_class is required for multi-output regression")
        elif isinstance(target_class, Integral) and not isinstance(target_class, bool):
            target = int(target_class)
        else:
            raise TypeError("target_class must be an integer index or None")

        if target < 0 or target >= n_outputs:
            raise ValueError(f"target_class must be in [0, {n_outputs - 1}], got {target}")
        if self.class_names is not None and len(self.class_names) != n_outputs:
            raise ValueError(
                f"class_names has length {len(self.class_names)}, but the "
                f"explained score has {n_outputs} outputs"
            )

        return instance_tensor, baseline_tensor, target, output_space

    @staticmethod
    def _validate_method(method: str) -> None:
        if method != "rescale":
            if method == "rescale_exact":
                raise ValueError(
                    "method='rescale_exact' was removed because it was a "
                    "low-step Integrated Gradients approximation, not DeepLIFT. "
                    "Use method='rescale'."
                )
            raise ValueError("DeepLIFT supports only method='rescale'")

    def _attribute_single(
        self,
        instance_tensor: "torch.Tensor",
        baseline_tensor: "torch.Tensor",
        target: int,
        return_delta: bool = False,
    ):
        with preserve_adapter_model_eval(self.model):
            backend = CaptumDeepLift(
                self._score_model,
                multiply_by_inputs=self.multiply_by_inputs,
                eps=self.eps,
            )
            return backend.attribute(
                instance_tensor,
                baselines=baseline_tensor,
                target=target,
                return_convergence_delta=return_delta,
            )

    def _compute_deeplift_rescale(
        self,
        instance: np.ndarray,
        baseline: np.ndarray,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        instance = self._as_feature_vector(instance, "instance")
        baseline = self._as_feature_vector(baseline, "baseline")
        input_tensor, baseline_tensor, target, _ = self._prepare(
            instance, baseline.reshape(1, -1), target_class
        )
        values = self._attribute_single(input_tensor, baseline_tensor, target)
        return values.detach().cpu().numpy().reshape(-1)

    def _label_name(self, target: int) -> str:
        if self.class_names is not None:
            return self.class_names[target]
        if self.model.task == "classification":
            return f"class_{target}"
        return "output" if target == 0 else f"output_{target}"

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        method: str = "rescale",
        return_convergence_delta: bool = False,
    ) -> Explanation:
        self._validate_method(method)
        if return_convergence_delta and not self.multiply_by_inputs:
            raise ValueError(
                "Convergence delta is defined for contributions only; initialize "
                "with multiply_by_inputs=True."
            )

        instance_array = self._as_feature_vector(instance, "instance")
        baseline_array = (
            self._as_feature_vector(baseline, "baseline")
            if baseline is not None
            else self._get_baseline(instance_array)
        )
        input_tensor, baseline_tensor, target, output_space = self._prepare(
            instance_array, baseline_array.reshape(1, -1), target_class
        )

        result = self._attribute_single(
            input_tensor, baseline_tensor, target, return_convergence_delta
        )
        if return_convergence_delta:
            attribution_tensor, captum_delta = result
        else:
            attribution_tensor = result
            captum_delta = None
        values = attribution_tensor.detach().cpu().numpy().reshape(-1)

        explanation_data = {
            "feature_attributions": {
                name: float(values[i]) for i, name in enumerate(self.feature_names)
            },
            "attributions_raw": values.tolist(),
            "baseline": baseline_array.tolist(),
            "method": "rescale",
            "backend": "captum.DeepLift",
            "backend_version": CAPTUM_VERSION,
            "output_space": output_space,
            "target_index": target,
            "multiply_by_inputs": self.multiply_by_inputs,
        }

        if return_convergence_delta:
            with preserve_adapter_model_eval(self.model), torch.no_grad():
                actual = float(self._score_model(input_tensor)[0, target].item())
                reference = float(self._score_model(baseline_tensor)[0, target].item())
            prediction_difference = actual - reference
            attribution_sum = float(values.sum())
            explanation_data.update(
                {
                    "convergence_delta": abs(prediction_difference - attribution_sum),
                    "captum_convergence_delta": float(
                        captum_delta.detach().abs().max().cpu().item()
                    ),
                    "prediction_difference": prediction_difference,
                    "attribution_sum": attribution_sum,
                }
            )

        return Explanation(
            explainer_name="DeepLIFT",
            target_class=self._label_name(target),
            explanation_data=explanation_data,
            feature_names=self.feature_names,
        )

    def explain_batch(
        self,
        X: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "rescale",
    ) -> List[Explanation]:
        self._validate_method(method)
        X = as_real_array(X, name="X", dtype=np.float32, require_finite=True)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must have shape (n_samples, len(feature_names))")
        return [self.explain(row, target_class=target_class, method=method) for row in X]

    def _attribute_multiple(
        self,
        instance: np.ndarray,
        baselines: np.ndarray,
        target_class: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, int, str]:
        input_tensor, baseline_tensor, target, output_space = self._prepare(
            instance, baselines, target_class
        )

        individual = []
        for baseline_row in baseline_tensor:
            attribution = self._attribute_single(input_tensor, baseline_row.unsqueeze(0), target)
            individual.append(attribution.detach().cpu().numpy().reshape(-1))
        individual_values = np.vstack(individual)

        if len(baseline_tensor) == 1:
            averaged = individual_values[0]
        else:
            if self.eps != 1e-10:
                raise ValueError(
                    "Captum DeepLiftShap does not expose a configurable epsilon; "
                    "multiple-baseline attribution requires eps=1e-10."
                )
            with preserve_adapter_model_eval(self.model):
                backend = CaptumDeepLiftShap(
                    self._score_model,
                    multiply_by_inputs=self.multiply_by_inputs,
                )
                averaged_tensor = backend.attribute(
                    input_tensor, baselines=baseline_tensor, target=target
                )
            averaged = averaged_tensor.detach().cpu().numpy().reshape(-1)

            # DeepLiftShap is defined as the expectation of DeepLIFT over the
            # baseline distribution. Abort if a backend change violates it.
            if not np.allclose(averaged, individual_values.mean(axis=0), atol=1e-6, rtol=1e-5):
                raise RuntimeError(
                    "Captum DeepLiftShap disagreed with the mean of its DeepLIFT "
                    "baseline contributions; attribution was aborted."
                )

        return averaged, individual_values, target, output_space

    def explain_with_multiple_baselines(
        self,
        instance: np.ndarray,
        baselines: np.ndarray,
        target_class: Optional[int] = None,
        method: str = "rescale",
    ) -> Explanation:
        self._validate_method(method)
        instance_array = self._as_feature_vector(instance, "instance")
        baseline_array = as_real_array(
            baselines,
            name="baselines",
            dtype=np.float32,
            require_finite=True,
        )
        if baseline_array.ndim == 1:
            baseline_array = baseline_array.reshape(1, -1)
        if baseline_array.ndim != 2 or baseline_array.shape[1] != len(self.feature_names):
            raise ValueError("baselines must have shape (n_baselines, len(feature_names))")
        if len(baseline_array) == 0:
            raise ValueError("baselines must contain at least one finite row")

        averaged, individual, target, output_space = self._attribute_multiple(
            instance_array, baseline_array, target_class
        )
        return Explanation(
            explainer_name="DeepLIFT_MultiBaseline",
            target_class=self._label_name(target),
            feature_names=self.feature_names,
            explanation_data={
                "feature_attributions": {
                    name: float(averaged[i]) for i, name in enumerate(self.feature_names)
                },
                "attributions_raw": averaged.tolist(),
                "attributions_std": individual.std(axis=0).tolist(),
                "n_baselines": len(baseline_array),
                "method": "rescale",
                "backend": (
                    "captum.DeepLiftShap" if len(baseline_array) > 1 else "captum.DeepLift"
                ),
                "backend_version": CAPTUM_VERSION,
                "output_space": output_space,
                "target_index": target,
            },
        )

    def compare_with_integrated_gradients(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        ig_steps: int = 50,
    ) -> dict:
        """Compare against Captum IG in the identical score / target space."""
        if ig_steps < 2:
            raise ValueError("ig_steps must be at least 2")
        instance_array = self._as_feature_vector(instance, "instance")
        baseline_array = (
            self._as_feature_vector(baseline, "baseline")
            if baseline is not None
            else self._get_baseline(instance_array)
        )
        input_tensor, baseline_tensor, target, output_space = self._prepare(
            instance_array, baseline_array.reshape(1, -1), target_class
        )
        dl_tensor = self._attribute_single(input_tensor, baseline_tensor, target)
        with preserve_adapter_model_eval(self.model):
            ig_tensor = CaptumIntegratedGradients(
                self._score_model, multiply_by_inputs=self.multiply_by_inputs
            ).attribute(
                input_tensor,
                baselines=baseline_tensor,
                target=target,
                n_steps=ig_steps,
                method="gausslegendre",
            )
        dl_values = dl_tensor.detach().cpu().numpy().reshape(-1)
        ig_values = ig_tensor.detach().cpu().numpy().reshape(-1)
        correlation_defined = not (
            float(np.ptp(dl_values)) == 0.0 or float(np.ptp(ig_values)) == 0.0
        )
        if correlation_defined:
            correlation = float(np.corrcoef(dl_values, ig_values)[0, 1])
        else:
            correlation = None
        return {
            "deeplift_attributions": dl_values.tolist(),
            "integrated_gradients_attributions": ig_values.tolist(),
            "correlation": correlation,
            "correlation_defined": correlation_defined,
            "correlation_undefined_reason": (
                None
                if correlation_defined
                else "Pearson correlation is undefined when either attribution vector is constant"
            ),
            "mse": float(np.mean((dl_values - ig_values) ** 2)),
            "max_difference": float(np.max(np.abs(dl_values - ig_values))),
            "ig_steps": ig_steps,
            "output_space": output_space,
            "target_index": target,
            "backends": ["captum.DeepLift", "captum.IntegratedGradients"],
        }


class DeepLIFTShapExplainer(DeepLIFTExplainer):
    """DeepLiftShap: expected DeepLIFT contributions over backgrounds."""

    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100,
        eps: float = 1e-10,
        random_state: Optional[int] = None,
    ):
        if not isinstance(n_background_samples, Integral) or isinstance(n_background_samples, bool):
            raise TypeError("n_background_samples must be an integer")
        if n_background_samples < 1:
            raise ValueError("n_background_samples must be at least 1")
        if eps != 1e-10:
            raise ValueError(
                "Captum DeepLiftShap does not expose a configurable epsilon; "
                "eps must remain 1e-10."
            )
        super().__init__(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            baseline=None,
            multiply_by_inputs=True,
            eps=eps,
            random_state=random_state,
        )
        self.n_background_samples: int = int(n_background_samples)
        self._background_data: Optional[np.ndarray] = None
        if background_data is not None:
            self.set_background(background_data)

    def set_background(self, data: np.ndarray) -> "DeepLIFTShapExplainer":
        data = as_real_array(
            data,
            name="background data",
            dtype=np.float32,
            require_finite=True,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError("background data must have shape " "(n_samples, len(feature_names))")
        if len(data) == 0:
            raise ValueError("background data must contain at least one finite row")
        if len(data) > self.n_background_samples:
            indices = self._new_rng().choice(
                len(data), size=self.n_background_samples, replace=False
            )
            data = data[indices]
        self._background_data = data.copy()
        return self

    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[Union[np.ndarray, str]] = None,
        method: str = "rescale",
        return_convergence_delta: bool = False,
    ) -> Explanation:
        # Before DeepLIFTShap exposed the parent-compatible baseline slot, its
        # third and fourth positional arguments were ``method`` and
        # ``return_convergence_delta``. Preserve those calls while presenting
        # the honest override signature to new callers and type checkers.
        if isinstance(baseline, str):
            legacy_method = baseline
            baseline = None
            if isinstance(method, bool):
                if return_convergence_delta:
                    raise TypeError(
                        "return_convergence_delta was supplied in both legacy and current slots"
                    )
                return_convergence_delta = method
                method = legacy_method
            elif method == "rescale":
                method = legacy_method
            else:
                raise TypeError("method was supplied in both legacy and current slots")
        if baseline is not None:
            raise ValueError(
                "DeepSHAP uses the background distribution set by set_background(); "
                "a per-call baseline is not supported"
            )
        self._validate_method(method)
        if self._background_data is None:
            raise ValueError("Background data not set. Call set_background() first.")

        instance_array = self._as_feature_vector(instance, "instance")
        values, individual, target, output_space = self._attribute_multiple(
            instance_array, self._background_data, target_class
        )
        explanation_data = {
            "feature_attributions": {
                name: float(values[i]) for i, name in enumerate(self.feature_names)
            },
            "attributions_raw": values.tolist(),
            "attributions_std": individual.std(axis=0).tolist(),
            "n_background_samples": len(self._background_data),
            "method": "rescale",
            "backend": (
                "captum.DeepLiftShap" if len(self._background_data) > 1 else "captum.DeepLift"
            ),
            "backend_version": CAPTUM_VERSION,
            "output_space": output_space,
            "target_index": target,
        }

        if return_convergence_delta:
            input_tensor, baseline_tensor, _, _ = self._prepare(
                instance_array, self._background_data, target
            )
            with preserve_adapter_model_eval(self.model), torch.no_grad():
                actual = float(self._score_model(input_tensor)[0, target].item())
                expected = float(self._score_model(baseline_tensor)[:, target].mean().item())
            difference = actual - expected
            attribution_sum = float(values.sum())
            explanation_data.update(
                {
                    "expected_output": expected,
                    "actual_output": actual,
                    "prediction_difference": difference,
                    "attribution_sum": attribution_sum,
                    "convergence_delta": abs(difference - attribution_sum),
                }
            )

        return Explanation(
            explainer_name="DeepSHAP",
            target_class=self._label_name(target),
            explanation_data=explanation_data,
            feature_names=self.feature_names,
        )
