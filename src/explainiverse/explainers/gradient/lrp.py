"""Verified Layer-wise Relevance Propagation for feed-forward PyTorch models.

The public implementation is deliberately narrower than arbitrary PyTorch:
LRP rules are only well-defined here for a single, feed-forward
``torch.nn.Sequential`` chain (or one supported leaf module). Epsilon, gamma,
z-plus, and composite propagation use Captum's canonical LRP machinery.
Alpha-beta propagation is implemented for chains of ``Linear`` and supported
pointwise / reshape layers; convolutional alpha-beta is rejected rather than
silently substituted with another rule.

All reported scores are raw outputs of the wrapped module. When the adapter
identifies a one-output classifier as a logit model, class 1 is the logit and
class 0 is its negative margin. Probability-space LRP would require propagating
through the output activation, which this implementation does not claim to do.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from numbers import Integral
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from explainiverse._torch_module_graph import registered_module_graph
from explainiverse.core.explainer import BaseExplainer, synchronized_explainer_method
from explainiverse.core.explanation import Explanation
from explainiverse.explainers.gradient._input import as_floating_array, scale_safe_sum
from explainiverse.explainers.gradient._model_state import preserve_adapter_model_eval
from explainiverse.explainers.gradient._module_integrity import (
    capture_canonical_forwards,
    require_module_integrity,
    require_no_global_execution_hooks,
)

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
else:
    try:
        import torch
        import torch.nn as nn

        TORCH_AVAILABLE = True
    except ImportError:  # pragma: no cover - exercised only without torch extra
        torch = None
        nn = None
        TORCH_AVAILABLE = False

try:
    from captum.attr import LRP as CaptumLRP
    from captum.attr._utils.lrp_rules import (
        Alpha1_Beta0_Rule,
        EpsilonRule,
        GammaRule,
        IdentityRule,
        PropagationRule,
    )

    CAPTUM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch extra
    CaptumLRP = None
    Alpha1_Beta0_Rule = None
    EpsilonRule = None
    GammaRule = None
    IdentityRule = None
    PropagationRule = object
    CAPTUM_AVAILABLE = False


VALID_RULES = ("epsilon", "gamma", "alpha_beta", "z_plus", "composite")
_CAPTUM_RULES = ("epsilon", "gamma", "z_plus")
_LRP_LOCK = threading.RLock()


if CAPTUM_AVAILABLE:

    class _ReshapeRule(PropagationRule):
        """Preserve relevance values while reversing Flatten / Unflatten."""

        def _manipulate_weights(self, module, inputs, outputs) -> None:
            return None

        def _create_backward_hook_input(self, inputs):
            def hook(grad):
                return self.relevance_output[grad.device].reshape_as(inputs)

            return hook


class LRPExplainer(BaseExplainer):
    """Layer-wise Relevance Propagation for verified sequential PyTorch graphs.

    Supported global rules are ``epsilon``, ``gamma``, ``z_plus``, and
    ``alpha_beta``. ``composite`` assigns epsilon, gamma, or z-plus to selected
    weighted layers and defaults the remainder to epsilon.

    Epsilon, gamma, z-plus, and composite support Linear, Conv2d, BatchNorm1d,
    BatchNorm2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, ReLU, LeakyReLU,
    ELU, Tanh, Sigmoid, Dropout, Flatten, and Unflatten in an
    ``nn.Sequential`` chain. Alpha-beta is limited to Linear plus those
    pointwise / reshape layers.
    """

    _POINTWISE_NATIVE = (nn.ReLU, nn.Tanh, nn.Dropout) if TORCH_AVAILABLE else ()
    _POINTWISE_IDENTITY = (nn.LeakyReLU, nn.ELU, nn.Sigmoid) if TORCH_AVAILABLE else ()
    _RESHAPE = (nn.Flatten, nn.Unflatten) if TORCH_AVAILABLE else ()
    _WEIGHTED = (nn.Linear, nn.Conv2d) if TORCH_AVAILABLE else ()
    _NORMALIZATION = (nn.BatchNorm1d, nn.BatchNorm2d) if TORCH_AVAILABLE else ()
    _POOLING = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d) if TORCH_AVAILABLE else ()
    _STANDARD_TYPES = (
        (nn.Sequential,)
        + _WEIGHTED
        + _NORMALIZATION
        + _POOLING
        + _POINTWISE_NATIVE
        + _POINTWISE_IDENTITY
        + _RESHAPE
        if TORCH_AVAILABLE
        else ()
    )
    _CANONICAL_FORWARDS = capture_canonical_forwards(_STANDARD_TYPES) if TORCH_AVAILABLE else {}
    _CANONICAL_CAPTUM_METHODS = (
        {
            name: getattr(nn.Module, name)
            for name in (
                "__call__",
                "_call_impl",
                "_wrapped_call_impl",
                "children",
                "modules",
                "register_forward_hook",
                "register_forward_pre_hook",
            )
            if hasattr(nn.Module, name)
        }
        if TORCH_AVAILABLE
        else {}
    )
    _CANONICAL_STATE_METHODS = (
        {
            "state_dict": nn.Module.state_dict,
            "load_state_dict": nn.Module.load_state_dict,
        }
        if TORCH_AVAILABLE
        else {}
    )

    def __init__(
        self,
        model,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        rule: str = "epsilon",
        epsilon: float = 1e-6,
        gamma: float = 0.25,
        alpha: float = 2.0,
        beta: float = 1.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("LRP requires PyTorch. Install explainiverse[torch].")
        if not CAPTUM_AVAILABLE:
            raise ImportError(
                "LRP requires Captum. Install explainiverse[torch] or pip install captum."
            )

        super().__init__(model)
        if not hasattr(model, "model") or not isinstance(model.model, nn.Module):
            raise TypeError("LRP requires a PyTorchAdapter wrapping an nn.Module.")
        if getattr(model, "task", None) not in {"classification", "regression"}:
            raise TypeError("LRP requires an adapter with an explicit task contract.")
        if rule not in VALID_RULES:
            raise ValueError(f"rule must be one of {VALID_RULES}; got {rule!r}")
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a finite non-negative number")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be a finite non-negative number")
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("alpha and beta must be finite")
        if alpha < 1 or beta < 0 or not np.isclose(alpha - beta, 1.0):
            raise ValueError("alpha_beta requires alpha >= 1, beta >= 0, and alpha - beta = 1")

        self.feature_names = self._validate_names(feature_names, "feature_names")
        inherited_classes = getattr(model, "class_names", None)
        effective_classes = class_names if class_names is not None else inherited_classes
        self.class_names = (
            self._validate_names(effective_classes, "class_names")
            if effective_classes is not None
            else None
        )
        self.rule = rule
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._layer_rules: Optional[Dict[int, str]] = None
        self._layers_info: Optional[List[Dict[str, Any]]] = None

        self._leaf_layers = self._validate_architecture(model.model)
        self._model_topology = self._architecture_token(model.model, self._leaf_layers)
        if self.rule == "alpha_beta":
            self._validate_alpha_beta_architecture()

    @staticmethod
    def _validate_names(names, field: str) -> List[str]:
        if isinstance(names, (str, bytes)):
            raise ValueError(f"{field} must be a non-empty sequence of unique strings")
        try:
            result = list(names)
        except TypeError as error:
            raise ValueError(f"{field} must be a non-empty sequence of unique strings") from error
        if not result:
            raise ValueError(f"{field} must be a non-empty sequence of unique strings")
        if any(not isinstance(name, str) or not name.strip() for name in result):
            raise ValueError(f"{field} must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"{field} must contain unique names")
        return result

    def _validate_architecture(self, model: nn.Module):
        supported = (
            self._WEIGHTED
            + self._NORMALIZATION
            + self._POOLING
            + self._POINTWISE_NATIVE
            + self._POINTWISE_IDENTITY
            + self._RESHAPE
        )
        if type(model) is not nn.Sequential and type(model) not in supported:
            raise TypeError(
                "LRP supports only an nn.Sequential feed-forward chain or one "
                f"supported leaf module; got {type(model).__name__}."
            )

        require_no_global_execution_hooks(context="LRP")
        registered_module_graph(model)

        leaves: List[Tuple[int, str, nn.Module]] = []
        seen: set[int] = set()

        def visit(module: nn.Module, path: str) -> None:
            if id(module) in seen:
                raise TypeError(
                    f"Layer {path or '<root>'} is reused; shared modules are unsupported."
                )
            seen.add(id(module))
            module_type = type(module)
            if module_type is not nn.Sequential and module_type not in supported:
                raise TypeError(
                    f"Unsupported LRP layer {module_type.__name__} at {path or '<root>'}."
                )
            require_module_integrity(
                module,
                path=path,
                context="LRP",
                canonical_forward=self._CANONICAL_FORWARDS[module_type],
                canonical_methods=self._CANONICAL_CAPTUM_METHODS,
                check_state_io=True,
                canonical_state_methods=self._CANONICAL_STATE_METHODS,
            )
            if type(module) is nn.Sequential:
                children = object.__getattribute__(module, "__dict__")["_modules"]
                for name, child in children.items():
                    child_path = f"{path}.{name}" if path else name
                    visit(child, child_path)
                return
            if isinstance(module, self._NORMALIZATION) and (
                not module.track_running_stats
                or module.running_mean is None
                or module.running_var is None
            ):
                raise TypeError(
                    f"{type(module).__name__} at {path or '<root>'} requires "
                    "tracked running statistics for deterministic eval-mode LRP."
                )
            if isinstance(module, nn.MaxPool2d) and module.return_indices:
                raise TypeError("MaxPool2d(return_indices=True) is unsupported by LRP")
            leaves.append((len(leaves), path or "model", module))

        visit(model, "")
        if not leaves:
            raise TypeError("LRP requires at least one supported leaf layer")
        return leaves

    @staticmethod
    def _architecture_token(model: nn.Module, leaves) -> tuple:
        return (
            id(model),
            type(model),
            tuple((index, path, id(layer), type(layer)) for index, path, layer in leaves),
        )

    def _validate_current_architecture(self, model: nn.Module) -> None:
        leaves = self._validate_architecture(model)
        if self._architecture_token(model, leaves) != self._model_topology:
            raise RuntimeError(
                "The LRP model graph changed after explainer construction; construct "
                "a new explainer for the new exact module graph."
            )

    def _validate_alpha_beta_architecture(self) -> None:
        supported = (nn.Linear,) + self._POINTWISE_NATIVE + self._POINTWISE_IDENTITY + self._RESHAPE
        unsupported = [
            (idx, name, type(layer).__name__)
            for idx, name, layer in self._leaf_layers
            if not isinstance(layer, supported)
        ]
        if unsupported:
            details = ", ".join(
                f"{layer_type} at {idx} ({name})" for idx, name, layer_type in unsupported
            )
            raise NotImplementedError(
                "alpha_beta is verified only for Linear plus pointwise/reshape "
                f"sequential layers; unsupported: {details}"
            )

    def _get_pytorch_model(self) -> nn.Module:
        return self.model.model

    def _get_model_device(self) -> torch.device:
        try:
            return next(self._get_pytorch_model().parameters()).device
        except StopIteration:
            return torch.device(getattr(self.model, "device", "cpu"))

    def _prepare_input_tensor(self, instance: np.ndarray) -> torch.Tensor:
        array = np.asarray(instance)
        if array.ndim == 0 or array.size == 0:
            raise ValueError("instance must contain at least one feature")
        array = as_floating_array(array, name="instance")
        if len(self.feature_names) != array.size:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but instance "
                f"contains {array.size} scalar features"
            )
        first_shape_layer = next(
            (
                layer
                for _, _, layer in self._leaf_layers
                if not isinstance(layer, self._POINTWISE_NATIVE + self._POINTWISE_IDENTITY)
            ),
            None,
        )
        spatial_layers = (nn.Conv2d, nn.BatchNorm2d) + self._POOLING
        if isinstance(first_shape_layer, spatial_layers) and array.ndim != 3:
            raise ValueError(
                "A spatial LRP instance must have shape (channels, height, width); "
                f"got {tuple(array.shape)}"
            )
        if (
            isinstance(first_shape_layer, (nn.Linear, nn.BatchNorm1d, nn.Unflatten))
            and array.ndim != 1
        ):
            raise ValueError(
                "A tabular LRP instance must be one-dimensional; " f"got {tuple(array.shape)}"
            )
        tensor = self.model._to_tensor(array).detach().clone()
        if not tensor.is_floating_point():
            raise TypeError("LRP inputs must resolve to a floating-point model dtype")
        return tensor.unsqueeze(0)

    def _forward_leafwise(self, x: torch.Tensor):
        current = x
        records = []
        for idx, name, layer in self._leaf_layers:
            activation = current.detach().clone()
            current = layer(current)
            records.append((idx, name, layer, activation))
        return current, records

    @staticmethod
    def _normalise_output(output: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("The PyTorch model must return a Tensor")
        if output.ndim == 1:
            output = output.unsqueeze(1)
        if output.ndim != 2 or output.shape[0] != 1:
            raise ValueError(
                "LRP requires one scalar or vector model output per instance; "
                f"got shape {tuple(output.shape)}"
            )
        if not bool(torch.isfinite(output).all()):
            raise ValueError("The model returned non-finite output values")
        return output

    def _resolve_target(
        self, output: torch.Tensor, target_class: Optional[int]
    ) -> Tuple[int, float, str, str]:
        n_outputs = output.shape[1]
        task = self.model.task

        if target_class is not None and (
            isinstance(target_class, (bool, np.bool_)) or not isinstance(target_class, Integral)
        ):
            raise TypeError("target_class must be an integer output index")

        if task == "regression":
            if n_outputs > 1 and target_class is None:
                raise ValueError(
                    "Multi-output regression requires an explicit output index in target_class"
                )
            index = 0 if target_class is None else int(target_class)
            if index < 0 or index >= n_outputs:
                raise ValueError(f"target_class output index must be in [0, {n_outputs - 1}]")
            label = "output" if n_outputs == 1 else f"output_{index}"
            return index, 1.0, label, "model_output"

        if n_outputs == 1:
            predictions = self.model._prediction_output(output)
            index = (
                int(torch.argmax(predictions, dim=1).item())
                if target_class is None
                else int(target_class)
            )
            if index not in (0, 1):
                raise ValueError("target_class must be 0 or 1 for a one-logit classifier")
            if self.class_names is not None and len(self.class_names) != 2:
                raise ValueError("A one-logit binary classifier requires two class_names")
            label = self.class_names[index] if self.class_names else f"class_{index}"
            if getattr(self.model, "output_activation", None) is None:
                if index == 0:
                    raise ValueError(
                        "Class-0 LRP is undefined for a one-output probability model: "
                        "the model exposes P(class 1), while 1-P(class 1) includes an "
                        "unrepresented constant. Explain class 1 or use a logit model."
                    )
                return 0, 1.0, label, "model_probability"
            if index == 0 and self._uses_sign_asymmetric_rule():
                raise ValueError(
                    "Class-0 LRP for a one-logit classifier is unsupported with "
                    "gamma, z-plus, alpha-beta, or a composite containing those "
                    "rules. Negating an attribution after propagation is not "
                    "equivalent to applying a sign-asymmetric rule to the "
                    "explicit -logit score. Use rule='epsilon' or expose two "
                    "model output scores."
                )
            return 0, (1.0 if index == 1 else -1.0), label, "binary_logit_margin"

        index = (
            int(torch.argmax(output, dim=1).item()) if target_class is None else int(target_class)
        )
        if index < 0 or index >= n_outputs:
            raise ValueError(f"target_class must be in [0, {n_outputs - 1}]")
        if self.class_names is not None and len(self.class_names) != n_outputs:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries for {n_outputs} outputs"
            )
        label = self.class_names[index] if self.class_names else f"class_{index}"
        return index, 1.0, label, "model_output"

    def _uses_sign_asymmetric_rule(self) -> bool:
        """Return whether the configured propagation changes under ``f -> -f``."""

        asymmetric = {"gamma", "z_plus", "alpha_beta"}
        if self.rule in asymmetric:
            return True
        if self.rule != "composite":
            return False
        return any(rule in asymmetric for rule in self._effective_layer_rules().values())

    def _get_rule_for_layer(self, layer_idx: int, layer_type: str = "") -> str:
        if self.rule != "composite":
            return self.rule
        if self._layer_rules and layer_idx in self._layer_rules:
            return self._layer_rules[layer_idx]
        return "epsilon"

    @synchronized_explainer_method
    def set_composite_rule(self, layer_rules: Dict[int, str]) -> "LRPExplainer":
        if self.rule != "composite":
            raise ValueError("set_composite_rule requires rule='composite'")
        if not isinstance(layer_rules, dict):
            raise TypeError("layer_rules must be a dictionary")
        weighted_indices = {
            idx for idx, _, layer in self._leaf_layers if isinstance(layer, self._WEIGHTED)
        }
        validated: Dict[int, str] = {}
        for index, layer_rule in layer_rules.items():
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError("composite layer indices must be integers")
            index = int(index)
            if index not in weighted_indices:
                raise ValueError(
                    f"Composite rule index {index} is not a weighted Linear/Conv2d layer"
                )
            if layer_rule not in _CAPTUM_RULES:
                raise ValueError(
                    "Composite layers support only epsilon, gamma, or z_plus; "
                    f"got {layer_rule!r}"
                )
            validated[index] = layer_rule
        self._layer_rules = validated
        return self

    def _validate_z_plus_activations(self, records) -> None:
        for idx, name, layer, activation in records:
            if not isinstance(layer, self._WEIGHTED):
                continue
            if self._get_rule_for_layer(idx) != "z_plus":
                continue
            minimum = float(activation.min().item())
            if minimum < -1e-7:
                raise ValueError(
                    "z_plus requires non-negative inputs to every weighted layer; "
                    f"layer {idx} ({name}) received minimum activation {minimum:.6g}"
                )

    def _make_weighted_rule(self, rule: str):
        if rule == "epsilon":
            return EpsilonRule(epsilon=self.epsilon)
        if rule == "gamma":
            return GammaRule(gamma=self.gamma)
        if rule == "z_plus":
            return Alpha1_Beta0_Rule(set_bias_to_zero=True)
        raise ValueError(f"Unsupported Captum LRP rule {rule!r}")

    def _attach_captum_rules(self):
        attached = []
        for idx, name, layer in self._leaf_layers:
            rule_object = None
            if isinstance(layer, self._WEIGHTED):
                rule_object = self._make_weighted_rule(self._get_rule_for_layer(idx))
            elif isinstance(layer, self._NORMALIZATION + self._POOLING):
                rule_object = EpsilonRule(epsilon=self.epsilon)
            elif isinstance(layer, self._RESHAPE):
                rule_object = _ReshapeRule()
            elif isinstance(layer, self._POINTWISE_IDENTITY):
                rule_object = IdentityRule()

            if rule_object is not None:
                layer.rule = rule_object
                attached.append((idx, name, layer, rule_object))
        return attached

    @staticmethod
    def _restore_training_flags(flags) -> None:
        for module, training in flags.items():
            module.training = training

    def _captum_lrp(self, x, target_index, sign, records):
        model = self._get_pytorch_model()
        if self.rule == "z_plus" or (
            self.rule == "composite"
            and self._layer_rules
            and "z_plus" in self._layer_rules.values()
        ):
            self._validate_z_plus_activations(records)

        training_flags = {module: module.training for module in registered_module_graph(model)}
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        old_rules = {
            layer: (hasattr(layer, "rule"), getattr(layer, "rule", None))
            for _, _, layer in self._leaf_layers
        }
        attached = []
        try:
            model.eval()
            attached = self._attach_captum_rules()
            x_for_lrp = x.detach().clone().requires_grad_(True)
            # Captum discovers leaf layers by traversing children. A module
            # that is itself a single Linear / Conv2d therefore needs a
            # transparent Sequential container for its rule to be attached.
            backend_model = model if type(model) is nn.Sequential else nn.Sequential(model)
            attributions, _ = CaptumLRP(backend_model).attribute(
                x_for_lrp,
                target=target_index,
                return_convergence_delta=True,
            )
            attributions = attributions * sign

            target_score = float(
                self._normalise_output(model(x.detach()))[0, target_index].item() * sign
            )
            layer_relevances: "OrderedDict[str, np.ndarray]" = OrderedDict()
            layer_relevances["output"] = np.asarray([target_score], dtype=float)
            for idx, name, _layer, rule_object in reversed(attached):
                stored = rule_object.relevance_input.get(x.device)
                if isinstance(stored, list):
                    if len(stored) != 1:
                        raise RuntimeError(
                            f"Layer {idx} ({name}) has multiple relevance inputs; "
                            "only single-input sequential layers are supported"
                        )
                    stored = stored[0]
                if stored is not None:
                    actual = stored.detach() * target_score
                    layer_relevances[f"layer_{idx}_{name}_{type(_layer).__name__}"] = (
                        self.model._to_numpy(actual).reshape(-1)
                    )
            layer_relevances["input"] = self.model._to_numpy(attributions).reshape(-1)
            return attributions.detach(), layer_relevances
        finally:
            model.load_state_dict(state, strict=True)
            for layer, (had_rule, old_rule) in old_rules.items():
                if had_rule:
                    layer.rule = old_rule
                elif hasattr(layer, "rule"):
                    delattr(layer, "rule")
                for temporary in ("activations",):
                    if hasattr(layer, temporary):
                        delattr(layer, temporary)
            self._restore_training_flags(training_flags)

    def _alpha_beta_linear(self, layer, activation, relevance):
        if activation.ndim != 2 or relevance.ndim != 2:
            raise ValueError("alpha_beta Linear propagation requires 2D activations")
        weight_positive = layer.weight.clamp(min=0)
        weight_negative = layer.weight.clamp(max=0)
        activation_positive = activation.clamp(min=0)
        activation_negative = activation.clamp(max=0)

        denominator_positive = (
            activation_positive @ weight_positive.T + activation_negative @ weight_negative.T
        )
        denominator_negative = (
            activation_positive @ weight_negative.T + activation_negative @ weight_positive.T
        )
        stable_positive = denominator_positive + self.epsilon
        stable_negative = denominator_negative - self.epsilon
        if self.epsilon == 0:
            stable_positive = torch.where(
                denominator_positive == 0,
                torch.ones_like(denominator_positive),
                denominator_positive,
            )
            stable_negative = torch.where(
                denominator_negative == 0,
                -torch.ones_like(denominator_negative),
                denominator_negative,
            )

        positive_scale = relevance / stable_positive
        negative_scale = relevance / stable_negative
        positive_contribution = activation_positive * (
            positive_scale @ weight_positive
        ) + activation_negative * (positive_scale @ weight_negative)
        negative_contribution = activation_positive * (
            negative_scale @ weight_negative
        ) + activation_negative * (negative_scale @ weight_positive)
        return self.alpha * positive_contribution - self.beta * negative_contribution

    def _alpha_beta_lrp(self, x, target_index, sign, output, records):
        self._validate_alpha_beta_architecture()
        relevance = torch.zeros_like(output)
        relevance[0, target_index] = output[0, target_index] * sign
        target_score = float(relevance[0, target_index].item())
        layer_relevances: "OrderedDict[str, np.ndarray]" = OrderedDict()
        layer_relevances["output"] = np.asarray([target_score], dtype=float)

        for idx, name, layer, activation in reversed(records):
            if isinstance(layer, nn.Linear):
                relevance = self._alpha_beta_linear(layer, activation, relevance)
            elif isinstance(layer, self._RESHAPE):
                relevance = relevance.reshape_as(activation)
            # Pointwise activations and eval-mode dropout preserve relevance.
            layer_relevances[f"layer_{idx}_{name}_{type(layer).__name__}"] = self.model._to_numpy(
                relevance
            ).reshape(-1)

        layer_relevances["input"] = self.model._to_numpy(relevance).reshape(-1)
        return relevance.detach(), layer_relevances

    def _compute(self, instance: np.ndarray, target_class: Optional[int]) -> Dict[str, Any]:
        model = self._get_pytorch_model()
        # The preliminary leafwise forward is model work too: user hooks can
        # consume Torch RNG or mutate registered buffers before Captum's own
        # state snapshot. Preserve the complete operation, including failures.
        with preserve_adapter_model_eval(self.model), _LRP_LOCK:
            self._validate_current_architecture(model)
            x = self._prepare_input_tensor(instance)
            training_flags = {module: module.training for module in registered_module_graph(model)}
            try:
                model.eval()
                with torch.no_grad():
                    output, records = self._forward_leafwise(x)
                    output = self._normalise_output(output)
                target_index, sign, label, score_space = self._resolve_target(output, target_class)

                if self.rule == "alpha_beta":
                    attributions, layer_relevances = self._alpha_beta_lrp(
                        x, target_index, sign, output, records
                    )
                else:
                    attributions, layer_relevances = self._captum_lrp(
                        x, target_index, sign, records
                    )
            finally:
                self._restore_training_flags(training_flags)

        flat = self.model._to_numpy(attributions).reshape(-1)
        if not np.isfinite(flat).all():
            raise FloatingPointError(
                "LRP produced non-finite relevance. Use epsilon > 0 or inspect "
                "zero denominators in the selected rule."
            )
        target_score = float(output[0, target_index].item() * sign)
        attribution_sum = float(scale_safe_sum(flat))
        signed_delta = target_score - attribution_sum
        semantic_target_index = target_index
        if self.model.task == "classification" and output.shape[1] == 1:
            semantic_target_index = 1 if sign > 0 else 0
        return {
            "attributions": flat,
            "layer_relevances": layer_relevances,
            "target_index": semantic_target_index,
            "backend_target_index": target_index,
            "label": label,
            "score_space": score_space,
            "target_score": target_score,
            "attribution_sum": attribution_sum,
            "signed_delta": signed_delta,
        }

    def _compute_lrp(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        return_layer_relevances: bool = False,
    ):
        if not isinstance(return_layer_relevances, (bool, np.bool_)):
            raise TypeError("return_layer_relevances must be a boolean")
        result = self._compute(instance, target_class)
        if return_layer_relevances:
            return result["attributions"], result["layer_relevances"]
        return result["attributions"]

    def _bias_treatment(self) -> str:
        if self.rule == "epsilon":
            return "included_in_denominators_not_redistributed"
        if self.rule == "gamma":
            return "retained_in_transformed_denominators_not_separately_redistributed"
        if self.rule == "z_plus":
            return (
                "weighted_layer_biases_excluded_from_local_denominators; "
                "original_output_relevance_is_propagated"
            )
        if self.rule == "alpha_beta":
            return (
                "excluded_from_local_denominators; " "output_relevance_including_bias_is_propagated"
            )
        return "per_layer_rule"

    def _effective_layer_rules(self) -> Dict[int, str]:
        assignments: Dict[int, str] = {}
        for index, _name, layer in self._leaf_layers:
            if isinstance(layer, self._WEIGHTED):
                assignments[index] = self._get_rule_for_layer(index)
            elif isinstance(layer, self._NORMALIZATION + self._POOLING):
                assignments[index] = "epsilon"
            elif isinstance(layer, self._RESHAPE):
                assignments[index] = "relevance_preserving_reshape"
            else:
                assignments[index] = "relevance_passthrough"
        return assignments

    def _effective_layer_parameters(self) -> Dict[int, Dict[str, Any]]:
        parameters: Dict[int, Dict[str, Any]] = {}
        for index, rule in self._effective_layer_rules().items():
            values: Dict[str, Any] = {"rule": rule}
            if rule == "epsilon":
                values["epsilon"] = self.epsilon
            elif rule == "gamma":
                values.update(
                    gamma=self.gamma,
                    stability_factor=float(GammaRule.STABILITY_FACTOR),
                )
            elif rule == "z_plus":
                values.update(
                    stability_factor=float(Alpha1_Beta0_Rule.STABILITY_FACTOR),
                    weighted_bias_set_to_zero=True,
                )
            elif rule == "alpha_beta":
                values.update(alpha=self.alpha, beta=self.beta, epsilon=self.epsilon)
            parameters[index] = values
        return parameters

    @synchronized_explainer_method
    def explain(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        return_convergence_delta: bool = False,
    ) -> Explanation:
        if not isinstance(return_convergence_delta, (bool, np.bool_)):
            raise TypeError("return_convergence_delta must be a boolean")
        original = np.asarray(instance)
        result = self._compute(original, target_class)
        raw = result["attributions"]
        attributions = {name: float(raw[index]) for index, name in enumerate(self.feature_names)}
        data = {
            "feature_attributions": attributions,
            "attributions_raw": [float(value) for value in raw],
            "rule": self.rule,
            "backend": ("native_alpha_beta" if self.rule == "alpha_beta" else "captum.attr.LRP"),
            "score_space": result["score_space"],
            "target_class_index": int(result["target_index"]),
            "backend_target_index": int(result["backend_target_index"]),
            "target_output": result["target_score"],
            "attribution_sum": result["attribution_sum"],
            "signed_convergence_delta": result["signed_delta"],
            "convergence_delta": abs(result["signed_delta"]),
            "bias_treatment": self._bias_treatment(),
            "effective_layer_rules": self._effective_layer_rules(),
            "effective_layer_parameters": self._effective_layer_parameters(),
            "epsilon": self.epsilon,
            "gamma": self.gamma if self.rule in {"gamma", "composite"} else None,
            "alpha": self.alpha if self.rule == "alpha_beta" else None,
            "beta": self.beta if self.rule == "alpha_beta" else None,
            "input_shape": list(original.shape),
            "supported_graph": "single_input_nn.Sequential_chain_or_supported_leaf",
        }
        if self.rule == "composite":
            data["layer_rules"] = dict(self._layer_rules or {})
        if not return_convergence_delta:
            # The score and residual remain present because they define the LRP
            # contract; the flag is retained for API compatibility.
            data["convergence_delta_requested"] = False

        return Explanation(
            explainer_name="LRP",
            target_class=result["label"],
            explanation_data=data,
            feature_names=self.feature_names,
        )

    @synchronized_explainer_method
    def explain_batch(self, X: np.ndarray, target_class: Optional[int] = None) -> List[Explanation]:
        array = np.asarray(X)
        if array.ndim == 1:
            return [self.explain(array, target_class=target_class)]
        if array.ndim < 2:
            raise ValueError("X must contain one or more instances")
        return [
            self.explain(array[index], target_class=target_class) for index in range(array.shape[0])
        ]

    @synchronized_explainer_method
    def explain_with_layer_relevances(
        self, instance: np.ndarray, target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        result = self._compute(instance, target_class)
        return {
            "input_relevances": [float(value) for value in result["attributions"]],
            "layer_relevances": {
                name: [float(value) for value in np.asarray(relevance).reshape(-1)]
                for name, relevance in result["layer_relevances"].items()
            },
            "target_class": int(result["target_index"]),
            "target_output": result["target_score"],
            "score_space": result["score_space"],
            "rule": self.rule,
            "feature_names": self.feature_names,
            "layer_relevance_scope": (
                "all_sequential_layer_inputs_plus_input_output"
                if self.rule == "alpha_beta"
                else "captum_rule_inputs_plus_input_output"
            ),
        }

    @synchronized_explainer_method
    def compare_rules(
        self,
        instance: np.ndarray,
        target_class: Optional[int] = None,
        rules: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        requested = ["epsilon", "gamma", "alpha_beta", "z_plus"] if rules is None else list(rules)
        invalid = [rule for rule in requested if rule not in VALID_RULES or rule == "composite"]
        if invalid:
            raise ValueError(f"Invalid standalone comparison rules: {invalid}")

        original_rule = self.rule
        results: Dict[str, Dict[str, Any]] = {}
        try:
            for rule in requested:
                self.rule = rule
                try:
                    computation = self._compute(instance, target_class)
                    values = computation["attributions"]
                    top_index = int(np.argmax(np.abs(values)))
                    results[rule] = {
                        "attributions": [float(value) for value in values],
                        "top_feature": self.feature_names[top_index],
                        "top_attribution": float(values[top_index]),
                        "attribution_sum": float(scale_safe_sum(values)),
                        "attribution_range": (
                            float(values.min()),
                            float(values.max()),
                        ),
                        "target_output": computation["target_score"],
                        "convergence_delta": abs(computation["signed_delta"]),
                    }
                except (ValueError, NotImplementedError, FloatingPointError) as error:
                    results[rule] = {"error": str(error)}
        finally:
            self.rule = original_rule
        return results
