# src/explainiverse/engine/suite.py
"""Run compatible explainers without pretending unlike outputs are comparable."""

import inspect
import warnings
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from explainiverse.core.explanation import Explanation


class _RegistryLike(Protocol):
    """Minimal registry surface used by the suite and its test doubles."""

    def create(self, name: str, **kwargs: Any) -> Any: ...

    def get_meta(self, name: str) -> Any: ...


class ExplanationSuite:
    """
    Run and compare multiple explainers on the same instances.

    This class provides a unified interface for:
    - Running multiple explainers on a single instance
    - Comparing attribution scores side-by-side
    - Listing metadata-compatible, accuracy-audited explainers
    - Evaluating explainers using ROAR (Remove And Retrain)

    Example:
        >>> from explainiverse import ExplanationSuite, SklearnAdapter
        >>> suite = ExplanationSuite(
        ...     model=adapter,
        ...     explainer_configs=[
        ...         ("lime", {"training_data": X_train, "feature_names": fnames, "class_names": cnames}),
        ...         ("shap", {"background_data": X_train[:50], "feature_names": fnames, "class_names": cnames}),
        ...     ]
        ... )
        >>> results = suite.run(X_test[0])
        >>> suite.compare(allow_incommensurate=True)  # descriptive display only
    """

    def __init__(
        self,
        model,
        explainer_configs: List[Tuple[str, Dict[str, Any]]],
        data_meta: Optional[Dict[str, Any]] = None,
        explainer_call_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        """
        Initialize the ExplanationSuite.

        Args:
            model: A model adapter (e.g., SklearnAdapter, PyTorchAdapter)
            explainer_configs: List of (explainer_name, kwargs) tuples.
                The explainer_name should match a registered explainer in
                the default_registry (e.g., "lime", "shap", "treeshap").
            data_meta: Optional metadata about the task, scope, or preference.
                Can include "task" ("classification" or "regression").
            explainer_call_kwargs: Optional per-explainer keyword arguments for
                ``explain``. This is separate from constructor configuration so
                methods with an additional invocation contract, such as
                ProtoDash's required ``X_reference``, can be called honestly.
        """
        if not isinstance(explainer_configs, list) or not explainer_configs:
            raise ValueError("explainer_configs must be a non-empty list.")
        validated_configs: List[Tuple[str, Dict[str, Any]]] = []
        names: List[str] = []
        for index, config in enumerate(explainer_configs):
            if not isinstance(config, tuple) or len(config) != 2:
                raise TypeError(f"explainer_configs[{index}] must be a (name, kwargs) tuple.")
            name, params = config
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"explainer_configs[{index}] has an invalid name.")
            if not isinstance(params, dict):
                raise TypeError(f"Configuration for {name!r} must be a dictionary.")
            names.append(name)
            validated_configs.append((name, dict(params)))
        if len(names) != len(set(names)):
            raise ValueError("explainer_configs must not contain duplicate names.")
        if data_meta is not None and not isinstance(data_meta, dict):
            raise TypeError("data_meta must be a dictionary or None.")
        if explainer_call_kwargs is not None and not isinstance(explainer_call_kwargs, Mapping):
            raise TypeError("explainer_call_kwargs must be a mapping or None.")

        validated_call_kwargs: Dict[str, Dict[str, Any]] = {}
        for name, call_kwargs in dict(explainer_call_kwargs or {}).items():
            if name not in names:
                raise ValueError(f"explainer_call_kwargs contains unconfigured explainer {name!r}.")
            if not isinstance(call_kwargs, Mapping):
                raise TypeError(f"Call arguments for {name!r} must be a mapping.")
            if "instance" in call_kwargs:
                raise ValueError(
                    f"Call arguments for {name!r} must not override the suite instance."
                )
            validated_call_kwargs[name] = dict(call_kwargs)

        self.model = model
        self.configs = validated_configs
        self.call_kwargs = validated_call_kwargs
        self.data_meta = dict(data_meta or {})
        self.explanations: Dict[str, Any] = {}
        self._registry: Optional[_RegistryLike] = None

    def _get_registry(self) -> _RegistryLike:
        """Lazy load the registry to avoid circular imports."""
        if self._registry is None:
            from explainiverse.core.registry import default_registry

            self._registry = default_registry
        return self._registry

    def _merged_call_kwargs(
        self,
        name: str,
        overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return validated per-method call arguments without mutating config."""
        merged = dict(self.call_kwargs.get(name, {}))
        if overrides is None:
            return merged
        if not isinstance(overrides, Mapping):
            raise TypeError("call_kwargs_by_explainer must be a mapping or None.")
        unknown = set(overrides) - set(self.list_explainers())
        if unknown:
            raise ValueError(
                "call_kwargs_by_explainer contains unconfigured explainer(s): "
                f"{sorted(unknown)}."
            )
        if name not in overrides:
            return merged
        supplied = overrides[name]
        if not isinstance(supplied, Mapping):
            raise TypeError(f"Call arguments for {name!r} must be a mapping.")
        if "instance" in supplied:
            raise ValueError(f"Call arguments for {name!r} must not override the suite instance.")
        merged.update(supplied)
        return merged

    @staticmethod
    def _require_local_scope(registry: _RegistryLike, name: str) -> None:
        """Reject dataset/global methods from the suite's instance runner."""
        get_meta = getattr(registry, "get_meta", None)
        if not callable(get_meta):
            return
        meta = get_meta(name)
        if meta.scope != "local":
            raise ValueError(
                f"Explainer {name!r} has scope={meta.scope!r}; ExplanationSuite.run() "
                "only executes local instance-level methods. Invoke the concrete "
                "explainer with its dataset/global contract instead."
            )

    @staticmethod
    def _explain_instance(explainer: Any, name: str, instance: np.ndarray, kwargs: Dict[str, Any]):
        """Bind and execute a concrete explainer's declared instance contract."""
        try:
            inspect.signature(explainer.explain).bind(instance, **kwargs)
        except TypeError as exc:
            raise ValueError(
                f"Explainer {name!r} cannot be invoked with only the suite instance and "
                f"configured call arguments {sorted(kwargs)}: {exc}. Supply its required "
                "arguments through explainer_call_kwargs or call_kwargs_by_explainer."
            ) from exc
        return explainer.explain(instance, **kwargs)

    def _configured_for_instance_run(
        self,
        registry: _RegistryLike,
        name: str,
        params: Mapping[str, Any],
    ) -> bool:
        """Check scope, construction, and call signatures without executing."""
        try:
            meta = registry.get_meta(name)
        except (AttributeError, KeyError):
            return True
        if meta.scope != "local":
            return False

        get_entry = getattr(registry, "get", None)
        if not callable(get_entry):
            return True
        explainer_class = get_entry(name)["class"]
        try:
            inspect.signature(explainer_class).bind(model=self.model, **params)
        except TypeError:
            return False
        except (ValueError, AttributeError):
            # Some extension types do not expose an inspectable signature. The
            # registry constructor remains the authoritative runtime check.
            pass

        method = getattr(explainer_class, "explain", None)
        if not callable(method):
            return False
        try:
            # Unbound methods consume ``self`` and the suite-supplied instance.
            inspect.signature(method).bind(None, object(), **self.call_kwargs.get(name, {}))
        except TypeError:
            return False
        return True

    def _create_configured_explainer(
        self,
        registry: _RegistryLike,
        name: str,
        params: Mapping[str, Any],
    ) -> Any:
        """Construct an explainer after reporting configuration errors clearly."""
        get_entry = getattr(registry, "get", None)
        if callable(get_entry):
            explainer_class = get_entry(name)["class"]
            try:
                inspect.signature(explainer_class).bind(model=self.model, **params)
            except TypeError as exc:
                raise ValueError(
                    f"Explainer {name!r} cannot be constructed with configured arguments "
                    f"{sorted(params)}: {exc}. Add its required constructor arguments to "
                    "explainer_configs."
                ) from exc
            except (ValueError, AttributeError):
                pass
        return registry.create(name, model=self.model, **params)

    def run(
        self,
        instance: np.ndarray,
        *,
        call_kwargs_by_explainer: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run all configured explainers on a single instance.

        Args:
            instance: Numeric input for one explanation call. Exact shape is
                part of each concrete explainer's contract.

        Returns:
            Dictionary mapping explainer names to Explanation objects
        """
        self.explanations = {}
        try:
            instance = np.asarray(instance)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"instance must be a numeric array: {exc}") from exc
        if instance.ndim == 0 or instance.size == 0:
            raise ValueError(
                f"instance must be a non-empty array with at least one dimension, "
                f"got {instance.shape}."
            )
        if instance.dtype == np.bool_:
            raise TypeError("instance must contain numeric values, not booleans.")
        if not np.issubdtype(instance.dtype, np.number):
            raise TypeError("instance must contain real numeric values.")
        if np.issubdtype(instance.dtype, np.complexfloating):
            raise TypeError("instance must contain real numeric values, not complex values.")
        if not np.all(np.isfinite(instance)):
            raise ValueError("instance must contain only finite values.")
        # Isolate caller state while retaining semantically meaningful dtypes
        # such as integer token indices and float32 model inputs.
        instance = instance.copy()
        if call_kwargs_by_explainer is not None:
            # Validate unknown names even when the suite has no explainers after
            # a future filtering extension.
            self._merged_call_kwargs(self.configs[0][0], call_kwargs_by_explainer)

        registry = self._get_registry()
        current: Dict[str, Explanation] = {}

        for name, params in self.configs:
            self._require_local_scope(registry, name)
            explainer = self._create_configured_explainer(registry, name, params)
            call_kwargs = self._merged_call_kwargs(name, call_kwargs_by_explainer)
            explanation = self._explain_instance(explainer, name, instance.copy(), call_kwargs)
            if not isinstance(explanation, Explanation):
                raise TypeError(
                    f"Explainer {name!r} returned {type(explanation).__name__}; "
                    "ExplanationSuite requires an Explanation instance."
                )
            current[name] = explanation

        self.explanations = current
        return self.explanations

    def compare(self, *, allow_incommensurate: bool = False) -> None:
        """
        Print attribution values side-by-side only under an honest contract.

        By default, two or more outputs require equal explicit
        ``metadata['comparison_contract']`` values, equal explained targets,
        and identical ordered feature identities. The comparison contract is
        a caller assertion; built-in explainers do not currently emit one.
        Passing
        ``allow_incommensurate=True`` opts into a descriptive display and emits
        a warning; it does not make unlike quantities scientifically comparable.
        """
        if not isinstance(allow_incommensurate, bool):
            raise TypeError("allow_incommensurate must be a boolean.")
        if not self.explanations:
            print("No explanations to compare. Run suite.run(instance) first.")
            return

        # Collect and validate the feature identity of every explanation. A
        # feature-name sequence is authoritative when supplied; otherwise the
        # insertion order of the attribution mapping defines the identity.
        all_keys: set[str] = set()
        feature_identities: Dict[str, Tuple[str, ...]] = {}
        for name, explanation in self.explanations.items():
            attrs = explanation.explanation_data.get("feature_attributions")
            if not isinstance(attrs, Mapping):
                raise ValueError(
                    "compare() requires every result to contain a " "feature_attributions mapping."
                )
            if not attrs:
                raise ValueError(
                    f"Explanation {name!r} must contain at least one feature attribution."
                )
            for key, value in attrs.items():
                if not isinstance(key, str):
                    raise TypeError("compare() requires string feature-attribution keys.")
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"Attribution {key!r} from {name!r} must be a real numeric scalar."
                    )
                if not np.isfinite(float(value)):
                    raise ValueError(f"Attribution {key!r} from {name!r} must be finite.")
                all_keys.add(key)
            if explanation.feature_names is not None:
                names = tuple(explanation.feature_names)
                if len(names) != len(attrs) or set(names) != set(attrs):
                    raise ValueError(
                        f"Explanation {name!r} has feature_names that do not exactly "
                        "match its feature_attributions keys."
                    )
                feature_identities[name] = names
            else:
                feature_identities[name] = tuple(attrs.keys())

        if len(self.explanations) > 1:
            contracts: Dict[str, Any] = {
                name: explanation.metadata.get("comparison_contract")
                for name, explanation in self.explanations.items()
            }
            contracts_match = (
                all(
                    isinstance(contract, str) and bool(contract.strip())
                    for contract in contracts.values()
                )
                and len(set(contracts.values())) == 1
            )
            targets = {
                name: repr(explanation.target_class)
                for name, explanation in self.explanations.items()
            }
            targets_match = len(set(targets.values())) == 1
            features_match = len(set(feature_identities.values())) == 1
            if not contracts_match or not targets_match or not features_match:
                reasons = []
                if not contracts_match:
                    reasons.append("missing or unequal metadata['comparison_contract'] values")
                if not targets_match:
                    reasons.append(f"different explained targets {targets}")
                if not features_match:
                    reasons.append(f"different ordered feature identities {feature_identities}")
                message = (
                    "ExplanationSuite cannot establish mathematical comparability: "
                    + "; ".join(reasons)
                    + "."
                )
                if not allow_incommensurate:
                    raise ValueError(
                        message + " Pass allow_incommensurate=True only for a descriptive "
                        "side-by-side display."
                    )
                warnings.warn(
                    message + " Displaying values descriptively only.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        print("\nSide-by-Side Comparison:")
        print("-" * 60)

        # Header
        header = ["Feature"] + list(self.explanations.keys())
        print(" | ".join(f"{h:>15}" for h in header))
        print("-" * 60)

        # Rows
        for key in sorted(all_keys):
            row = [f"{key:>15}"]
            for name in self.explanations:
                value = (
                    self.explanations[name]
                    .explanation_data.get("feature_attributions", {})
                    .get(key, None)
                )
                if value is not None:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"Attribution {key!r} from {name!r} must be scalar numeric."
                        ) from exc
                    if not np.isfinite(numeric_value):
                        raise ValueError(f"Attribution {key!r} from {name!r} must be finite.")
                    row.append(f"{numeric_value:>15.4f}")
                else:
                    row.append(f"{'—':>15}")
            print(" | ".join(row))

    def suggest_compatible(
        self,
        *,
        include_statuses: Tuple[str, ...] = ("verified",),
        max_results: Optional[int] = None,
        for_instance_run: bool = True,
    ) -> List[str]:
        """List configured explainers compatible with explicit metadata.

        Compatibility is not a quality ranking. Results preserve configuration
        order. ``data_meta`` may specify ``scope``, ``model_type``, ``data_type``,
        and ``task``; omitted fields are not inferred.
        """
        valid_statuses = {"verified", "quarantined", "unverified"}
        if (
            not isinstance(include_statuses, tuple)
            or not include_statuses
            or any(status not in valid_statuses for status in include_statuses)
        ):
            raise ValueError(
                "include_statuses must be a non-empty tuple containing only "
                "verified, quarantined, or unverified."
            )
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, Integral):
                raise TypeError("max_results must be a non-negative integer or None.")
            if max_results < 0:
                raise ValueError("max_results must be non-negative.")
        if not isinstance(for_instance_run, bool):
            raise TypeError("for_instance_run must be a boolean.")

        # Validate caller metadata before status or run-contract filtering so a
        # typo cannot be hidden merely because no configured entry reaches
        # ``meta.matches`` below.
        from explainiverse.core.registry import ExplainerMeta

        ExplainerMeta._validate_match_criteria(
            scope=self.data_meta.get("scope"),
            model_type=self.data_meta.get("model_type"),
            data_type=self.data_meta.get("data_type"),
            task_type=self.data_meta.get("task"),
        )

        registry = self._get_registry()
        compatible: List[str] = []
        for name, params in self.configs:
            meta = registry.get_meta(name)
            if meta.claim_status not in include_statuses:
                continue
            if for_instance_run and not self._configured_for_instance_run(registry, name, params):
                continue
            if meta.matches(
                scope=self.data_meta.get("scope"),
                model_type=self.data_meta.get("model_type"),
                data_type=self.data_meta.get("data_type"),
                task_type=self.data_meta.get("task"),
            ):
                compatible.append(name)
        return compatible if max_results is None else compatible[: int(max_results)]

    def suggest_best(self) -> str:
        """Return the first verified metadata-compatible configured explainer.

        The historical method name does not identify a scientifically "best"
        explainer. Use :meth:`suggest_compatible` and choose with a validated,
        task-specific evaluation protocol.
        """
        warnings.warn(
            "suggest_best cannot establish explainer quality; it returns the first "
            "verified metadata-compatible configured explainer. Use "
            "suggest_compatible() and a validated task-specific evaluation.",
            FutureWarning,
            stacklevel=2,
        )
        candidates = self.suggest_compatible(max_results=1)
        if not candidates:
            raise ValueError("No configured explainer is both metadata-compatible and verified.")
        return candidates[0]

    def evaluate_roar(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        top_k: float = 0.1,
        model_class=None,
        model_kwargs: Optional[Dict] = None,
        *,
        baseline_value="mean",
        n_repeats: int = 5,
        random_state: Optional[int] = 0,
        task: Optional[str] = None,
        scoring=None,
        scoring_greater_is_better: Optional[bool] = None,
        ranking: str = "descending",
    ) -> Dict[str, float]:
        """
        Evaluate each explainer using ROAR (Remove And Retrain).

        ROAR retrains fresh models after removing each row's top-ranked
        features. Both train and held-out test rows receive their own
        explanation, as required by the ROAR protocol. One score drop is
        descriptive only; a method-quality claim requires a removal curve and
        an aligned random-ranking control.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            top_k: Feature fraction in (0, 1] or positive feature count.
            model_class: Estimator class, factory, or sklearn estimator.
                        If None, clones the wrapped raw model configuration.
            model_kwargs: Optional keyword args for new model instance
            baseline_value: Training-derived replacement rule or scalar.
            n_repeats: Number of paired clean/masked retraining repetitions.
            random_state: Starting seed for paired repetitions.
            task: Optional explicit ``classification`` or ``regression``.
            scoring: Optional supported scoring name or callable.
            scoring_greater_is_better: Required direction for a callable
                scorer; built-in accuracy and R2 are greater-is-better.
            ranking: ``descending`` or explicit ``absolute`` attribution order.

        Returns:
            Dict mapping explainer names to clean-minus-masked score drops.

        Raises:
            Any explanation, alignment, fitting, or scoring error. Failed rows
            are never skipped or converted to a numeric score.
        """
        from explainiverse.evaluation.metrics import compute_roar

        model_kwargs = model_kwargs or {}

        # A fitted sklearn estimator is accepted by compute_roar and cloned for
        # each repeat, preserving its hyperparameter configuration.
        if model_class is None:
            model_class = self.model.model if hasattr(self.model, "model") else self.model

        roar_scores: Dict[str, float] = {}
        registry = self._get_registry()

        for name, params in self.configs:
            self._require_local_scope(registry, name)
            explainer = self._create_configured_explainer(registry, name, params)
            call_kwargs = self._merged_call_kwargs(name)
            train_explanations = [
                self._explain_instance(explainer, name, row, dict(call_kwargs))
                for row in np.asarray(X_train)
            ]
            test_explanations = [
                self._explain_instance(explainer, name, row, dict(call_kwargs))
                for row in np.asarray(X_test)
            ]
            result = compute_roar(
                model_class=model_class,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                explanations=train_explanations,
                test_explanations=test_explanations,
                top_k=top_k,
                baseline_value=baseline_value,
                model_kwargs=model_kwargs,
                n_repeats=n_repeats,
                random_state=random_state,
                task=task,
                scoring=scoring,
                scoring_greater_is_better=scoring_greater_is_better,
                ranking=ranking,
            )
            if isinstance(result, bool) or not isinstance(result, Real):
                raise TypeError("compute_roar returned details where a scalar score was required.")
            roar_scores[name] = float(result)

        return roar_scores

    def get_explanation(self, name: str):
        """
        Get a specific explanation by explainer name.

        Args:
            name: Name of the explainer

        Returns:
            Explanation object or None if not found
        """
        return self.explanations.get(name)

    def list_explainers(self) -> List[str]:
        """
        List all configured explainer names.

        Returns:
            List of explainer names
        """
        return [name for name, _ in self.configs]

    def list_completed(self) -> List[str]:
        """
        List explainers that have been run successfully.

        Returns:
            List of explainer names with results
        """
        return list(self.explanations.keys())
