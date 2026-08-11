# src/explainiverse/core/explanation.py
"""Validated container for explanation results."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Optional

EXPLANATION_WIRE_SCHEMA_VERSION = "explainiverse.explanation.v1"
_WIRE_FIELDS = frozenset(
    {
        "schema_version",
        "explainer_name",
        "target_class",
        "explanation_data",
        "feature_names",
        "metadata",
    }
)
_JS_MIN_SAFE_INTEGER = -(2**53 - 1)
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def _clone_wire_value(value: Any, path: str, active: set[int]) -> Any:
    """Validate and detach the exact finite JSON subset shared with JavaScript."""

    if value is None or type(value) in {str, bool}:
        return value
    if type(value) is int:
        if not _JS_MIN_SAFE_INTEGER <= value <= _JS_MAX_SAFE_INTEGER:
            raise ValueError(
                f"{path} integer must be within JavaScript's safe-integer range "
                "for exact transport"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(
                f"{path} must not contain negative zero because JSON transport erases its sign"
            )
        if value.is_integer() and not _JS_MIN_SAFE_INTEGER <= value <= _JS_MAX_SAFE_INTEGER:
            raise ValueError(
                f"{path} integer must be within JavaScript's safe-integer range "
                "for exact transport"
            )
        return value
    if type(value) not in {list, dict}:
        raise TypeError(
            f"{path} must contain only JSON null/boolean/string/number values, "
            "ordinary lists, and plain string-keyed dictionaries"
        )

    identity = id(value)
    if identity in active:
        raise TypeError(f"{path} must not contain cyclic references")
    active.add(identity)
    try:
        if isinstance(value, list):
            return [
                _clone_wire_value(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            ]
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} object keys must be strings")
            result[key] = _clone_wire_value(item, f"{path}.{key}", active)
        return result
    finally:
        active.remove(identity)


class Explanation:
    """
    Unified container for explanation results.

    Attributes:
        explainer_name: Name of the explainer that generated this explanation
        target_class: The class/output being explained
        explanation_data: Dictionary containing explanation details
            (e.g., feature_attributions, heatmaps, rules)
        feature_names: Optional list of feature names for index resolution
        metadata: Optional additional metadata about the explanation

    Example:
        >>> explanation = Explanation(
        ...     explainer_name="LIME",
        ...     target_class="cat",
        ...     explanation_data={"feature_attributions": {"fur": 0.8, "whiskers": 0.6}},
        ...     feature_names=["fur", "whiskers", "tail", "ears"]
        ... )
        >>> print(explanation.get_top_features(k=2))
        [('fur', 0.8), ('whiskers', 0.6)]
    """

    def __init__(
        self,
        explainer_name: str,
        target_class: Any,
        explanation_data: Mapping[str, Any],
        feature_names: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Initialize an Explanation object.

        Args:
            explainer_name: Name of the explainer (e.g., "LIME", "SHAP")
            target_class: The target class or output being explained
            explanation_data: Dictionary containing the explanation details.
                Common keys include:
                - "feature_attributions": Dict[str, float] mapping names to attribution values
                - "attributions_raw": List[float] of raw attribution values
                - "heatmap": np.ndarray for image explanations
                - "rules": List of rule strings for rule-based explanations
            feature_names: Optional list of feature names. If provided, enables
                index-based lookup in evaluation metrics.
            metadata: Optional additional metadata (e.g., computation time, parameters)
        """
        if not isinstance(explainer_name, str) or not explainer_name.strip():
            raise ValueError("explainer_name must be a non-empty string")
        if not isinstance(explanation_data, Mapping):
            raise TypeError("explanation_data must be a mapping")
        if feature_names is not None:
            if isinstance(feature_names, (str, bytes)) or not isinstance(feature_names, Sequence):
                raise TypeError("feature_names must be a sequence of strings or None")
            if any(not isinstance(name, str) or not name.strip() for name in feature_names):
                raise ValueError("feature_names must contain non-empty strings")
            if len(feature_names) != len(set(feature_names)):
                raise ValueError("feature_names must be unique")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        self.explainer_name = explainer_name
        self.target_class = target_class
        self.explanation_data = copy.deepcopy(dict(explanation_data))
        self.feature_names = (
            copy.deepcopy(list(feature_names)) if feature_names is not None else None
        )
        self.metadata = copy.deepcopy(dict(metadata)) if metadata is not None else {}

    def __repr__(self) -> str:
        n_features = len(self.feature_names) if self.feature_names else "N/A"
        return (
            f"Explanation(explainer='{self.explainer_name}', "
            f"target='{self.target_class}', "
            f"keys={list(self.explanation_data.keys())}, "
            f"n_features={n_features})"
        )

    def get_attributions(self) -> Optional[dict[str, Any]]:
        """
        Get feature attributions if available.

        Returns:
            Dictionary mapping feature names to attribution values,
            or None if not available.
        """
        attributions = self.explanation_data.get("feature_attributions")
        if attributions is None:
            return None
        if not isinstance(attributions, Mapping):
            raise TypeError("feature_attributions must be a mapping when present")
        return dict(attributions)

    def get_top_features(self, k: int = 5, absolute: bool = True) -> list[tuple[str, float]]:
        """
        Get the features with the largest attribution values.

        Args:
            k: Number of top features to return
            absolute: If True, rank by absolute value of attribution

        Returns:
            List of ``(feature_name, attribution_value)`` tuples sorted by the
            selected attribution ordering.
        """
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(absolute, bool):
            raise TypeError("absolute must be a boolean")

        attributions = self.get_attributions()
        if attributions is None or not attributions:
            return []

        validated: list[tuple[str, float]] = []
        for feature_name, value in attributions.items():
            if not isinstance(feature_name, str) or not feature_name:
                raise ValueError("attribution keys must be non-empty feature names")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("attribution values must be real numeric scalars")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError("attribution values must be finite")
            validated.append((feature_name, numeric_value))

        if absolute:
            sorted_items = sorted(
                validated,
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        else:
            sorted_items = sorted(
                validated,
                key=lambda item: item[1],
                reverse=True,
            )

        return sorted_items[:k]

    def get_feature_index(self, feature_name: str) -> Optional[int]:
        """
        Get the index of a feature by name.

        Args:
            feature_name: Name of the feature

        Returns:
            Index of the feature, or None if not found or feature_names not set
        """
        if self.feature_names is None:
            return None
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            return None

    def plot(self, plot_type: str = "bar", **kwargs: Any) -> None:
        """
        Visualize the explanation.

        Args:
            plot_type: Type of plot ('bar', 'waterfall', 'heatmap')
            **kwargs: Additional arguments passed to the plotting function

        Raises:
            NotImplementedError: Always. Explainiverse does not currently provide a
                plotting backend for generic explanation payloads.
        """
        del plot_type, kwargs
        raise NotImplementedError(
            "Explanation.plot() has no implemented plotting backend; inspect "
            "explanation_data or use a payload-specific plotting library"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a detached dictionary representation.

        Payload values retain their original Python/NumPy types. The returned
        mapping is therefore not guaranteed to be directly JSON serializable.

        Returns:
            Defensive-copy dictionary representation of the explanation
        """
        return copy.deepcopy(
            {
                "explainer_name": self.explainer_name,
                "target_class": self.target_class,
                "explanation_data": self.explanation_data,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            }
        )

    def to_wire_dict(self) -> dict[str, Any]:
        """Return the versioned, finite-JSON Python/JavaScript wire payload.

        Unlike :meth:`to_dict`, this opt-in endpoint rejects NumPy objects,
        non-finite numbers, negative zero, unsafe integer-valued numbers,
        non-string object keys, custom containers, and cyclic values. It never
        silently coerces a broader Python payload into a lossy JSON representation.
        """

        if not isinstance(self.target_class, str) or not self.target_class.strip():
            raise ValueError(
                "Explanation wire v1 target_class must be a non-empty string; "
                "legacy to_dict() retains broader Python target values"
            )
        payload = {
            "schema_version": EXPLANATION_WIRE_SCHEMA_VERSION,
            "explainer_name": self.explainer_name,
            "target_class": self.target_class,
            "explanation_data": self.explanation_data,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
        cloned = _clone_wire_value(payload, "payload", set())
        if not isinstance(cloned, dict):  # pragma: no cover - fixed construction
            raise RuntimeError("wire payload validation did not return an object")
        return cloned

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Explanation":
        """
        Create an Explanation from a dictionary.

        Args:
            data: Dictionary with explanation data

        Returns:
            Explanation instance
        """
        if not isinstance(data, Mapping):
            raise TypeError("data must be a mapping")
        missing = {
            "explainer_name",
            "target_class",
            "explanation_data",
        }.difference(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"data is missing required field(s): {missing_text}")

        return cls(
            explainer_name=data["explainer_name"],
            target_class=data["target_class"],
            explanation_data=data["explanation_data"],
            feature_names=data.get("feature_names"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_wire_dict(cls, data: Any) -> "Explanation":
        """Construct an Explanation from an untrusted versioned wire payload."""

        cloned = _clone_wire_value(data, "payload", set())
        if not isinstance(cloned, dict):
            raise TypeError("wire payload must be a plain JSON object")
        fields = set(cloned)
        missing = _WIRE_FIELDS - fields
        unknown = fields - _WIRE_FIELDS
        if missing:
            raise ValueError(
                "wire payload is missing required field(s): " + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "wire payload contains unknown field(s): " + ", ".join(sorted(unknown))
            )
        if cloned["schema_version"] != EXPLANATION_WIRE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported Explanation wire schema_version; expected "
                f"{EXPLANATION_WIRE_SCHEMA_VERSION!r}"
            )
        if not isinstance(cloned["target_class"], str) or not cloned["target_class"].strip():
            raise ValueError("wire target_class must be a non-empty string")
        if not isinstance(cloned["explanation_data"], dict):
            raise TypeError("wire explanation_data must be a plain JSON object")
        if cloned["feature_names"] is not None and not isinstance(cloned["feature_names"], list):
            raise TypeError("wire feature_names must be an array or null")
        if not isinstance(cloned["metadata"], dict):
            raise TypeError("wire metadata must be a plain JSON object")
        return cls(
            explainer_name=cloned["explainer_name"],
            target_class=cloned["target_class"],
            explanation_data=cloned["explanation_data"],
            feature_names=cloned["feature_names"],
            metadata=cloned["metadata"],
        )
