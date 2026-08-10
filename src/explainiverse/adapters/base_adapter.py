from abc import ABC, abstractmethod


class BaseModelAdapter(ABC):
    """Minimal interface shared by Explainiverse model adapters.

    The base class stores a model reference and optional feature labels, and
    requires a concrete ``predict`` method.  It does not reshape prediction
    inputs, infer whether an input is one sample or a batch, select a target,
    normalize outputs, or enforce an output type or shape.  Those semantics
    depend on the wrapped model and are documented by each concrete adapter.
    """

    def __init__(self, model, feature_names=None):
        """Store an adapter's model and validated feature-name metadata.

        Args:
            model: The model object to wrap.  The exact object is retained by
                reference.
            feature_names: Optional iterable of unique, non-empty strings.  A
                new list is stored so later changes to the caller's container
                cannot alter the adapter's metadata.

        Raises:
            ValueError: If ``model`` is ``None`` or a feature name is empty or
                duplicated.
            TypeError: If ``feature_names`` is not an iterable of strings.
        """
        if model is None:
            raise ValueError("model must not be None")

        if feature_names is None:
            names = None
        else:
            if isinstance(feature_names, (str, bytes)):
                raise TypeError("feature_names must be an iterable of strings or None")
            try:
                names = list(feature_names)
            except TypeError as exc:
                raise TypeError("feature_names must be an iterable of strings or None") from exc
            if any(not isinstance(name, str) for name in names):
                raise TypeError("feature_names must contain only strings")
            if any(not name for name in names):
                raise ValueError("feature_names must contain non-empty strings")
            if len(set(names)) != len(names):
                raise ValueError("feature_names must be unique")

        self.model = model
        self.feature_names = names

    @abstractmethod
    def predict(self, data):
        """Run prediction according to the concrete adapter's contract.

        Args:
            data: Adapter-specific model input.

        Raises:
            NotImplementedError: If an override delegates to this abstract
                fallback instead of providing an implementation.

        Notes:
            This base method makes no claim that ``data`` may be a single
            sample, a batch, or both, and does not prescribe target selection
            or a universal output representation.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement its own predict() method")
