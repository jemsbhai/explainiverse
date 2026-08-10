from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """Minimal interface shared by Explainiverse explainers.

    Explanation methods do not have one universal input signature: local
    explainers commonly accept an instance, while global and example-based
    explainers may accept features, datasets, or no call-time input.  Target
    selection and batching are likewise optional, method-specific
    capabilities.  Concrete explainers must therefore document their own
    arguments and return contract.

    This abstract base class only requires an ``explain`` implementation and
    stores the supplied model reference.  It does not copy or configure that
    object, validate explanation inputs or outputs, or provide an implicit
    batch implementation.
    """

    def __init__(self, model):
        """Store the model object used by the concrete explainer.

        Args:
            model: A model adapter, a raw model, or ``None`` for an explainer
                whose operation is independent of model predictions.  The
                object is retained by reference.
        """
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs):
        """Generate an explanation using the concrete method's contract.

        Args:
            *args: Method-specific positional arguments.
            **kwargs: Method-specific keyword arguments.

        Raises:
            NotImplementedError: If an override delegates to this abstract
                fallback instead of providing an implementation.

        Notes:
            ``ABC`` verifies that a concrete subclass overrides this method;
            it does not enforce the override's signature or return type at
            runtime.  Built-in explainers document those details individually.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement its own explain() method")
