from abc import ABC, abstractmethod
from functools import wraps
from threading import RLock
from typing import Callable, TypeVar, cast

_Method = TypeVar("_Method", bound=Callable)


def synchronized_explainer_method(method: _Method) -> _Method:
    """Serialize one public operation on a shared explainer instance.

    Built-in explainers use this decorator around operations that read or
    mutate persistent configuration.  The lock is re-entrant because public
    convenience methods commonly delegate to another synchronized method.
    """

    if getattr(method, "__explainiverse_instance_synchronized__", False):
        return method

    @wraps(method)
    def synchronized(self, *args, **kwargs):
        lock = getattr(self, "_instance_lock", None)
        if lock is None:  # Defensive support for a subclass that skipped ``super``.
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)

    setattr(synchronized, "__explainiverse_instance_synchronized__", True)
    return cast(_Method, synchronized)


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

    def __init_subclass__(cls, **kwargs):
        """Serialize every public instance operation declared by a subclass.

        Persistent state is not limited to gradient explainers: built-ins also
        retain RNGs, fitted backend objects, background data, concept stores,
        and cached configuration. Wrapping at the common class boundary keeps
        all public operations on one instance in a single re-entrant schedule.
        Private helpers execute under their public caller's lock. Static and
        class methods have no explainer instance and remain outside this
        ownership contract.
        """

        super().__init_subclass__(**kwargs)
        for name, attribute in tuple(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            if isinstance(attribute, (staticmethod, classmethod, property)):
                continue
            if callable(attribute):
                setattr(cls, name, synchronized_explainer_method(attribute))

    def __init__(self, model):
        """Store the model object used by the concrete explainer.

        Args:
            model: A model adapter, a raw model, or ``None`` for an explainer
                whose operation is independent of model predictions.  The
                object is retained by reference.
        """
        self.model = model
        # Persistent explainer fields are independent of the wrapped model's
        # state.  Model-level locking therefore cannot make a shared explainer
        # instance atomic. ``__init_subclass__`` enrolls every public instance
        # operation in this re-entrant schedule.
        self._instance_lock = RLock()

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
