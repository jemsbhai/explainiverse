"""Fail-closed local operator boundary for the Lambda CUDA release controller.

The package initializer deliberately imports nothing.  The supported pinned
``-I -S -B -c <reviewed-shim>`` preloader verifies the interpreter, source, and
resource boundary before importing any controller module.
"""

__all__: list[str] = []
