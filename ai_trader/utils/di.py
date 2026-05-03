"""Lightweight dependency injection container."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Container:
    """Simple DI container for wiring dependencies across modules.

    Supports singleton and factory registrations.
    """

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable[..., Any]] = {}

    def register_singleton(self, key: str, instance: Any) -> None:
        """Register a pre-built instance."""
        self._singletons[key] = instance

    def register_factory(self, key: str, factory: Callable[..., Any]) -> None:
        """Register a factory callable that creates instances on demand."""
        self._factories[key] = factory

    def resolve(self, key: str) -> Any:
        """Resolve a dependency by key. Singletons take priority over factories."""
        if key in self._singletons:
            return self._singletons[key]
        if key in self._factories:
            instance = self._factories[key]()
            self._singletons[key] = instance
            return instance
        raise KeyError(f"No dependency registered for key: {key}")

    def has(self, key: str) -> bool:
        return key in self._singletons or key in self._factories

    def reset(self) -> None:
        """Clear all registrations (useful for testing)."""
        self._singletons.clear()
        self._factories.clear()
