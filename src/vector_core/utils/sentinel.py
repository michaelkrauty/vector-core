"""Sentinel value for distinguishing 'not provided' from None.

This module provides a type-safe sentinel pattern for optional parameters
where None is a valid value distinct from "not provided".

Usage:
    from vector_core.utils.sentinel import UNSET, UnsetType, is_set

    def update(
        name: str | None | UnsetType = UNSET,
        age: int | None | UnsetType = UNSET,
    ) -> None:
        if is_set(name):
            # name was explicitly provided (could be None or a string)
            self.name = name
        # else: name was not provided, don't update

Example:
    update(name="Alice")      # Updates name to "Alice"
    update(name=None)         # Updates name to None (clears it)
    update()                  # Does not update name at all
"""

from typing import Final, TypeGuard, TypeVar

T = TypeVar("T")


class UnsetType:
    """Sentinel type for 'not provided' values.

    This is a singleton - use the UNSET instance, don't create new ones.
    """

    _instance: "UnsetType | None" = None

    def __new__(cls) -> "UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        # UNSET is falsy, which is intuitive for conditionals
        return False

    def __copy__(self) -> "UnsetType":
        return self

    def __deepcopy__(self, memo: dict) -> "UnsetType":
        return self


# The singleton instance
UNSET: Final[UnsetType] = UnsetType()


def is_set(value: T | UnsetType) -> TypeGuard[T]:
    """Check if a value was explicitly provided (is not UNSET).

    This is a type guard that narrows the type from `T | UnsetType` to `T`.

    Args:
        value: The value to check

    Returns:
        True if value is not UNSET (was explicitly provided)

    Example:
        def update(name: str | None | UnsetType = UNSET):
            if is_set(name):
                # Here, name is narrowed to `str | None`
                self.name = name
    """
    return value is not UNSET
