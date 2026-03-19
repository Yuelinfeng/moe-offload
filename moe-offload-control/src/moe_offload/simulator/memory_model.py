"""Memory model — tracks expert slot occupancy in GPU memory.

Extreme-minimal implementation: each expert occupies exactly one slot.
No multi-level cache, no per-expert size differences.
"""

from __future__ import annotations


class MemoryModel:
    """Tracks which experts are resident in GPU memory.

    Parameters
    ----------
    capacity : int
        Maximum number of expert slots available.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._residents: set[int] = set()

    # -- queries --

    @property
    def capacity(self) -> int:
        return self._capacity

    def current_usage(self) -> int:
        """Number of slots currently occupied."""
        return len(self._residents)

    def can_fit(self, n: int) -> bool:
        """Return True if *n* additional experts can fit."""
        return self.current_usage() + n <= self._capacity

    def is_resident(self, expert_id: int) -> bool:
        return expert_id in self._residents

    def resident_set(self) -> set[int]:
        """Return a *copy* of the current resident set."""
        return set(self._residents)

    # -- mutations --

    def allocate(self, expert_ids: list[int]) -> None:
        """Add *expert_ids* to the resident set.

        Raises :class:`ValueError` if capacity would be exceeded or if any
        expert is already resident.
        """
        new = [eid for eid in expert_ids if eid not in self._residents]
        if self.current_usage() + len(new) > self._capacity:
            raise ValueError(
                f"Cannot allocate {len(new)} experts: "
                f"usage={self.current_usage()}, capacity={self._capacity}"
            )
        self._residents.update(new)

    def release(self, expert_ids: list[int]) -> None:
        """Remove *expert_ids* from the resident set.

        Silently ignores experts that are not resident.
        """
        self._residents -= set(expert_ids)

    def reset(self, initial_residents: set[int] | None = None) -> None:
        """Clear all residents and optionally set initial ones."""
        self._residents = set(initial_residents) if initial_residents else set()
