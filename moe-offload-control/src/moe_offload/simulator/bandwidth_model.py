"""Bandwidth model — computes transfer cost and stall for fetching experts.

Extreme-minimal implementation: linear model with a per-step capacity cap.
No multi-stream, no overlap, no queueing.
"""

from __future__ import annotations


class BandwidthModel:
    """Models the host-to-GPU bandwidth constraint.

    Parameters
    ----------
    capacity : int
        Maximum number of experts that can be transferred per step
        without any stall.
    per_expert_latency : float
        Stall penalty (in abstract time units) for each expert that
        exceeds the per-step bandwidth capacity.
    """

    def __init__(
        self,
        capacity: int,
        per_expert_latency: float = 1.0,
    ) -> None:
        self._capacity = capacity
        self._per_expert_latency = per_expert_latency

    @property
    def capacity(self) -> int:
        return self._capacity

    def compute_transfer(self, n_fetch: int) -> tuple[float, float]:
        """Compute transfer cost and stall for fetching *n_fetch* experts.

        Returns
        -------
        transfer_cost : float
            Total transfer cost (= *n_fetch*).
        stall : float
            Latency stall due to bandwidth overflow.  Zero if
            ``n_fetch <= capacity``.
        """
        transfer_cost = float(n_fetch)
        if n_fetch <= self._capacity:
            stall = 0.0
        else:
            overflow = n_fetch - self._capacity
            stall = overflow * self._per_expert_latency
        return transfer_cost, stall
