from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class GroupHoldoutSplit(Generic[T]):
    """Deterministic train/holdout partition with an equal holdout per group."""

    training: tuple[T, ...]
    holdout: tuple[T, ...]
    group_counts: dict[str, dict[str, int]]


def deterministic_group_holdout(
    items: Iterable[T],
    *,
    group_key: Callable[[T], str],
    item_key: Callable[[T], str],
    holdout_per_group: int = 3,
    seed: int = 20260915,
) -> GroupHoldoutSplit[T]:
    """Select a stable pseudo-random holdout without changing source records.

    Hash ranking makes the result independent of filesystem traversal order while
    retaining the user-requested random selection semantics and a recorded seed.
    Every group must leave at least one item for training.
    """

    if holdout_per_group <= 0:
        raise ValueError("holdout_per_group must be positive")
    grouped: dict[str, list[T]] = defaultdict(list)
    seen_ids: set[str] = set()
    for item in items:
        identifier = item_key(item)
        if identifier in seen_ids:
            raise ValueError(f"duplicate holdout item key: {identifier}")
        seen_ids.add(identifier)
        grouped[group_key(item)].append(item)
    if not grouped:
        raise ValueError("group holdout needs at least one sample")

    training: list[T] = []
    holdout: list[T] = []
    counts: dict[str, dict[str, int]] = {}
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) <= holdout_per_group:
            raise ValueError(
                f"group {key!r} has {len(group)} samples; it needs at least "
                f"{holdout_per_group + 1}"
            )

        def rank(item: T) -> tuple[str, str]:
            identifier = item_key(item)
            payload = f"{seed}\0{key}\0{identifier}".encode("utf-8")
            return hashlib.sha256(payload).hexdigest(), identifier

        ranked = sorted(group, key=rank)
        selected = ranked[:holdout_per_group]
        selected_ids = {item_key(item) for item in selected}
        holdout.extend(selected)
        training.extend(item for item in group if item_key(item) not in selected_ids)
        counts[key] = {
            "total": len(group),
            "training": len(group) - holdout_per_group,
            "holdout": holdout_per_group,
        }
    return GroupHoldoutSplit(tuple(training), tuple(holdout), counts)

