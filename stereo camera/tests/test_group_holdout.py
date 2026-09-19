from dataclasses import dataclass

import pytest

from sorting_vision.group_holdout import deterministic_group_holdout


@dataclass(frozen=True)
class Item:
    group: str
    identifier: str


def test_group_holdout_is_deterministic_and_keeps_three_per_group():
    items = [
        Item(group, f"{group}-{index:02d}")
        for group in ("red/triangle", "blue/cone")
        for index in range(15)
    ]
    first = deterministic_group_holdout(
        items,
        group_key=lambda item: item.group,
        item_key=lambda item: item.identifier,
        holdout_per_group=3,
        seed=23,
    )
    second = deterministic_group_holdout(
        reversed(items),
        group_key=lambda item: item.group,
        item_key=lambda item: item.identifier,
        holdout_per_group=3,
        seed=23,
    )
    assert {item.identifier for item in first.holdout} == {
        item.identifier for item in second.holdout
    }
    assert len(first.training) == 24
    assert len(first.holdout) == 6
    assert all(value == {"total": 15, "training": 12, "holdout": 3} for value in first.group_counts.values())
    assert not ({item.identifier for item in first.training} & {item.identifier for item in first.holdout})


def test_group_holdout_rejects_a_group_that_would_have_no_training_item():
    items = [Item("only", str(index)) for index in range(3)]
    with pytest.raises(ValueError, match="needs at least 4"):
        deterministic_group_holdout(
            items,
            group_key=lambda item: item.group,
            item_key=lambda item: item.identifier,
            holdout_per_group=3,
        )

