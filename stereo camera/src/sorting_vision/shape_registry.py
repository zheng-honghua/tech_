from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ShapeClass:
    class_id: str
    name_zh: str
    family: str
    aliases: tuple[str, ...] = ()
    enabled: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ShapeClass":
        return cls(
            class_id=str(value["id"]).strip(),
            name_zh=str(value.get("name_zh", value["id"])).strip(),
            family=str(value.get("family", "other")).strip(),
            aliases=tuple(str(item).strip() for item in value.get("aliases", ())),
            enabled=bool(value.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.class_id,
            "name_zh": self.name_zh,
            "family": self.family,
            "aliases": list(self.aliases),
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ShapeRegistry:
    version: int
    classes: tuple[ShapeClass, ...]

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.classes:
            raise ValueError("shape registry needs a positive version and at least one class")
        identifiers: set[str] = set()
        aliases: dict[str, str] = {}
        for item in self.classes:
            if not item.class_id or item.class_id in identifiers:
                raise ValueError(f"duplicate or empty shape id: {item.class_id!r}")
            identifiers.add(item.class_id)
            for alias in (item.class_id, item.name_zh, *item.aliases):
                key = self._key(alias)
                owner = aliases.get(key)
                if owner is not None and owner != item.class_id:
                    raise ValueError(
                        f"shape alias {alias!r} belongs to both {owner!r} and {item.class_id!r}"
                    )
                aliases[key] = item.class_id

    @staticmethod
    def _key(value: str) -> str:
        return "".join(str(value).strip().lower().replace("-", "_").split())

    @property
    def enabled_classes(self) -> tuple[ShapeClass, ...]:
        return tuple(item for item in self.classes if item.enabled)

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(item.class_id for item in self.enabled_classes)

    @property
    def names(self) -> dict[str, str]:
        return {item.class_id: item.name_zh for item in self.enabled_classes}

    def resolve(self, value: str, *, require_enabled: bool = True) -> str:
        key = self._key(value)
        for item in self.classes:
            if key in {self._key(alias) for alias in (item.class_id, item.name_zh, *item.aliases)}:
                if require_enabled and not item.enabled:
                    raise ValueError(f"shape class is disabled: {item.class_id}")
                return item.class_id
        raise ValueError(f"unknown shape class or alias: {value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "version": self.version,
            "classes": [item.to_dict() for item in self.classes],
        }

    @property
    def registry_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "ShapeRegistry":
        source = Path(path)
        if source.suffix.lower() == ".json":
            value = json.loads(source.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("classes"), list):
            raise ValueError("shape registry must contain a classes list")
        return cls(
            version=int(value.get("version", value.get("schema_version", 1))),
            classes=tuple(ShapeClass.from_dict(item) for item in value["classes"]),
        )

    def validate_model_classes(
        self, class_ids: Iterable[str], registry_hash: str
    ) -> None:
        if str(registry_hash) != self.registry_hash:
            raise ValueError("shape registry hash mismatch")
        actual = tuple(map(str, class_ids))
        if actual != self.class_ids:
            raise ValueError(
                f"model class order mismatch: expected {self.class_ids}, got {actual}"
            )


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "shapes" / "competition-11.yaml"


def load_shape_registry(path: str | Path | None = None) -> ShapeRegistry:
    return ShapeRegistry.load(default_registry_path() if path is None else path)
